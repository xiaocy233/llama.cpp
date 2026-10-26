#include "moe-gate.cuh"

#include "ggml-backend.h"

#include <atomic>
#include <thread>

// A park is 1.5 ms at worst when the worker is healthy, so this is 30x the measured range: it can only
// fire if the worker is stuck, and the gate then aborts the run rather than pick another expert. Well
// under the 2 s WDDM TDR either way.
#define MOE_GATE_TIMEOUT_NS 50000000ull

static __device__ __forceinline__ uint64_t moe_gate_now_ns() {
    uint64_t v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}

static __device__ __forceinline__ void moe_gate_relax() {
#if __CUDA_ARCH__ >= 700
    __nanosleep(128);
#else
    __threadfence_block();
#endif
}

// loc_map is deliberately not const or __restrict__: the host rewrites it while this kernel parks,
// so every read of it must go to memory. Both reads below go through this.
static __device__ __forceinline__ int32_t moe_gate_loc(const int32_t * loc_map, int32_t e) {
    return ((volatile const int32_t *) loc_map)[e];
}

template <bool publish, bool resolve>
static __global__ void moe_gate_kernel(
        const int32_t * loc_map,
        const int32_t * __restrict__ ids_in,
        const int32_t * __restrict__ seq_p,
        ggml_moe_gate_entry * __restrict__ ring,
        volatile uint32_t * __restrict__ release,
        volatile int32_t * __restrict__ timeout,
        int32_t * __restrict__ out,
        const float * __restrict__ probs,
        const int n_ids, const int n_pred, const int layer, const int n_slots,
        const int n_expert, const float threshold, const bool hybrid) {
    const uint32_t seq = (uint32_t) *seq_p;
    const int      t   = threadIdx.x;

    ggml_moe_gate_entry * box = ring + layer;
    if (publish) {
        __shared__ uint32_t s_miss;
        if (t == 0) {
            s_miss = 0u;
        }
        __syncthreads();
        for (int i = t; i < n_ids; i += blockDim.x) {
            if (moe_gate_loc(loc_map, ids_in[i]) >= n_slots) {
                atomicOr(&s_miss, n_ids <= 32 ? (1u << i) : 1u);
            }
        }
        for (int i = t; i < n_ids + n_pred; i += blockDim.x) {
            box->ids[i] = ids_in[i];
        }
        if (probs) {
            for (int i = t; i < n_expert; i += blockDim.x) {
                box->router_probs[i] = probs[i];
            }
        }
        __syncthreads();
        if (t == 0) {
            box->layer     = (uint32_t) layer;
            box->miss_mask = s_miss;
            box->n_ids     = (uint32_t) n_ids;
            box->n_pred    = (uint32_t) n_pred;
            box->mode     = hybrid ? GGML_MOE_GATE_MODE_HYBRID : probs ? GGML_MOE_GATE_MODE_SUBSTITUTE : GGML_MOE_GATE_MODE_NORMAL;
            box->substitute_threshold = threshold;
            uint32_t hash = 0;
            if (probs) {
                hash = 2166136261u;
                for (int i = 0; i < n_ids + n_pred; i++) {
                    hash = (hash ^ (uint32_t) ids_in[i]) * 16777619u;
                }
                for (int i = 0; i < n_expert; i++) {
                    hash = (hash ^ __float_as_uint(probs[i])) * 16777619u;
                }
            }
            box->payload_hash = hash;
        }
        __threadfence_system();
        __syncthreads();
        if (t == 0) {
            box->seq = seq;
        }
    }

    if (!resolve) {
        return;
    }

    // A hit can still have a prefetch in flight or be evicted before the worker handles this gate.
    if (t == 0) {
        const uint64_t t0 = moe_gate_now_ns();
        while (release[layer] < seq) {
            if (moe_gate_now_ns() - t0 > MOE_GATE_TIMEOUT_NS) {
                atomicAdd((int32_t *) timeout + layer, 1);
                __trap();
            }
            moe_gate_relax();
        }
    }
    __syncthreads();
    __threadfence_system();

    // The worker selects substitutes and fills all remaining misses before release.
    for (int i = t; i < n_ids; i += blockDim.x) {
        if (hybrid) {
            out[i] = ((volatile const int32_t *) box->selected_ids)[i];
            continue;
        }
        const int32_t e = probs ? ((volatile const int32_t *) box->selected_ids)[i] : ids_in[i];
        int32_t       p = moe_gate_loc(loc_map, e);
        if (p >= n_slots) {
            const uint64_t t0 = moe_gate_now_ns();
            while (p >= n_slots) {
                if (moe_gate_now_ns() - t0 > MOE_GATE_TIMEOUT_NS) {
                    atomicAdd((int32_t *) timeout + layer, 1);
                    __trap();
                }
                moe_gate_relax();
                p = moe_gate_loc(loc_map, e);
            }
        }
        out[i] = p;
        if (probs) {
            out[n_ids + i] = e;
        }
    }
}

static __global__ void moe_head_kernel(
        const int32_t * ids, const int32_t * pred_ids, const int32_t * seq,
        ggml_moe_gate_entry * ring, int32_t * out, int layer, int n_ids, int n_pred) {
    ggml_moe_gate_entry * box = ring + layer;
    for (int i = threadIdx.x; i < n_pred; i += blockDim.x) {
        box->ids[i] = pred_ids[i];
    }
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        out[i] = ids[i];
    }
    if (threadIdx.x == 0) {
        box->layer = layer;
        box->miss_mask = 0;
        box->n_ids = 0;
        box->n_pred = n_pred;
        box->mode = GGML_MOE_GATE_MODE_HEAD;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        box->seq = (uint32_t) *seq;
    }
}

// Order the release after the copies on this stream.
static __global__ void moe_gate_release_kernel(volatile uint32_t * release, const int layer, const uint32_t seq) {
    __threadfence_system();   // the copies ahead of this must be visible to the gate first
    release[layer] = seq;
}

void ggml_cuda_moe_gate_release(ggml_backend_cuda_context & ctx, cudaStream_t stream, int layer, uint32_t seq) {
    const ggml_cuda_moe_gate_channel & ch = ctx.moe_gate_channel();

    GGML_ASSERT(layer < GGML_MOE_GATE_MAX_LAYERS);

    moe_gate_release_kernel<<<1, 1, 0, stream>>>(ch.release_dev, layer, seq);
}

void ggml_cuda_op_moe_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool capturing) {
    const ggml_tensor * loc_map = dst->src[0];
    const ggml_tensor * ids_in  = dst->src[1];
    const ggml_tensor * seq     = dst->src[2];

    const int layer   = ggml_get_op_params_i32(dst, 0);
    const int n_ids   = ggml_get_op_params_i32(dst, 1);
    const int n_pred  = ggml_get_op_params_i32(dst, 2);
    const int n_slots = ggml_get_op_params_i32(dst, 3);
    if (ggml_get_op_params_i32(dst, 4) == GGML_MOE_GATE_MODE_HEAD) {
        const ggml_cuda_moe_gate_channel & ch = ctx.moe_gate_channel();
        GGML_ASSERT(n_ids > 0 && n_ids <= GGML_MOE_GATE_MAX_IDS && n_pred <= GGML_MOE_GATE_MAX_IDS);
        GGML_ASSERT(layer >= 0 && layer < GGML_MOE_GATE_MAX_LAYERS);
        moe_head_kernel<<<1, 32, 0, ctx.stream()>>>(
                (const int32_t *) loc_map->data, (const int32_t *) ids_in->data,
                (const int32_t *) seq->data, ch.ring_dev, (int32_t *) dst->data, layer, n_ids, n_pred);
        return;
    }
    const bool hybrid = ggml_get_op_params_i32(dst, 4) == GGML_MOE_GATE_MODE_HYBRID;
    const bool substitute = ggml_get_op_params_i32(dst, 4) == GGML_MOE_GATE_MODE_SUBSTITUTE;
    const float * probs = substitute ? (const float *) dst->src[3]->data : nullptr;
    const float threshold = substitute ? ggml_get_op_params_f32(dst, 5) : 0.0f;
    const int n_expert = (int) ggml_nelements(loc_map);

    GGML_ASSERT(n_ids > 0 && (hybrid || n_ids <= n_slots) && n_ids + n_pred <= GGML_MOE_GATE_MAX_IDS);
    GGML_ASSERT(layer < GGML_MOE_GATE_MAX_LAYERS);

    const ggml_cuda_moe_gate_channel & ch = ctx.moe_gate_channel();
    if (capturing) {
        moe_gate_kernel<true, true><<<1, 32, 0, ctx.stream()>>>(
                (const int32_t *) loc_map->data, (const int32_t *) ids_in->data,
                (const int32_t *) seq->data, ch.ring_dev, ch.release_dev, ch.timeout_dev,
                (int32_t *) dst->data, probs, n_ids, n_pred, layer, n_slots, n_expert, threshold, hybrid);
    } else {
        // A parked kernel can block driver submission before capture. Wait on the host between kernels.
        moe_gate_kernel<true, false><<<1, 32, 0, ctx.stream()>>>(
                (const int32_t *) loc_map->data, (const int32_t *) ids_in->data,
                (const int32_t *) seq->data, ch.ring_dev, ch.release_dev, ch.timeout_dev,
                (int32_t *) dst->data, probs, n_ids, n_pred, layer, n_slots, n_expert, threshold, hybrid);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        const uint32_t token = ((volatile const ggml_moe_gate_entry *) ch.ring_host)[layer].seq;
        const int64_t start = ggml_time_us();
        while (((volatile const uint32_t *) ch.release_host)[layer] < token) {
            if (ggml_time_us() - start > 5000000) {
                GGML_ABORT("MoE gate timed out waiting for expert fills");
            }
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        moe_gate_kernel<false, true><<<1, 32, 0, ctx.stream()>>>(
                (const int32_t *) loc_map->data, (const int32_t *) ids_in->data,
                (const int32_t *) seq->data, ch.ring_dev, ch.release_dev, ch.timeout_dev,
                (int32_t *) dst->data, probs, n_ids, n_pred, layer, n_slots, n_expert, threshold, hybrid);
    }
}

static __global__ void moe_cpu_merge_kernel(float * dst, const float * values, const int32_t * rows, int width) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        dst[(int64_t) rows[row] * width + col] = values[(int64_t) row * width + col];
    }
}

void ggml_cuda_moe_cpu_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * values, const int32_t * rows, int count) {
    GGML_ASSERT(count > 0 && ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32);
    ggml_cuda_pool_alloc<float> values_dev(ctx.pool(), count * dst->ne[0]);
    ggml_cuda_pool_alloc<int32_t> rows_dev(ctx.pool(), count);
    CUDA_CHECK(cudaMemcpyAsync(values_dev.get(), values, count * dst->ne[0] * sizeof(float), cudaMemcpyHostToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(rows_dev.get(), rows, count * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
    const dim3 blocks((dst->ne[0] + 255) / 256, count);
    moe_cpu_merge_kernel<<<blocks, 256, 0, ctx.stream()>>>((float *) dst->data, values_dev.get(), rows_dev.get(), dst->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void moe_cpu_wait_kernel(const volatile uint32_t * complete, uint32_t * status) {
    const uint64_t start = moe_gate_now_ns();
    uint32_t value;
    while ((value = *complete) == 0) {
        if (moe_gate_now_ns() - start > MOE_GATE_TIMEOUT_NS) {
            __trap();
        }
        moe_gate_relax();
    }
    __threadfence_system();
    *status = value;
}

struct moe_cpu_row_indices {
    int32_t ids[32];
};

static __global__ void moe_cpu_merge_mapped_kernel(float * dst, const float * values, moe_cpu_row_indices rows, int width, const uint32_t * complete) {
    if (*complete != 1) {
        return;
    }
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        dst[(int64_t) rows.ids[row] * width + col] = values[(int64_t) row * width + col];
    }
}

bool ggml_cuda_moe_cpu_merge_begin(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * values, const int32_t * rows, int count, const uint32_t * complete) {
    GGML_ASSERT(count > 0 && ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32);
    if (count > 32 || count * dst->ne[0] * sizeof(float) > 65536) {
        return false;
    }
    float * mapped_values = nullptr;
    uint32_t * mapped_complete = nullptr;
    cudaError_t result = cudaHostGetDevicePointer((void **) &mapped_values, const_cast<float *>(values), 0);
    if (result == cudaSuccess) {
        result = cudaHostGetDevicePointer((void **) &mapped_complete, const_cast<uint32_t *>(complete), 0);
    }
    if (result != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    // Queue after the hot experts. The CPU publishes only after writing all results.
    ggml_cuda_pool_alloc<uint32_t> status_dev(ctx.pool(), 1);
    moe_cpu_wait_kernel<<<1, 1, 0, ctx.stream()>>>(mapped_complete, status_dev.get());
    moe_cpu_row_indices indices{};
    memcpy(indices.ids, rows, count * sizeof(int32_t));
    const dim3 blocks((dst->ne[0] + 255) / 256, count);
    moe_cpu_merge_mapped_kernel<<<blocks, 256, 0, ctx.stream()>>>((float *) dst->data, mapped_values, indices, dst->ne[0], status_dev.get());
    CUDA_CHECK(cudaGetLastError());
    return true;
}
