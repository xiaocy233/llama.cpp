#include "moe-gate.cuh"

#include "ggml-backend.h"

// The decode MoE gate. It replaces the host round trip that used to cut every streamed layer into
// CUDA -> CPU -> CUDA. See probe/GATE-PLAN.md for why that split existed and what this costs.
//
// Publishing happens on every layer, hit or miss, because the host needs the ids to keep its LFU
// metadata and the prediction to prefetch. seq is written last, after a system fence: that order is
// the only thing that keeps the host from reading a half-written entry.
//
// The gate never runs a layer on an expert the router did not choose. If an expert is not in a slot it
// waits for the copy, however long that takes, and gives up only by aborting.

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

static __global__ void moe_gate_kernel(
        const int32_t * loc_map,
        const int32_t * __restrict__ ids_in,
        const int32_t * __restrict__ seq_p,
        ggml_moe_gate_entry * __restrict__ ring,
        volatile uint32_t * __restrict__ release,
        volatile int32_t * __restrict__ timeout,
        int32_t * __restrict__ out,
        const int n_ids, const int n_pred, const int layer, const int n_slots, const int64_t n_in) {
    const uint32_t seq = (uint32_t) *seq_p;
    const int      t   = threadIdx.x;

    // publish-only form. Prefill uses it: report the newest ids so the host can warm the cache before
    // decode, then hand the ids on untouched. Nothing here can miss, so nothing here can park.
    if (n_ids == 0) {
        ggml_moe_gate_entry * box = ring + layer;
        for (int i = t; i < n_pred; i += blockDim.x) {
            box->ids[i] = ids_in[n_in - n_pred + i];
        }
        if (t == 0) {
            box->layer     = (uint32_t) layer;
            box->miss_mask = 0u;
            box->n_ids     = 0u;
            box->n_pred    = (uint32_t) n_pred;
        }
        __syncthreads();
        __threadfence_system();
        if (t == 0) {
            box->seq = seq;
        }

        for (int64_t i = t; i < n_in; i += blockDim.x) {
            out[i] = ids_in[i];
        }
        return;
    }

    __shared__ uint32_t s_miss;
    __shared__ int      s_discover;
    if (t == 0) {
        s_miss = 0u;
        // A discovery pass: the host released this gate before the graph started. That pass reaches the
        // driver node by node, so a park in it stops the device while the host is still submitting.
        // Report everything missing and read nothing - a read here would race the worker and leave a
        // different cache behind on every run.
        s_discover = release[layer] >= seq ? 1 : 0;
    }
    __syncthreads();

    if (s_discover) {
        if (t == 0) {
            s_miss = n_ids >= 32 ? ~0u : (1u << n_ids) - 1u;
        }
    } else {
        for (int i = t; i < n_ids; i += blockDim.x) {
            if (moe_gate_loc(loc_map, ids_in[i]) >= n_slots) {
                atomicOr(&s_miss, 1u << i);
            }
        }
    }
    __syncthreads();

    ggml_moe_gate_entry * box = ring + layer;
    for (int i = t; i < n_ids + n_pred; i += blockDim.x) {
        box->ids[i] = ids_in[i];
    }
    if (t == 0) {
        box->layer     = (uint32_t) layer;
        box->miss_mask = s_miss;
        box->n_ids     = (uint32_t) n_ids;
        box->n_pred    = (uint32_t) n_pred;
    }
    __syncthreads();
    __threadfence_system();
    if (t == 0) {
        box->seq = seq;
    }

    if (s_discover) {
        // the output of this pass is thrown away, and it may not wait for any part of it
        for (int i = t; i < n_ids; i += blockDim.x) {
            out[i] = 0;
        }
        return;
    }

    if (s_miss != 0u) {
        if (t == 0) {
            const uint64_t t0 = moe_gate_now_ns();
            while (release[layer] < seq) {
                if (moe_gate_now_ns() - t0 > MOE_GATE_TIMEOUT_NS) {
                    break;   // a late release is not fatal if the rows arrived; the resolve decides
                }
                moe_gate_relax();
            }
        }
        __syncthreads();
        __threadfence_system();   // order the resolve below after the release we just saw
    }

    // Resolve. An expert that has not arrived is waited for, never replaced: there is no host bank
    // behind the slots, so any substitute would be a different expert than the router chose. The wait
    // ends because the worker fills every expert the host has no slot for, and this is always a
    // captured pass - the host is not in the driver, so the worker can always get its copies through.
    for (int i = t; i < n_ids; i += blockDim.x) {
        const int32_t e = ids_in[i];
        int32_t       p = moe_gate_loc(loc_map, e);
        if (p >= n_slots) {
            const uint64_t t0 = moe_gate_now_ns();
            while (p >= n_slots) {
                if (moe_gate_now_ns() - t0 > MOE_GATE_TIMEOUT_NS) {
                    atomicAdd((int32_t *) timeout + layer, 1);
                    printf("MOE gate: layer %d expert %d never arrived\n", layer, e);
                    __trap();
                }
                moe_gate_relax();
                p = moe_gate_loc(loc_map, e);
            }
        }
        out[i] = p;
    }
}

// The host cannot store the release while the gate is parked: on the first decode graph the nodes go
// to the driver one at a time, the parked gate stops the device, and the host then blocks in the
// driver with the queue full - so the wait for the copies cannot finish. Put the release behind the
// copies on the same stream instead, and the host never has to wait at all.
static __global__ void moe_gate_release_kernel(volatile uint32_t * release, const int layer, const uint32_t seq) {
    __threadfence_system();   // the copies ahead of this must be visible to the gate first
    release[layer] = seq;
}

void ggml_cuda_moe_gate_release(ggml_backend_cuda_context & ctx, cudaStream_t stream, int layer, uint32_t seq) {
    const ggml_cuda_moe_gate_channel & ch = ctx.moe_gate_channel();

    GGML_ASSERT(layer < GGML_MOE_GATE_MAX_LAYERS);

    moe_gate_release_kernel<<<1, 1, 0, stream>>>(ch.release_dev, layer, seq);
}

void ggml_cuda_op_moe_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * loc_map = dst->src[0];
    const ggml_tensor * ids_in  = dst->src[1];
    const ggml_tensor * seq     = dst->src[2];

    const int layer   = ggml_get_op_params_i32(dst, 0);
    const int n_ids   = ggml_get_op_params_i32(dst, 1);
    const int n_pred  = ggml_get_op_params_i32(dst, 2);
    const int n_slots = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(n_ids <= 32 && n_ids + n_pred <= GGML_MOE_GATE_MAX_IDS);   // miss_mask caps n_ids
    GGML_ASSERT(layer < GGML_MOE_GATE_MAX_LAYERS);

    const ggml_cuda_moe_gate_channel & ch = ctx.moe_gate_channel();

    // one block: the park spins on a host store, so more blocks would only spin harder. The
    // publish-only form copies the ids through, which is why it gets the wider block.
    const int n_thread = n_ids > 0 ? 32 : 256;

    moe_gate_kernel<<<1, n_thread, 0, ctx.stream()>>>(
            (const int32_t *) loc_map->data, (const int32_t *) ids_in->data,
            (const int32_t *) seq->data, ch.ring_dev, ch.release_dev, ch.timeout_dev,
            (int32_t *) dst->data, n_ids, n_pred, layer, n_slots, ggml_nelements(ids_in));
}
