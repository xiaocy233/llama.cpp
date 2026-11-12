#include "draft-acceptance.cuh"
#include "common.cuh"
#include "ggml-backend-impl.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <math_constants.h>

static constexpr int acceptance_threads = 256;

static __device__ double acceptance_sum(double value, double * shared) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < acceptance_threads / 32 ? shared[lane] : 0.0;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

static __global__ void acceptance_stats_partial(
        const float * logits, int vocab, int stride, const ggml_draft_acceptance_batch * batch, double * partial) {
    const int row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    double weight = 0.0, weighted = 0.0;
    if (index < vocab) {
        const double delta = double(logits[(row + batch->first_row) * stride + index]) - batch->top_logits[row][0];
        weight = exp(delta);
        weighted = weight > 0.0 ? weight * delta : 0.0;
    }
    __shared__ double sums[2][acceptance_threads / 32];
    const double sum = acceptance_sum(weight, sums[0]);
    const double moment = acceptance_sum(weighted, sums[1]);
    if (threadIdx.x == 0) {
        const int offset = (row * gridDim.x + blockIdx.x) * 2;
        partial[offset] = sum;
        partial[offset + 1] = moment;
    }
}

static __global__ void acceptance_stats_finish(
        const float * logits, int stride, const ggml_draft_acceptance_batch * batch, const double * partial, int parts, ggml_draft_acceptance_result * result) {
    const int row = blockIdx.x;
    double sum = 0.0, weighted = 0.0;
    for (int i = threadIdx.x; i < parts; i += blockDim.x) {
        sum += partial[(row * parts + i) * 2];
        weighted += partial[(row * parts + i) * 2 + 1];
    }
    __shared__ double sums[2][acceptance_threads / 32];
    sum = acceptance_sum(sum, sums[0]);
    weighted = acceptance_sum(weighted, sums[1]);
    if (threadIdx.x == 0) {
        double sum10 = 0.0;
        for (int i = 0; i < GGML_DRAFT_ACCEPTANCE_TOP_K; ++i) {
            sum10 += exp(double(batch->top_logits[row][i]) - batch->top_logits[row][0]);
        }
        const double log_sum = log(sum);
        const double logit = logits[(row + batch->first_row) * stride + batch->tokens[row]];
        result->stats[row][0] = float(logit - batch->top_logits[row][0] - log_sum);
        result->stats[row][1] = batch->top_logits[row][0] - batch->top_logits[row][1];
        result->stats[row][2] = float(log_sum - weighted / sum);
        result->stats[row][3] = float(1.0 / sum10);
    }
}

static __global__ void acceptance_normalize(
        ggml_draft_acceptance_weights weights, const ggml_draft_acceptance_batch * batch,
        const ggml_draft_acceptance_result * result, const float * hidden, int hidden_stride, float * normalized) {
    const int row = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= weights.inputs) {
        return;
    }
    float value;
    if (index < 4) {
        value = result->stats[row][index];
    } else if (index == 4) {
        value = __fdiv_rn(float(row + 1), float(GGML_DRAFT_ACCEPTANCE_COUNT));
    } else if (index == 5) {
        value = GGML_DRAFT_ACCEPTANCE_COUNT;
    } else if (index == 6) {
        value = batch->position_log1p;
    } else if (index == 7) {
        value = 0.0f;
        for (int i = 0; i <= row; ++i) {
            value = __fadd_rn(value, result->stats[i][0]);
        }
    } else if (index == 8) {
        value = 1.0f;
        for (int i = 0; i <= row; ++i) {
            value = fminf(value, float(exp(double(result->stats[i][0]))));
        }
    } else {
        value = hidden[(row + batch->first_row) * hidden_stride + index - 9];
    }
    if (!isfinite(value)) {
        normalized[row * weights.inputs + index] = CUDART_NAN_F;
        return;
    }
    value = __fdiv_rn(__fsub_rn(value, weights.mean[index]), weights.stddev[index]);
    normalized[row * weights.inputs + index] = fmaxf(-10.0f, fminf(10.0f, value));
}

static __global__ void acceptance_head(
        ggml_draft_acceptance_weights weights, const float * normalized, ggml_draft_acceptance_result * result) {
    const int row = blockIdx.x;
    const int neuron = threadIdx.x;
    __shared__ float linear[64];
    float value = weights.b1[neuron];
    // Input-major weights keep loads contiguous while preserving the CPU accumulation order.
    for (int i = 0; i < weights.inputs; ++i) {
        value = __fadd_rn(value, __fmul_rn(weights.w1[i * 64 + neuron], normalized[row * weights.inputs + i]));
    }
    linear[neuron] = value;
    __syncthreads();
    if (neuron == 0) {
        float output = weights.b2[0];
        for (int i = 0; i < 64; ++i) {
            const float denominator = __fadd_rn(1.0f, float(exp(double(-linear[i]))));
            output = __fadd_rn(output, __fdiv_rn(__fmul_rn(weights.w2[i], linear[i]), denominator));
        }
        const float calibrated = __fadd_rn(__fmul_rn(output, weights.scale), weights.bias);
        result->probabilities[row] = __fdiv_rn(1.0f, __fadd_rn(1.0f, float(exp(double(-calibrated)))));
    }
}

static __global__ void acceptance_cutoff(ggml_draft_acceptance_result * result, float threshold) {
    float survival = 1.0f;
    result->kept = 0;
    for (int i = 0; i < GGML_DRAFT_ACCEPTANCE_COUNT; ++i) {
        if (!isfinite(result->probabilities[i])) {
            result->kept = -1;
            return;
        }
        survival = __fmul_rn(survival, result->probabilities[i]);
        if (survival < threshold) {
            return;
        }
        ++result->kept;
    }
}

struct cuda_draft_acceptance {
    ggml_backend_cuda_context * backend;
    ggml_draft_acceptance_weights weights;
    float * storage = nullptr;
    float * normalized = nullptr;
    double * partial = nullptr;
    int parts = 0;
    ggml_draft_acceptance_batch * batch_device = nullptr;
    ggml_draft_acceptance_batch * batch_host = nullptr;
    ggml_draft_acceptance_result * result_device = nullptr;
    ggml_draft_acceptance_result * result_host = nullptr;
};

static void acceptance_free(void * context) {
    auto * state = static_cast<cuda_draft_acceptance *>(context);
    ggml_cuda_set_device(state->backend->device);
    CUDA_CHECK(cudaFree(state->storage));
    CUDA_CHECK(cudaFree(state->normalized));
    CUDA_CHECK(cudaFree(state->partial));
    CUDA_CHECK(cudaFree(state->batch_device));
    CUDA_CHECK(cudaFree(state->result_device));
    CUDA_CHECK(cudaFreeHost(state->batch_host));
    CUDA_CHECK(cudaFreeHost(state->result_host));
    delete state;
}

static void acceptance_evaluate(
        void * context, const ggml_tensor * logits, const ggml_tensor * hidden,
        const ggml_draft_acceptance_batch * batch, ggml_draft_acceptance_result * result) {
    auto * state = static_cast<cuda_draft_acceptance *>(context);
    GGML_ASSERT(logits && logits->type == GGML_TYPE_F32 && ggml_is_contiguous(logits));
    GGML_ASSERT(logits->ne[0] >= 10 && logits->ne[0] <= INT32_MAX && ggml_is_matrix(logits));
    GGML_ASSERT(batch->first_row >= 0 && batch->first_row + GGML_DRAFT_ACCEPTANCE_COUNT <= logits->ne[1]);
    if (state->weights.hidden) {
        GGML_ASSERT(hidden && hidden->type == GGML_TYPE_F32 && ggml_is_contiguous(hidden));
        GGML_ASSERT(hidden->ne[0] == state->weights.hidden && hidden->ne[1] == logits->ne[1]);
    }
    for (int i = 0; i < GGML_DRAFT_ACCEPTANCE_COUNT; ++i) {
        GGML_ASSERT(batch->tokens[i] >= 0 && batch->tokens[i] < logits->ne[0]);
    }
    ggml_cuda_set_device(state->backend->device);
    const auto stream = state->backend->stream();
    const int vocab = logits->ne[0];
    const int parts = (vocab + acceptance_threads - 1) / acceptance_threads;
    if (parts != state->parts) {
        CUDA_CHECK(cudaFree(state->partial));
        CUDA_CHECK(cudaMalloc(&state->partial, parts * GGML_DRAFT_ACCEPTANCE_COUNT * 2 * sizeof(double)));
        state->parts = parts;
    }
    *state->batch_host = *batch;
    CUDA_CHECK(cudaMemcpyAsync(state->batch_device, state->batch_host, sizeof(*batch), cudaMemcpyHostToDevice, stream));
    acceptance_stats_partial<<<dim3(parts, GGML_DRAFT_ACCEPTANCE_COUNT), acceptance_threads, 0, stream>>>(
            static_cast<const float *>(logits->data), vocab, logits->nb[1] / sizeof(float), state->batch_device, state->partial);
    acceptance_stats_finish<<<GGML_DRAFT_ACCEPTANCE_COUNT, acceptance_threads, 0, stream>>>(
            static_cast<const float *>(logits->data), logits->nb[1] / sizeof(float), state->batch_device, state->partial, parts, state->result_device);
    acceptance_normalize<<<dim3((state->weights.inputs + 255) / 256, GGML_DRAFT_ACCEPTANCE_COUNT), 256, 0, stream>>>(
            state->weights, state->batch_device, state->result_device, hidden ? static_cast<const float *>(hidden->data) : nullptr,
            hidden ? hidden->nb[1] / sizeof(float) : 0, state->normalized);
    acceptance_head<<<GGML_DRAFT_ACCEPTANCE_COUNT, 64, 0, stream>>>(state->weights, state->normalized, state->result_device);
    acceptance_cutoff<<<1, 1, 0, stream>>>(state->result_device, state->weights.threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(state->result_host, state->result_device, sizeof(*result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *result = *state->result_host;
}

ggml_draft_acceptance ggml_cuda_draft_acceptance_init(ggml_backend_t backend, const ggml_draft_acceptance_weights * weights) {
    GGML_ASSERT(weights && weights->width == 64 && weights->inputs == weights->hidden + 9);
    auto * state = new cuda_draft_acceptance{};
    state->backend = static_cast<ggml_backend_cuda_context *>(backend->context);
    state->weights = *weights;
    ggml_cuda_set_device(state->backend->device);
    std::vector<float> packed;
    auto add = [&](const float * data, size_t count) {
        const size_t offset = packed.size();
        packed.insert(packed.end(), data, data + count);
        return offset;
    };
    const size_t mean = add(weights->mean, weights->inputs);
    const size_t stddev = add(weights->stddev, weights->inputs);
    std::vector<float> transposed(weights->inputs * 64);
    for (int i = 0; i < weights->inputs; ++i) {
        for (int j = 0; j < 64; ++j) {
            transposed[i * 64 + j] = weights->w1[j * weights->inputs + i];
        }
    }
    const size_t w1 = add(transposed.data(), transposed.size());
    const size_t b1 = add(weights->b1, 64);
    const size_t w2 = add(weights->w2, 64);
    const size_t b2 = add(weights->b2, 1);
    CUDA_CHECK(cudaMalloc(&state->storage, packed.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state->storage, packed.data(), packed.size() * sizeof(float), cudaMemcpyHostToDevice));
    state->weights.mean = state->storage + mean;
    state->weights.stddev = state->storage + stddev;
    state->weights.w1 = state->storage + w1;
    state->weights.b1 = state->storage + b1;
    state->weights.w2 = state->storage + w2;
    state->weights.b2 = state->storage + b2;
    CUDA_CHECK(cudaMalloc(&state->normalized, GGML_DRAFT_ACCEPTANCE_COUNT * weights->inputs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->batch_device, sizeof(ggml_draft_acceptance_batch)));
    CUDA_CHECK(cudaMalloc(&state->result_device, sizeof(ggml_draft_acceptance_result)));
    CUDA_CHECK(cudaMallocHost(&state->batch_host, sizeof(ggml_draft_acceptance_batch)));
    CUDA_CHECK(cudaMallocHost(&state->result_host, sizeof(ggml_draft_acceptance_result)));
    return {state, acceptance_free, acceptance_evaluate};
}

#endif
