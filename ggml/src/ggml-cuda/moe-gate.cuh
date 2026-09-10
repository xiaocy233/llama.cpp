#pragma once

#include "common.cuh"

void ggml_cuda_op_moe_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool capturing);

// Release a parked gate from the device, ordered after everything already on [stream].
void ggml_cuda_moe_gate_release(ggml_backend_cuda_context & ctx, cudaStream_t stream, int layer, uint32_t seq);
