#pragma once

#include "common.cuh"
#include "ggml-freetoken.h"

bool ggml_cuda_freetoken_is_op(const ggml_tensor * op);
void ggml_cuda_freetoken_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op);
const ggml_freetoken_api * ggml_cuda_freetoken_get_api();
