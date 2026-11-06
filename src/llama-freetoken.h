#pragma once

#include "ggml-freetoken.h"

#include <memory>
#include <vector>

class llama_freetoken_cpu {
public:
    llama_freetoken_cpu(const std::vector<ggml_freetoken_layer> & layers, int32_t n_threads);
    ~llama_freetoken_cpu();
    ggml_freetoken_cpu callbacks();
    double profile();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
