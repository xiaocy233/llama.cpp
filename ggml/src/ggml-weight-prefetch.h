#pragma once
#include "ggml-backend.h"
#include <vector>

struct ggml_weight_prefetch_entry { int split; int input; };

struct ggml_weight_prefetch_plan {
    std::vector<ggml_weight_prefetch_entry> entries;
    std::vector<int> entry_of_split;
    std::vector<int> issue_at_split;
    void schedule(int n_splits, int depth);
};

bool ggml_weight_prefetch_eligible(const ggml_cgraph * graph, const ggml_tensor * input, const ggml_tensor * copy);
