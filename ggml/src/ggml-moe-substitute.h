#pragma once
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>

struct ggml_moe_substitute_result {
    int32_t ids[GGML_MOE_GATE_MAX_IDS];
};

static inline void ggml_moe_substitute_host(const int32_t * ids, const float * probs, const int32_t * map,
        int n_ids, int n_expert, int n_slots, float threshold, ggml_moe_substitute_result & out) {
    GGML_ASSERT(n_ids > 0 && n_ids <= n_expert);
    GGML_ASSERT(n_expert > 0 && n_expert <= GGML_MOE_GATE_MAX_EXPERTS);
    GGML_ASSERT(n_slots >= n_ids && std::isfinite(threshold) && threshold >= 0 && threshold <= 1);
    bool used[GGML_MOE_GATE_MAX_EXPERTS] = {};
    bool resident[GGML_MOE_GATE_MAX_EXPERTS] = {};
    float sum = 0.0f;
    for (int e = 0; e < n_expert; e++) resident[e] = map[e] >= 0 && map[e] < n_slots;
    // Protect all original experts, including those not processed yet.
    for (int i = 0; i < n_ids; i++) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert && !used[ids[i]]);
        used[ids[i]] = true;
        sum += probs[ids[i]];
    }
    const bool valid = std::isfinite(sum) && sum > 0.0f;
    for (int i = 0; i < n_ids; i++) {
        const int original = ids[i];
        int selected = original;
        if (valid && !resident[original] && probs[original] / sum < threshold) {
            float score = -INFINITY;
            for (int e = 0; e < n_expert; e++) {
                if (resident[e] && !used[e] && std::isfinite(probs[e]) && probs[e] > score) {
                    selected = e;
                    score = probs[e];
                }
            }
            if (selected != original) {
                used[selected] = true;
            }
        }
        out.ids[i] = selected;
    }
}
