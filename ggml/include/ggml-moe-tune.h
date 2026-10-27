#pragma once

#include "ggml-backend.h"

// Experimental telemetry for one completed, single-sequence verification batch.
#define GGML_MOE_TUNE_EXPERTS 256
#define GGML_MOE_TUNE_IDS 256

struct ggml_moe_tune_layer {
    int layer, experts, slots, top_k, width;
    uint64_t bytes[3];
    int types[3];
    int owners[GGML_MOE_TUNE_EXPERTS];
    uint64_t freq[GGML_MOE_TUNE_EXPERTS];
    bool prefetchable[GGML_MOE_TUNE_EXPERTS];
    int n_ids, n_pred, assignments, loads, prefetches;
    int ids[GGML_MOE_TUNE_IDS];
    int predicted[GGML_MOE_TUNE_IDS];
    double gate_us, fill_us, prefetch_us, service_us, cpu_us;
};

#ifdef __cplusplus
extern "C" {
#endif
// Call only between completed graphs. Prediction width selects a prefix of the built predictor.
// Fixed configuration stays active between sampled batches.
GGML_API bool ggml_backend_sched_moe_tune_begin(ggml_backend_sched_t sched, int width, int cpu_cap, bool profile, bool fixed);
GGML_API int ggml_backend_sched_moe_tune_end(ggml_backend_sched_t sched, struct ggml_moe_tune_layer * layers, int capacity);
#ifdef __cplusplus
}
#endif
