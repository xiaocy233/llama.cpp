#pragma once

#include "ggml-backend.h"

// Experimental telemetry for one completed, single-sequence verification batch.
#define GGML_MOE_TUNE_IDS 256

struct ggml_moe_tune_layer {
    int layer, experts, slots, top_k, width;
    uint64_t bytes[3];
    int types[3];
    const int * owners;           // [slots]
    const uint64_t * freq;        // [experts]
    const uint8_t * prefetchable; // [slots]
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
// trace_only requires completed graphs and reads host state without device synchronization.
GGML_API bool ggml_backend_sched_moe_tune_begin(ggml_backend_sched_t sched, int width, int cpu_cap, bool profile, bool fixed, bool trace_only);
// Save or restore expert cache state between completed requests for controlled tuning trials.
GGML_API bool ggml_backend_sched_moe_tune_cache(ggml_backend_sched_t sched, bool capture);
// Returned arrays are borrowed until the next tune_begin or cache destruction.
GGML_API int ggml_backend_sched_moe_tune_end(ggml_backend_sched_t sched, struct ggml_moe_tune_layer * layers, int capacity, bool trace_only);
#ifdef __cplusplus
}
#endif
