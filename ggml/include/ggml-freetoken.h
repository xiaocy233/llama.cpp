#pragma once

#include "ggml-backend.h"

// Private backend extension for the FreeToken paper implementation.
struct ggml_freetoken_group {
    int32_t n_expert;
    int32_t n_slots;
    int32_t n_banks;
    int32_t n_dummy;
    ggml_tensor * banks[4];
    size_t strides[4];
};

struct ggml_freetoken_layer {
    int32_t group;
    ggml_tensor * weights[4];
};

struct ggml_freetoken_job {
    int32_t layer;
    int32_t n_tokens;
    int32_t n_used;
    const float * input;
    const float * gates;
    const int32_t * ids; // -1 for GPU work
    float * output;
};

struct ggml_freetoken_cpu {
    void * userdata;
    void (*submit)(void *, const ggml_freetoken_job &);
    void (*wait)(void *);
    double (*profile)(void *);
};

struct ggml_freetoken_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t fills;
    uint64_t cpu;
    uint64_t prefill_layers;
    uint64_t prefetch_layers;
    uint64_t invalid_ids;
};

struct ggml_freetoken_api {
    void * (*create)(ggml_backend_t, const ggml_freetoken_group *, int32_t,
                     const ggml_freetoken_layer *, int32_t, ggml_freetoken_cpu, double, double);
    void (*destroy)(void *);
    ggml_tensor * (*plan)(void *, ggml_context *, int32_t, ggml_tensor *, ggml_tensor *, ggml_tensor *);
    ggml_tensor * (*merge)(void *, ggml_context *, int32_t, ggml_tensor *, ggml_tensor *);
    ggml_freetoken_stats (*stats)(void *);
    void (*bandwidths)(void *, double *, double *);
};

using ggml_freetoken_get_api_t = const ggml_freetoken_api * (*)();
