#pragma once

#include "ggml-backend.h"

constexpr int GGML_DRAFT_ACCEPTANCE_COUNT = 7;
constexpr int GGML_DRAFT_ACCEPTANCE_TOP_K = 10;

struct ggml_draft_acceptance_weights {
    int inputs, hidden, width;
    float threshold, scale, bias;
    const float * mean;
    const float * stddev;
    const float * w1;
    const float * b1;
    const float * w2;
    const float * b2;
};

struct ggml_draft_acceptance_batch {
    int32_t tokens[GGML_DRAFT_ACCEPTANCE_COUNT];
    float top_logits[GGML_DRAFT_ACCEPTANCE_COUNT][GGML_DRAFT_ACCEPTANCE_TOP_K];
    float position_log1p;
    int first_row;
};

struct ggml_draft_acceptance_result {
    int kept;
    float stats[GGML_DRAFT_ACCEPTANCE_COUNT][4];
    float probabilities[GGML_DRAFT_ACCEPTANCE_COUNT];
};

struct ggml_draft_acceptance {
    void * context = nullptr;
    void (*free)(void * context) = nullptr;
    void (*evaluate)(void * context, const ggml_tensor * logits, const ggml_tensor * hidden, const ggml_draft_acceptance_batch * batch, ggml_draft_acceptance_result * result) = nullptr;
};

using ggml_draft_acceptance_init_fn = ggml_draft_acceptance (*)(ggml_backend_t, const ggml_draft_acceptance_weights *);
