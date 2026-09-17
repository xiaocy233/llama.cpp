#pragma once
#include "ggml-backend.h"

inline bool ggml_moe_use_bulk_copy(int64_t n_tokens, int64_t n_expert_used, int64_t n_expert, int64_t n_slots = 0) {
    GGML_ASSERT(n_tokens >= 0 && n_expert_used > 0 && n_expert > 0 && n_slots >= 0);
    // With slots, keep the whole batch in cache. Without slots, retain the full-bank threshold.
    return n_slots > 0 ? n_tokens > n_slots / n_expert_used : n_tokens > (n_expert - 1) / n_expert_used;
}

struct ggml_moe_backend_caps {
    int transfer_stream;
    bool host_copies;
    bool substitute;
};
using ggml_moe_backend_caps_fn = ggml_moe_backend_caps (*)(ggml_backend_dev_t);

inline ggml_moe_backend_caps ggml_moe_backend_get_caps(ggml_backend_dev_t device) {
    auto reg = ggml_backend_dev_backend_reg(device);
    auto get = (ggml_moe_backend_caps_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_get_caps");
    return get ? get(device) : ggml_moe_backend_caps{0, false, false};
}

struct ggml_moe_prefill {
    void * context;
    void (*free)(void * context);
    void (*add_pool)(void * context, const ggml_tensor * source, int layer, int fd, uint64_t offset);
    ggml_tensor * (*bind_pool)(void * context, const char * name, int layer);
    void (*finalize)(void * context);
    void (*prepare)(void * context);
    void (*release)(void * context);
    bool (*active)(const void * context);
};
using ggml_moe_prefill_init_fn = ggml_moe_prefill (*)(ggml_backend_t, int);
