#pragma once
#include "ggml-backend.h"
#include <vector>

inline bool ggml_moe_batch_fits_slots(int64_t n_tokens, int64_t n_expert_used, int64_t n_slots) {
    GGML_ASSERT(n_tokens >= 0 && n_expert_used > 0 && n_slots > 0);
    // This upper bound selects the gate path before routing IDs are available.
    return n_tokens <= n_slots / n_expert_used;
}

inline int ggml_moe_count_unique(const int32_t * ids, int n_ids, int n_expert) {
    GGML_ASSERT(n_ids >= 0 && n_expert > 0 && (ids != nullptr || n_ids == 0));
    std::vector<bool> seen(n_expert, false);
    int count = 0;
    for (int i = 0; i < n_ids; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
        if (!seen[ids[i]]) {
            seen[ids[i]] = true;
            ++count;
        }
    }
    return count;
}

inline bool ggml_moe_use_bulk_copy(int64_t n_unique, int64_t n_slots) {
    GGML_ASSERT(n_unique >= 0 && n_slots > 0);
    return n_unique > n_slots;
}

inline bool ggml_moe_use_full_bank_copy(int64_t n_tokens, int64_t n_expert_used, int64_t n_expert) {
    GGML_ASSERT(n_tokens >= 0 && n_expert_used > 0 && n_expert > 0);
    return n_tokens > (n_expert - 1) / n_expert_used;
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

using ggml_moe_cpu_max_tokens_fn = int (*)(ggml_backend_t, ggml_type);
using ggml_moe_cpu_merge_begin_fn = bool (*)(ggml_backend_t, ggml_tensor *, const float *, const int32_t *, int, const uint32_t *);
using ggml_moe_cpu_merge_fn = void (*)(ggml_backend_t, ggml_tensor *, const float *, const int32_t *, int);

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
