#pragma once
#include "ggml-backend.h"

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
