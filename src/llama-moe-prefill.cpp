#include "llama-moe-prefill.h"

static ggml_moe_prefill_init_fn get_init(ggml_backend_t backend) {
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    return (ggml_moe_prefill_init_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_prefill_init");
}

bool llama_moe_prefill_offload::supported(ggml_backend_t backend) { return get_init(backend) != nullptr; }

llama_moe_prefill_offload::llama_moe_prefill_offload(ggml_backend_t backend, int n_buffers) {
    auto init = get_init(backend);
    GGML_ASSERT(init);
    impl = init(backend, n_buffers);
}

llama_moe_prefill_offload::~llama_moe_prefill_offload() { impl.free(impl.context); }
