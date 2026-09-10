#pragma once
#include "../ggml/src/ggml-moe-backend.h"

class llama_moe_prefill_offload {
public:
    static bool supported(ggml_backend_t backend);
    llama_moe_prefill_offload(ggml_backend_t backend, int n_buffers);
    ~llama_moe_prefill_offload();
    void add_pool(const ggml_tensor * src, int layer, int fd, uint64_t offset) { impl.add_pool(impl.context, src, layer, fd, offset); }
    ggml_tensor * bind_pool(const char * name, int layer) { return impl.bind_pool(impl.context, name, layer); }
    void finalize_buffers() { impl.finalize(impl.context); }
    void prepare_prefill() { impl.prepare(impl.context); }
    void release_prefill_buffers() { impl.release(impl.context); }
    bool active() const { return impl.active(impl.context); }
private:
    ggml_moe_prefill impl;
};
