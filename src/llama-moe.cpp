#include "llama-context.h"
#include "llama-model.h"
#include "ggml-alloc.h"

#include <stdexcept>

void llama_context::init_moe_slot_caches() {
    if (model.moe_slot_layers.empty() || !moe_slot_caches.empty()) {
        return;
    }

    const size_t n_meta = model.moe_slot_layers.size() * 4 + 1;
    ctx_moe.reset(ggml_init({ggml_tensor_overhead() * n_meta, nullptr, true}));
    if (!ctx_moe) {
        throw std::runtime_error("failed to allocate MoE cache metadata");
    }
    auto * gate_seq = ggml_new_tensor_1d(ctx_moe.get(), GGML_TYPE_I32, 1);
    ggml_set_name(gate_seq, "moe_gate_seq");

    for (const auto & L : model.moe_slot_layers) {
        ggml_moe_slot_cache c = {};
        for (int m = 0; m < 3; m++) {
            c.src[m] = L.src[m];
            c.src_off[m] = L.src_off[m];
            c.slots[m] = ggml_new_tensor_3d(ctx_moe.get(), L.src[m]->type, L.src[m]->ne[0], L.src[m]->ne[1], L.n_slots);
            ggml_format_name(c.slots[m], "blk.%d.moe_slots.%d", L.layer, m);
            GGML_ASSERT(c.slots[m]->nb[2] == L.src[m]->nb[2]);
        }
        c.loc_map = ggml_new_tensor_1d(ctx_moe.get(), GGML_TYPE_I32, L.n_expert);
        ggml_format_name(c.loc_map, "blk.%d.moe_loc_map", L.layer);
        c.gate_seq    = gate_seq;
        c.n_expert    = L.n_expert;
        c.n_slots     = L.n_slots;
        c.layer       = L.layer;
        c.pinned      = L.pinned;
        c.src_fd      = L.src_fd;
        c.pred_w      = L.pred_w;
        c.n_pred      = L.n_pred;
        c.pred_target = L.pred_target;
        c.head_pred_w      = L.head_pred_w;
        c.head_n_pred      = L.head_n_pred;
        c.head_pred_source = L.head_pred_source;
        c.head_gate_slot   = -1;
        moe_slot_caches.push_back(c);
    }

    buf_moe.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_moe.get(), model.moe_slot_buft));
    if (!buf_moe) {
        throw std::runtime_error("failed to allocate MoE slot buffers");
    }
    ggml_backend_buffer_set_usage(buf_moe.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const int32_t seq0 = 0;
    ggml_backend_tensor_set(gate_seq, &seq0, 0, sizeof(seq0));
}
