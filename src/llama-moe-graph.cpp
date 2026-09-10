#include "llama-graph.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-moe-backend.h"

#include <cassert>
#include <cmath>
#include <cstring>

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;
    ggml_tensor * router_logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
        router_logits = logits;
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    // Expert prediction for the next cached layer, decode only.
    //
    // cur is the router's input: past the attention residual and this layer's ffn norm. Feeding it
    // to the *next* layer's router is the cheapest possible guess at what that layer will select -
    // the residual stream moves slowly, and the router is a plain matrix, so no extra weights and no
    // training are involved. It is deliberately the simplest form; the input position, the weight
    // and how far ahead to look are all things to vary later.
    //
    // Runs on the device next to the real router. build_lora_mm_id appends it to the ids the gate
    // reads, so it reaches the host inside the entry the gate publishes anyway. Nothing downstream
    // reads it, so a bad guess cannot change the output.
    if (n_tokens == 1) {
        const ggml_moe_slot_cache * head = ggml_backend_sched_find_moe_head_predictor(sched, il);
        if (head && head->head_pred_w && head->head_n_pred > 0 && head->head_gate_slot >= 0) {
            ggml_tensor * pl = ggml_mul_mat(ctx0, head->head_pred_w, cur);
            ggml_tensor * pi = ggml_top_k(ctx0, pl, head->head_n_pred);
            cb(pl, "ffn_moe_head_pred_logits", il);
            if (router_logits != nullptr) {
                ggml_build_forward_expand(gf, router_logits);
            }
            ggml_build_forward_expand(gf, pl);
            selected_experts = ggml_moe_head(
                    ctx0, selected_experts, pi, head->gate_seq, head->head_gate_slot);
            cb(selected_experts, "ffn_moe_head_join", il);
        }

        const ggml_moe_slot_cache * sc = ggml_backend_sched_find_moe_slot_cache(sched, up_exps);
        if (sc == nullptr) {
            sc = ggml_backend_sched_find_moe_slot_cache(sched, gate_exps);
        }
        if (sc && sc->pred_w && sc->n_pred > 0) {
            ggml_tensor * pl = ggml_mul_mat(ctx0, sc->pred_w, cur);      // [n_expert, 1]
            ggml_tensor * pi = ggml_top_k(ctx0, pl, sc->n_pred);         // [n_pred, 1] I32
            cb(pl, "ffn_moe_pred_logits", il);
            moe_slot_preds.push_back({ sc->loc_map, pi });
        }
    }

    if (n_tokens == 1 && cparams.moe_substitute_threshold > 0.0f) {
        GGML_ASSERT(arch == LLM_ARCH_QWEN35MOE && norm_w && gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX);
        const ggml_moe_slot_cache * cache = ggml_backend_sched_find_moe_slot_cache(sched, up_exps);
        if (!cache) cache = ggml_backend_sched_find_moe_slot_cache(sched, gate_exps);
        if (cache) {
            auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(cache->slots[0]->buffer));
            GGML_ASSERT(ggml_moe_backend_get_caps(device).substitute);
            ggml_tensor * input_ids = ggml_is_contiguous(selected_experts) ? selected_experts : ggml_cont(ctx0, selected_experts);
            for (const auto & prediction : moe_slot_preds) {
                if (prediction.loc_map == cache->loc_map) {
                    input_ids = ggml_concat(ctx0, input_ids, prediction.ids, 0);
                    break;
                }
            }
            ggml_tensor * decision = ggml_moe_gate_substitute(ctx0, cache->loc_map, input_ids, cache->gate_seq,
                    probs, cache->gate_slot, n_expert_used, cache->n_slots, cparams.moe_substitute_threshold);
            ggml_tensor * resolved = ggml_view_2d(ctx0, decision, n_expert_used, 1, decision->nb[1], 0);
            selected_experts = ggml_view_2d(ctx0, decision, n_expert_used, 1, decision->nb[1], n_expert_used * sizeof(int32_t));
            moe_slot_remaps.push_back({cache->loc_map, selected_experts, resolved});
            cb(selected_experts, "ffn_moe_substituted_ids", il);
        }
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    // Metal prefill layer ring: bind the disk-resident banks to their ring slots. The pool tensors
    // are regular device-resident weights from the scheduler's point of view - no slot cache
    // registration, so build_lora_mm_id falls back to the plain MUL_MAT_ID path, and the
    // interceptor in ggml-metal-ops.cpp adds beacon + event wait in front of it.
    if (moe_prefill != nullptr && n_tokens > 1) {
        auto bind = [&](ggml_tensor * orig) -> ggml_tensor * {
            if (orig == nullptr) {
                return nullptr;
            }
            // only disk banks ride the ring; layers pinned in VRAM keep their weights as they are
            if (orig->buffer == nullptr ||
                    ggml_backend_buffer_get_type(orig->buffer) != llama_moe_cache_disk_buft()) {
                return orig;
            }
            // the pools were pre-created at context setup, keyed by the bank tensor name
            ggml_tensor * bound = moe_prefill->bind_pool(orig->name, il);
            if (bound == nullptr) {
                GGML_ABORT("moe prefill: tensor not registered as a disk bank: %s", orig->name);
            }
            return bound;
        };
        gate_up_exps = bind(gate_up_exps);
        up_exps      = bind(up_exps);
        gate_exps    = bind(gate_exps);
        down_exps    = bind(down_exps);
    }

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora