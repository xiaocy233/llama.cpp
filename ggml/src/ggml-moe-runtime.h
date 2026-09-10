#pragma once
#include "ggml-backend.h"

struct ggml_moe_runtime;
struct ggml_moe_slot_state;
ggml_moe_runtime * ggml_moe_runtime_new(ggml_backend_t * backends, int n_backends);
void ggml_moe_runtime_free(ggml_moe_runtime * runtime);
bool ggml_moe_runtime_active(const ggml_moe_runtime * runtime);
bool ggml_moe_runtime_drain(const ggml_moe_runtime * runtime);
void ggml_moe_runtime_set_decode(ggml_moe_runtime * runtime, bool decode);
const ggml_moe_slot_state * ggml_moe_runtime_bank_of(ggml_moe_runtime * runtime, const ggml_tensor * src);
bool ggml_moe_runtime_bank_range(const ggml_moe_slot_state * state, const ggml_tensor * src, int * expert, size_t * offset, size_t * size);
void ggml_moe_runtime_quiesce(ggml_moe_runtime * runtime);
void ggml_moe_runtime_publish_split(ggml_moe_runtime * runtime, const ggml_cgraph * graph);
void ggml_moe_runtime_new_token(ggml_moe_runtime * runtime);
bool ggml_moe_runtime_add_moe_slot_cache(ggml_moe_runtime * runtime, const ggml_moe_slot_cache * cache, int n_expert_used);
void ggml_moe_runtime_set_moe_gate_seq(ggml_moe_runtime * runtime, ggml_tensor * seq);
void ggml_moe_runtime_set_moe_substitute_enabled(ggml_moe_runtime * runtime, bool enabled);
const ggml_moe_slot_cache * ggml_moe_runtime_find_moe_slot_cache(ggml_moe_runtime * runtime, const ggml_tensor * src);
const ggml_moe_slot_cache * ggml_moe_runtime_find_moe_head_predictor(ggml_moe_runtime * runtime, int source_layer);
