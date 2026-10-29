#pragma once
#include "ggml-backend.h"
#include "ggml-moe-tune.h"

struct ggml_moe_runtime;
struct ggml_moe_slot_state;
ggml_moe_runtime * ggml_moe_runtime_new(ggml_backend_t * backends, int n_backends);
void ggml_moe_runtime_free(ggml_moe_runtime * runtime);
bool ggml_moe_runtime_active(const ggml_moe_runtime * runtime);
bool ggml_moe_runtime_drain(const ggml_moe_runtime * runtime);
void ggml_moe_runtime_set_decode(ggml_moe_runtime * runtime, bool decode);
ggml_moe_slot_state * ggml_moe_runtime_bank_of(ggml_moe_runtime * runtime, const ggml_tensor * src);
bool ggml_moe_runtime_bank_range(const ggml_moe_slot_state * state, const ggml_tensor * src, int * expert, size_t * offset, size_t * size);
bool ggml_moe_runtime_has_expert(const ggml_moe_slot_state * state, int expert);
bool ggml_moe_runtime_prepare_batch(ggml_moe_runtime * runtime, ggml_moe_slot_state * state, const int32_t * ids, int n_ids, const int32_t * predicted, int n_predicted);
enum ggml_status ggml_moe_runtime_compute_cpu_node(ggml_moe_runtime * runtime, ggml_backend_t backend, ggml_tensor * node);
void ggml_moe_runtime_quiesce(ggml_moe_runtime * runtime);
void ggml_moe_runtime_publish_split(ggml_moe_runtime * runtime, const ggml_cgraph * graph);
void ggml_moe_runtime_new_token(ggml_moe_runtime * runtime);
bool ggml_moe_runtime_add_moe_slot_cache(ggml_moe_runtime * runtime, const ggml_moe_slot_cache * cache, int n_expert_used);
void ggml_moe_runtime_set_moe_gate_seq(ggml_moe_runtime * runtime, ggml_tensor * seq);
void ggml_moe_runtime_set_moe_substitute_enabled(ggml_moe_runtime * runtime, bool enabled);
const ggml_moe_slot_cache * ggml_moe_runtime_find_moe_slot_cache(ggml_moe_runtime * runtime, const ggml_tensor * src);
const ggml_moe_slot_cache * ggml_moe_runtime_find_moe_head_predictor(ggml_moe_runtime * runtime, int source_layer);
bool ggml_moe_runtime_tune_begin(ggml_moe_runtime * runtime, int width, int cpu_cap, bool profile, bool fixed, bool trace_only);
int ggml_moe_runtime_tune_end(ggml_moe_runtime * runtime, ggml_moe_tune_layer * layers, int capacity, bool trace_only);

bool ggml_moe_runtime_tune_cache(ggml_moe_runtime * runtime, bool capture);
