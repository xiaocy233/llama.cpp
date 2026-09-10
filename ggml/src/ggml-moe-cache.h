#pragma once
#include <cstdint>

struct ggml_moe_cache_policy {
    int n_expert;
    int n_slots;
    int32_t * slot_of_expert;
    int32_t * expert_of_slot;
    uint32_t * freq;
    uint32_t * pinned;
    uint32_t epoch;
    int32_t * free_slot;
    int free_cnt;
    uint32_t * free_pass;
    uint32_t * fill_seq;
};

int32_t ggml_moe_cache_victim(const ggml_moe_cache_policy * cache, bool same_token);
int32_t ggml_moe_cache_take(ggml_moe_cache_policy * cache, uint32_t completed_pass);
