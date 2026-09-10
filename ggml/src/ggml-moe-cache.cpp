#include "ggml-moe-cache.h"
#include <climits>

int32_t ggml_moe_cache_victim(const ggml_moe_cache_policy * st, bool same_tok) {
    int32_t  v    = -1;
    uint32_t best = UINT32_MAX;
    for (int c = 0; c < st->n_slots; c++) {
        if (st->pinned[c] == st->epoch || (same_tok && st->fill_seq[c] >= st->epoch)) {
            continue;
        }
        if (st->expert_of_slot[c] < 0) {
            continue;   // already in the pool, or about to be
        }
        const uint32_t f = st->freq[st->expert_of_slot[c]];
        if (f < best) {
            best = f;
            v    = c;
        }
    }
    return v;
}

int32_t ggml_moe_cache_take(
        ggml_moe_cache_policy * st, uint32_t completed_pass) {
    for (int i = st->free_cnt - 1; i >= 0; i--) {
        const int32_t v = st->free_slot[i];
        if (st->free_pass[v] == completed_pass) {
            continue;
        }
        st->free_slot[i] = st->free_slot[--st->free_cnt];
        return v;
    }
    return -1;
}
