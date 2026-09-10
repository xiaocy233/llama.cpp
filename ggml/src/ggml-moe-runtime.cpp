#include "ggml-moe-runtime.h"
#include "ggml-moe-backend.h"
#include "ggml-moe-cache.h"
#include "ggml-backend-impl.h"
#include "ggml-moe-substitute.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <climits>

#ifndef GGML_SCHED_MAX_MOE_CACHES
#define GGML_SCHED_MAX_MOE_CACHES 128
#endif

#define GGML_SCHED_MAX_MOE_PRED 64

#define GGML_SCHED_MOE_STAGE 12

struct ggml_moe_slot_state : ggml_moe_cache_policy {
    struct ggml_moe_slot_cache cache;

    int n_expert_used;
    size_t expert_nb[3];        // bytes of one expert in each matrix (host and slots share nb)

    bool loc_dirty;             // device loc_map is behind slot_of_expert

    // Publish buffers stay pinned until the worker stream completes.
    ggml_backend_buffer_t pub_buf;
    int32_t  * loc_pub;         // [n_expert]
    uint32_t * pub_pass;        // [n_expert] worker sync pass that last published the expert

    uint64_t n_hit;
    uint64_t n_miss;
    uint64_t n_tokens;

    uint64_t n_load_sync, n_bytes_sync;    // brought in by the blocking fill on the consuming layer
    uint64_t n_load_pf,   n_bytes_pf;      // brought in ahead of use

    uint64_t n_pf_pred;                    // experts predicted for this layer
    uint64_t n_pf_useful;                  // of those, how many the layer then actually used

    int  dev_backend_id;          // backend owning slots, -1 if it could not be determined
    int  gate_slot;               // mailbox index, also the retire order
    uint32_t last_seen;           // last mailbox seq the worker consumed for this layer
    uint32_t head_last_seen;      // last seq consumed from the optional head mailbox

    int32_t  pred_batch[GGML_SCHED_MAX_MOE_PRED];
    int      n_pred_batch;

};

struct ggml_moe_worker {
    std::thread       thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> idle{true};   // a full pass found no new mailbox entry
};

struct ggml_moe_runtime {
    std::vector<ggml_backend_t> backends;
    int n_backends;
    bool has_moe_gate;
    int transfer_stream;
    struct ggml_moe_slot_state * moe_caches[GGML_SCHED_MAX_MOE_CACHES];
    int  n_moe_caches;
    bool moe_stats;   // env: GGML_SCHED_MOE_SLOT_STATS

    struct ggml_moe_gate_channel moe_ch;
    struct ggml_tensor *         moe_seq_t;
    uint32_t                     moe_seq;
    int                          moe_krel_tok;   // decode tokens whose gates the device releases
    struct ggml_moe_worker *     moe_worker;
    std::atomic<bool> *          moe_substitute_enabled;
    bool                         moe_host_gate;  // the backend runs its gates on the host (Metal)
    uint32_t                     moe_pass;       // bumped on every worker stream sync

    // Stage pageable weights before copying them past a parked device gate.
    ggml_backend_buffer_t        moe_stage_buf;
    char *                       moe_stage[GGML_SCHED_MOE_STAGE];
    uint32_t                     moe_stage_pass[GGML_SCHED_MOE_STAGE];
    size_t                       moe_stage_sz;
    int                          moe_stage_i;

};

ggml_moe_runtime * ggml_moe_runtime_new(ggml_backend_t * backends, int n_backends) {
    auto * sched = new ggml_moe_runtime{};
    sched->backends.assign(backends, backends + n_backends);
    sched->n_backends = n_backends;
    sched->moe_substitute_enabled = new std::atomic<bool>(false);
    const char * stats = getenv("GGML_SCHED_MOE_SLOT_STATS");
    sched->moe_stats = stats && atoi(stats) != 0;
    sched->moe_pass = 1;
    const char * release = getenv("GGML_SCHED_MOE_KREL");
    sched->moe_krel_tok = release ? atoi(release) : 0;
    return sched;
}

bool ggml_moe_runtime_active(const ggml_moe_runtime * sched) { return sched->n_moe_caches > 0; }
bool ggml_moe_runtime_drain(const ggml_moe_runtime * sched) { return sched->moe_stats && sched->n_moe_caches > 0; }
void ggml_moe_runtime_set_decode(ggml_moe_runtime * sched, bool decode) { sched->has_moe_gate = decode; }

static const char * ggml_moe_runtime_host_ptr(
        const struct ggml_moe_slot_state * st, int m, int32_t e) {
    GGML_ASSERT(e >= 0 && e < st->n_expert);
    return (const char *) st->cache.src[m]->data + (size_t) e * st->expert_nb[m];
}

const struct ggml_moe_slot_state * ggml_moe_runtime_bank_of(
        ggml_moe_runtime * sched, const struct ggml_tensor * src) {
    for (int c = 0; c < sched->n_moe_caches; c++) {
        for (int m = 0; m < 3; m++) {
            if (sched->moe_caches[c]->cache.src[m] == src) {
                return sched->moe_caches[c];
            }
        }
    }

    return NULL;
}

bool ggml_moe_runtime_bank_range(
        const struct ggml_moe_slot_state * st, const struct ggml_tensor * src,
        int * e, size_t * off, size_t * size) {
    if (st == NULL) {
        if (*e != 0) {
            return false;
        }
        *e    = 1;
        *off  = 0;
        *size = ggml_nbytes(src);
        return true;
    }

    while (*e < st->n_expert && st->slot_of_expert[*e] >= 0) {
        (*e)++;
    }
    if (*e >= st->n_expert) {
        return false;
    }

    const int first = *e;
    while (*e < st->n_expert && st->slot_of_expert[*e] < 0) {
        (*e)++;
    }

    *off  = (size_t) first * src->nb[2];
    *size = (size_t) (*e - first) * src->nb[2];

    return true;
}

static void ggml_moe_runtime_publish_loc(struct ggml_moe_slot_state * st) {
    if (!st->loc_dirty || st->cache.loc_map == NULL || st->cache.loc_map->data == NULL) {
        return;
    }

    std::vector<int32_t> loc((size_t) st->n_expert);
    for (int e = 0; e < st->n_expert; e++) {
        loc[e] = st->slot_of_expert[e] >= 0 ? st->slot_of_expert[e] : st->n_slots + e;
    }
    ggml_backend_tensor_set(st->cache.loc_map, loc.data(), 0, (size_t) st->n_expert * sizeof(int32_t));

    st->loc_dirty = false;
}

#define GGML_SCHED_MOE_POOL 2

static void ggml_moe_runtime_sync(ggml_moe_runtime * sched, ggml_backend_t bk) {
    bk->iface.synchronize_stream(bk, sched->transfer_stream);
    sched->moe_pass++;
}

static void ggml_moe_runtime_release(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st, uint32_t seq, bool wait) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];
    if (wait || sched->moe_ch.gate_signal == NULL) {
        if (st->n_tokens <= (uint64_t) sched->moe_krel_tok && bk->iface.moe_gate_release_stream != NULL) {
            bk->iface.moe_gate_release_stream(bk, sched->transfer_stream, st->gate_slot, seq);
        }
        ggml_moe_runtime_sync(sched, bk);
        std::atomic_thread_fence(std::memory_order_release);
        sched->moe_ch.release[st->gate_slot] = seq;
    }
    if (sched->moe_ch.gate_signal != NULL) {
        sched->moe_ch.gate_signal(sched->moe_ch.gate_ctx, st->gate_slot, seq);
    }
}

static const char * ggml_moe_runtime_stage(
        ggml_moe_runtime * sched, ggml_backend_t bk, const char * src, size_t nb) {
    if (sched->moe_stage_buf == NULL || nb > sched->moe_stage_sz) {
        return src;
    }
    const int i = sched->moe_stage_i;
    if (sched->moe_stage_pass[i] == sched->moe_pass) {
        ggml_moe_runtime_sync(sched, bk);
    }
    memcpy(sched->moe_stage[i], src, nb);
    sched->moe_stage_pass[i] = sched->moe_pass;
    sched->moe_stage_i       = (i + 1) % GGML_SCHED_MOE_STAGE;
    return sched->moe_stage[i];
}

static void ggml_moe_runtime_copy_expert(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st,
        ggml_backend_t bk, int32_t e, int32_t v, bool is_pf) {
    for (int m = 0; m < 3; m++) {
        if (st->cache.src[m] == NULL) {
            continue;
        }
        const size_t nb  = st->expert_nb[m];
        if (st->cache.src_fd >= 0) {
            const bool ok = bk->iface.set_tensor_async_file(bk, sched->transfer_stream,
                    st->cache.slots[m], st->cache.src_fd,
                    st->cache.src_off[m] + (uint64_t) e * nb, (size_t) v * nb, nb);
            GGML_ASSERT(ok);
        } else {
            const char * src = ggml_moe_runtime_host_ptr(st, m, e);
            if (!st->cache.pinned) {
                src = ggml_moe_runtime_stage(sched, bk, src, nb);
            }
            bk->iface.set_tensor_async_stream(bk, sched->transfer_stream,
                    st->cache.slots[m], src, (size_t) v * nb, nb);
        }
        if (is_pf) {
            st->n_bytes_pf += nb;
        } else {
            st->n_bytes_sync += nb;
        }
    }
    st->slot_of_expert[e] = v;
    st->expert_of_slot[v] = e;
    st->fill_seq[v]       = st->epoch;
}

static void ggml_moe_runtime_publish_row(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st, ggml_backend_t bk, int32_t e) {
    if (st->pub_pass[e] == sched->moe_pass) {
        ggml_moe_runtime_sync(sched, bk);
    }
    st->loc_pub[e]  = st->slot_of_expert[e] >= 0 ? st->slot_of_expert[e] : st->n_slots + e;
    st->pub_pass[e] = sched->moe_pass;

    bk->iface.set_tensor_async_stream(bk, sched->transfer_stream,
            st->cache.loc_map, &st->loc_pub[e], (size_t) e * sizeof(int32_t), sizeof(int32_t));
}

static void ggml_moe_runtime_prefetch(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st,
        const int32_t * experts, int n_experts) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    for (int i = 0; i < n_experts; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= st->n_expert || st->slot_of_expert[e] >= 0) {
            continue;
        }
        int32_t v = ggml_moe_cache_take(st, sched->moe_pass);
        if (v < 0) {
            v = ggml_moe_cache_victim(st, /* same_tok = */ true);
            if (v < 0) {
                continue;
            }
            const int32_t old = st->expert_of_slot[v];
            st->slot_of_expert[old] = -1;
            ggml_moe_runtime_publish_row(sched, st, bk, old);
        }

        ggml_moe_runtime_copy_expert(sched, st, bk, e, v, /* is_pf = */ true);
        st->n_load_pf++;
        ggml_moe_runtime_publish_row(sched, st, bk, e);
    }
}

static uint32_t ggml_moe_payload_hash(const struct ggml_moe_gate_entry * box, int n_expert) {
    uint32_t hash = 2166136261u;
    GGML_ASSERT(box->n_ids + box->n_pred <= GGML_MOE_GATE_MAX_IDS);
    for (uint32_t i = 0; i < box->n_ids + box->n_pred; i++) {
        hash = (hash ^ (uint32_t) box->ids[i]) * 16777619u;
    }
    for (int i = 0; i < n_expert; i++) {
        uint32_t bits;
        memcpy(&bits, &box->router_probs[i], sizeof(bits));
        hash = (hash ^ bits) * 16777619u;
    }
    return hash;
}

static void ggml_moe_runtime_service(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st,
        const struct ggml_moe_gate_entry * box) {
    struct ggml_moe_gate_entry adjusted;
    struct ggml_moe_gate_entry verified;
    const bool has_substitution = box->mode == GGML_MOE_GATE_MODE_SUBSTITUTE;
    ggml_moe_substitute_result selection{};
    const int32_t * original_ids = box->ids;
    if (has_substitution) {
        const uint32_t expected_seq = box->seq;
        const int64_t verify_begin = ggml_time_us();
        while (ggml_moe_payload_hash(box, st->n_expert) != box->payload_hash) {
            GGML_ASSERT(ggml_time_us() - verify_begin < 1000000);
            std::this_thread::yield();
            std::atomic_thread_fence(std::memory_order_acquire);
            memcpy(&verified, &sched->moe_ch.ring[st->gate_slot], sizeof(verified));
            std::atomic_thread_fence(std::memory_order_acquire);
            GGML_ASSERT(verified.seq == expected_seq);
            box = &verified;
        }
        original_ids = box->ids;
        adjusted = *box;
        const int n = (int) box->n_ids;
        ggml_moe_substitute_host(original_ids, box->router_probs, st->slot_of_expert, n, st->n_expert,
                st->n_slots, !sched->moe_substitute_enabled->load(std::memory_order_relaxed)
                    ? 0.0f : box->substitute_threshold, selection);
        for (int i = 0; i < n; i++) {
            adjusted.ids[i] = selection.ids[i];
            if (selection.replaced_mask & (1u << i)) {
                adjusted.miss_mask &= ~(1u << i);
            }
        }
        memcpy(sched->moe_ch.ring[st->gate_slot].selected_ids, selection.ids, n * sizeof(int32_t));
        std::atomic_thread_fence(std::memory_order_release);
        box = &adjusted;
    }
    const int32_t * ids    = box->ids;
    const int       n_ids  = std::min<int>(box->n_ids, st->n_expert_used);
    const int       n_pred = std::min<int>(box->n_pred, GGML_SCHED_MAX_MOE_PRED);
    const uint32_t  seq    = box->seq;

    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    st->n_tokens++;

    if (st->n_pred_batch > 0) {
        if (sched->moe_stats) {
            st->n_pf_pred += (uint64_t) st->n_pred_batch;
            for (int j = 0; j < st->n_pred_batch; j++) {
                for (int i = 0; i < n_ids; i++) {
                    if (original_ids[i] == st->pred_batch[j]) {
                        st->n_pf_useful++;
                        break;
                    }
                }
            }
        }
        st->n_pred_batch = 0;
    }

    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && e < st->n_expert && st->slot_of_expert[e] >= 0) {
            st->pinned[st->slot_of_expert[e]] = st->epoch;
        }
    }

    int n_hit  = 0;
    int n_miss = 0;

    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e < 0 || e >= st->n_expert) {
            continue;
        }

        int32_t v = st->slot_of_expert[e];
        if (v >= 0) {
            st->freq[e]++;
            st->pinned[v] = st->epoch;
            n_hit++;
            continue;
        }

        v = ggml_moe_cache_take(st, sched->moe_pass);
        if (v < 0) {
            v = ggml_moe_cache_victim(st, /* same_tok = */ false);
            GGML_ASSERT(v >= 0);
            const int32_t old = st->expert_of_slot[v];
            st->slot_of_expert[old] = -1;
            ggml_moe_runtime_publish_row(sched, st, bk, old);
        }

        ggml_moe_runtime_copy_expert(sched, st, bk, e, v, /* is_pf = */ false);
        ggml_moe_runtime_publish_row(sched, st, bk, e);
        st->n_load_sync++;
        st->freq[e]++;
        st->pinned[v] = st->epoch;
        n_miss++;
    }

    ggml_moe_runtime_release(sched, st, seq, box->miss_mask != 0 || n_miss > 0);

    st->n_hit  += n_hit;
    st->n_miss += n_miss;
    st->epoch++;

    if (n_pred > 0 && st->cache.pred_target >= 0) {
        struct ggml_moe_slot_state * target = NULL;
        for (int c = 0; c < sched->n_moe_caches; c++) {
            if (sched->moe_caches[c]->cache.layer == st->cache.pred_target) {
                target = sched->moe_caches[c];
                break;
            }
        }
        if (target != NULL) {
            memcpy(target->pred_batch, ids + n_ids, (size_t) n_pred * sizeof(int32_t));
            target->n_pred_batch = n_pred;
            ggml_moe_runtime_prefetch(sched, target, target->pred_batch, n_pred);
        }
    }
}

static void ggml_moe_runtime_service_head(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * target,
        const struct ggml_moe_gate_entry * box) {
    GGML_ASSERT(box->n_ids == 0);
    const int n_pred = std::min<int>(
            std::min<int>(box->n_pred, target->cache.head_n_pred), GGML_SCHED_MAX_MOE_PRED);
    if (n_pred > 0) {
        memcpy(target->pred_batch, box->ids, (size_t) n_pred * sizeof(int32_t));
        target->n_pred_batch = n_pred;
        ggml_moe_runtime_prefetch(sched, target, target->pred_batch, n_pred);
    }

}

static void ggml_moe_runtime_worker(ggml_moe_runtime * sched) {
    struct ggml_moe_gate_entry box;

    while (!sched->moe_worker->stop.load(std::memory_order_relaxed)) {
        bool did_work = false;

        for (int c = 0; c < sched->n_moe_caches; c++) {
            struct ggml_moe_slot_state * st = sched->moe_caches[c];
            if (st->cache.head_gate_slot < 0) {
                continue;
            }

            const struct ggml_moe_gate_entry * live =
                    &sched->moe_ch.ring[st->cache.head_gate_slot];
            const uint32_t seq = ((volatile const uint32_t *) &live->seq)[0];
            if (seq <= st->head_last_seen) {
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            memcpy(&box, live, sizeof(box));
            std::atomic_thread_fence(std::memory_order_acquire);
            if (((volatile const uint32_t *) &live->seq)[0] != seq) {
                continue;
            }
            box.seq = seq;
            st->head_last_seen = seq;
            sched->moe_worker->idle.store(false, std::memory_order_release);
            ggml_moe_runtime_service_head(sched, st, &box);
            did_work = true;
        }

        for (int c = 0; c < sched->n_moe_caches; c++) {
            struct ggml_moe_slot_state * st = sched->moe_caches[c];

            const struct ggml_moe_gate_entry * live = &sched->moe_ch.ring[st->gate_slot];

            const uint32_t seq = ((volatile const uint32_t *) &live->seq)[0];
            if (seq <= st->last_seen) {
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);

            memcpy(&box, live, sizeof(box));
            std::atomic_thread_fence(std::memory_order_acquire);
            if (((volatile const uint32_t *) &live->seq)[0] != seq) {
                continue;   // the device moved on mid-read, take it next round
            }
            box.seq = seq;

            if (st->cache.head_gate_slot >= 0 && st->head_last_seen < seq) {
                st->head_last_seen = seq;
            }

            st->last_seen = seq;

            sched->moe_worker->idle.store(false, std::memory_order_release);
            ggml_moe_runtime_service(sched, st, &box);
            did_work = true;
        }

        if (!did_work) {
            sched->moe_worker->idle.store(true, std::memory_order_release);
            for (int i = 0; i < 2000; i++) {
                std::atomic_signal_fence(std::memory_order_acquire);
            }
            std::this_thread::yield();
        }
    }
}

void ggml_moe_runtime_quiesce(ggml_moe_runtime * sched) {
    if (sched->moe_worker == NULL) {
        return;
    }

    while (!sched->moe_worker->idle.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    ggml_moe_runtime_sync(sched, sched->backends[sched->moe_caches[0]->dev_backend_id]);
}

void ggml_moe_runtime_publish_split(
        ggml_moe_runtime * sched, const ggml_cgraph * graph) {
    if (sched->has_moe_gate) {
        return;   // decode, the worker owns the table
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[4] == NULL) {
            continue;
        }

        for (int c = 0; c < sched->n_moe_caches; c++) {
            struct ggml_moe_slot_state * st = sched->moe_caches[c];
            if (st->cache.loc_map != node->src[4]) {
                continue;
            }
            ggml_moe_runtime_publish_loc(st);
            break;
        }
    }
}

static void ggml_moe_runtime_warm(ggml_moe_runtime * sched, struct ggml_moe_slot_state * st) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    for (int m = 0; m < 3; m++) {
        if (st->cache.src[m] == NULL) {
            continue;
        }
        const size_t nb  = st->expert_nb[m];
        if (st->cache.src_fd >= 0) {
            const bool ok = bk->iface.set_tensor_async_file(bk, sched->transfer_stream,
                    st->cache.slots[m], st->cache.src_fd, st->cache.src_off[m], 0, nb);
            GGML_ASSERT(ok);
            continue;
        }
        const char * src = ggml_moe_runtime_host_ptr(st, m, 0);
        if (!st->cache.pinned) {
            src = ggml_moe_runtime_stage(sched, bk, src, nb);
        }
        bk->iface.set_tensor_async_stream(bk, sched->transfer_stream, st->cache.slots[m], src, 0, nb);
    }
    ggml_moe_runtime_publish_row(sched, st, bk, 0);

    bk->iface.synchronize_stream(bk, sched->transfer_stream);
    sched->moe_pass++;
}

void ggml_moe_runtime_new_token(ggml_moe_runtime * sched) {
    if (!sched->has_moe_gate || sched->moe_seq_t == NULL) {
        return;
    }

    if (sched->has_moe_gate && sched->moe_worker == NULL) {
        ggml_backend_synchronize(sched->backends[sched->moe_caches[0]->dev_backend_id]);

        ggml_moe_runtime_warm(sched, sched->moe_caches[0]);
        for (int c = 0; c < sched->n_moe_caches; c++) {
            if (!sched->moe_caches[c]->cache.pinned) {
                ggml_moe_runtime_warm(sched, sched->moe_caches[c]);
                break;
            }
        }

        sched->moe_worker = new ggml_moe_worker();
        sched->moe_worker->thread = std::thread(ggml_moe_runtime_worker, sched);
    }

    GGML_ASSERT(sched->moe_seq < UINT32_MAX);
    sched->moe_seq++;
    const int32_t v = (int32_t) sched->moe_seq;
    ggml_backend_tensor_set(sched->moe_seq_t, &v, 0, sizeof(v));
}

bool ggml_moe_runtime_add_moe_slot_cache(
        ggml_moe_runtime * sched, const struct ggml_moe_slot_cache * cache, int n_expert_used) {
    GGML_ASSERT(sched != NULL);
    GGML_ASSERT(cache != NULL);

    if (sched->n_moe_caches >= GGML_SCHED_MAX_MOE_CACHES) {
        return false;
    }

    if (sched->n_moe_caches >= GGML_MOE_GATE_MAX_LAYERS) {
        return false;
    }

    const int n_expert = cache->n_expert;
    const int n_slots  = cache->n_slots;
    if (n_expert <= 0 || n_slots <= 0) {
        return false;
    }

    size_t nb[3] = { 0, 0, 0 };

    for (int m = 0; m < 3; m++) {
        if (cache->src[m] == NULL) {
            if (cache->slots[m] != NULL) {
                return false;
            }
            continue;
        }
        if (cache->slots[m] == NULL) {
            return false;
        }
        if (cache->src[m]->ne[0] != cache->slots[m]->ne[0] ||
            cache->src[m]->ne[1] != cache->slots[m]->ne[1] ||
            cache->src[m]->type  != cache->slots[m]->type) {
            return false;
        }
        if (cache->src[m]->buffer == NULL || !ggml_backend_buffer_is_host(cache->src[m]->buffer)) {
            return false;
        }
        if ((int) cache->src[m]->ne[2] != n_expert) {
            return false;
        }
        if ((int) cache->slots[m]->ne[2] != n_slots) {
            return false;
        }
        nb[m] = cache->src[m]->nb[2];
    }

    if (nb[0] == 0 && nb[1] == 0 && nb[2] == 0) {
        return false;
    }
    if (n_slots < n_expert_used || n_slots > n_expert) {
        return false;
    }
    if (cache->loc_map == NULL || cache->loc_map->type != GGML_TYPE_I32 ||
            (int) ggml_nelements(cache->loc_map) != n_expert) {
        return false;
    }
    if (n_slots < n_expert_used + GGML_SCHED_MOE_POOL) {
        return false;
    }
    if (cache->n_pred > GGML_SCHED_MAX_MOE_PRED) {
        return false;
    }
    const bool has_head = cache->head_pred_w != NULL && cache->head_n_pred > 0;
    if (has_head && cache->head_n_pred > GGML_SCHED_MAX_MOE_PRED) {
        return false;
    }
    if (has_head && cache->head_pred_source < 0) {
        return false;
    }
    if (has_head) {
        for (int c = 0; c < sched->n_moe_caches; c++) {
            if (sched->moe_caches[c]->cache.head_gate_slot >= 0) {
                return false;
            }
        }
    }

    struct ggml_moe_slot_state * st =
        (struct ggml_moe_slot_state *) calloc(1, sizeof(struct ggml_moe_slot_state));
    st->cache         = *cache;
    st->n_expert      = n_expert;
    st->n_slots       = n_slots;
    st->n_expert_used = n_expert_used;
    for (int m = 0; m < 3; m++) {
        st->expert_nb[m] = nb[m];
    }
    st->slot_of_expert = (int32_t  *) malloc(sizeof(int32_t)  * n_expert);
    st->expert_of_slot = (int32_t  *) malloc(sizeof(int32_t)  * n_slots);
    st->freq           = (uint32_t *) calloc(n_expert, sizeof(uint32_t));
    st->pinned         = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));
    st->free_slot      = (int32_t  *) malloc(sizeof(int32_t)  * n_slots);
    st->free_pass      = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));
    st->fill_seq       = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));

    for (int e = 0; e < n_expert; e++) { st->slot_of_expert[e] = -1; }
    for (int v = 0; v < n_slots;  v++) { st->expert_of_slot[v] = -1; st->free_slot[v] = v; }
    st->free_cnt = n_slots;

    st->epoch     = 1;   // 0 means "never pinned", so start above it
    st->loc_dirty = true;
    st->gate_slot = sched->n_moe_caches;
    st->cache.gate_slot = st->gate_slot;   // the graph builder reads it back through find()
    st->cache.head_gate_slot = has_head ? GGML_MOE_GATE_MAX_LAYERS - 1 : -1;

    st->dev_backend_id = -1;
    for (int b = 0; b < sched->n_backends; b++) {
        if (cache->slots[0] && cache->slots[0]->buffer &&
            ggml_backend_supports_buft(sched->backends[b], cache->slots[0]->buffer->buft)) {
            st->dev_backend_id = b;
            break;
        }
    }
    if (st->dev_backend_id < 0) {
        free(st);
        return false;
    }

    ggml_backend_t bk = sched->backends[st->dev_backend_id];
    if (bk->iface.set_tensor_async_stream == NULL || bk->iface.synchronize_stream == NULL ||
            bk->iface.moe_gate_channel == NULL) {
        free(st);
        return false;
    }
    if (cache->src_fd >= 0 && bk->iface.set_tensor_async_file == NULL) {
        free(st);
        return false;
    }
    if (sched->n_moe_caches == 0 && !bk->iface.moe_gate_channel(bk, &sched->moe_ch)) {
        free(st);
        return false;
    }
    if (sched->n_moe_caches == 0) {
        for (int i = 0; i < GGML_MOE_GATE_MAX_LAYERS; i++) {
            sched->moe_seq = std::max(sched->moe_seq, sched->moe_ch.ring[i].seq);
            sched->moe_seq = std::max(sched->moe_seq, (uint32_t) sched->moe_ch.release[i]);
        }
    }
    st->last_seen = st->head_last_seen = sched->moe_seq;
    if (st->cache.head_gate_slot >= 0) {
        if (sched->moe_ch.gate_events == NULL) {
            st->cache.head_pred_w      = NULL;
            st->cache.head_n_pred      = 0;
            st->cache.head_pred_source = -1;
            st->cache.head_gate_slot   = -1;
        }
    }

    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(bk));
    auto get_caps = (ggml_moe_backend_caps_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_get_caps");
    GGML_ASSERT(get_caps);
    const auto caps = get_caps(ggml_backend_get_device(bk));
    sched->transfer_stream = caps.transfer_stream;
    sched->moe_host_gate = caps.host_copies;

    if (sched->moe_host_gate) {
        st->cache.pinned = true;
    }

    ggml_backend_buffer_type_t buft_pin = ggml_backend_dev_host_buffer_type(bk->device);
    if (buft_pin == NULL) {
        buft_pin = ggml_backend_cpu_buffer_type();
    }
    st->pub_buf = buft_pin != NULL ? ggml_backend_buft_alloc_buffer(buft_pin, (size_t) n_expert * sizeof(int32_t)) : NULL;
    if (st->pub_buf == NULL) {
        free(st);
        return false;
    }
    st->loc_pub  = (int32_t  *) ggml_backend_buffer_get_base(st->pub_buf);
    st->pub_pass = (uint32_t *) calloc(n_expert, sizeof(uint32_t));

    if (!cache->pinned) {
        const size_t need = std::max(std::max(nb[0], nb[1]), nb[2]);
        if (need > sched->moe_stage_sz) {
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft_pin, need * GGML_SCHED_MOE_STAGE);
            if (buf == NULL) {
                ggml_backend_buffer_free(st->pub_buf);
                free(st->pub_pass);
                free(st);
                return false;
            }
            ggml_backend_buffer_free(sched->moe_stage_buf);
            sched->moe_stage_buf = buf;
            sched->moe_stage_sz  = need;
            char * base = (char *) ggml_backend_buffer_get_base(buf);
            for (int i = 0; i < GGML_SCHED_MOE_STAGE; i++) {
                sched->moe_stage[i]      = base + (size_t) i * need;
                sched->moe_stage_pass[i] = 0;
            }
        }
    }

    ggml_moe_runtime_publish_loc(st);

    sched->moe_caches[sched->n_moe_caches++] = st;

    return true;
}

void ggml_moe_runtime_set_moe_gate_seq(ggml_moe_runtime * sched, struct ggml_tensor * seq) {
    GGML_ASSERT(sched != NULL);
    sched->moe_seq_t = seq;
}

void ggml_moe_runtime_set_moe_substitute_enabled(ggml_moe_runtime * sched, bool enabled) {
    if (sched->moe_substitute_enabled->load(std::memory_order_relaxed) != enabled) {
        for (int i = 0; i < sched->n_backends; i++) { ggml_backend_synchronize(sched->backends[i]); }
        sched->moe_substitute_enabled->store(enabled, std::memory_order_relaxed);
    }
}

const struct ggml_moe_slot_cache * ggml_moe_runtime_find_moe_slot_cache(
        ggml_moe_runtime * sched, const struct ggml_tensor * src) {
    if (sched == NULL || src == NULL) {
        return NULL;
    }
    for (int i = 0; i < sched->n_moe_caches; i++) {
        const struct ggml_moe_slot_cache * c = &sched->moe_caches[i]->cache;
        for (int m = 0; m < 3; m++) {
            if (c->src[m] == src) {
                return c;
            }
        }
    }
    return NULL;
}

const struct ggml_moe_slot_cache * ggml_moe_runtime_find_moe_head_predictor(
        ggml_moe_runtime * sched, int source_layer) {
    if (sched == NULL || source_layer < 0) {
        return NULL;
    }
    for (int i = 0; i < sched->n_moe_caches; i++) {
        const struct ggml_moe_slot_cache * c = &sched->moe_caches[i]->cache;
        if (c->head_pred_source == source_layer && c->head_pred_w != NULL && c->head_gate_slot >= 0) {
            return c;
        }
    }
    return NULL;
}

static void ggml_moe_runtime_stats_report(ggml_moe_runtime * sched) {
    GGML_LOG_DEBUG("\n");
    GGML_LOG_DEBUG("MoE slot cache, decode only: %d layers, one load = one expert (up+gate+down)\n",
            sched->n_moe_caches);
    GGML_LOG_DEBUG("  %-5s %6s %8s %8s %11s %11s %8s %11s %11s\n",
            "layer", "slots", "tokens", "hit", "load_sync", "load_pf", "pf_prec", "MiB_sync", "MiB_pf");

    uint64_t t_hit = 0, t_use = 0, t_ls = 0, t_lp = 0, t_bs = 0, t_bp = 0, t_pp = 0, t_pu = 0, max_tokens = 0;

    for (int i = 0; i < sched->n_moe_caches; i++) {
        const struct ggml_moe_slot_state * st = sched->moe_caches[i];
        if (st->n_tokens == 0) {
            continue;
        }
        const double   tok = (double) st->n_tokens;
        const uint64_t use = st->n_hit + st->n_miss;

        GGML_LOG_DEBUG("  %-5d %6d %8llu %8.4f %11.3f %11.3f %8.4f %11.1f %11.1f\n",
                st->cache.layer, st->n_slots, (unsigned long long) st->n_tokens,
                use ? (double) st->n_hit / (double) use : 0.0,
                (double) st->n_load_sync / tok, (double) st->n_load_pf / tok,
                st->n_pf_pred ? (double) st->n_pf_useful / (double) st->n_pf_pred : 0.0,
                (double) st->n_bytes_sync / 1024.0 / 1024.0,
                (double) st->n_bytes_pf   / 1024.0 / 1024.0);

        t_hit += st->n_hit;        t_use += use;
        t_ls  += st->n_load_sync;  t_lp  += st->n_load_pf;
        t_bs  += st->n_bytes_sync; t_bp  += st->n_bytes_pf;
        t_pp  += st->n_pf_pred;    t_pu  += st->n_pf_useful;
        max_tokens = std::max(max_tokens, st->n_tokens);
    }

    if (max_tokens > 0) {
        const double tok = (double) max_tokens;
        GGML_LOG_DEBUG("  %-5s %6s %8llu %8.4f %11.3f %11.3f %8.4f %11.1f %11.1f\n",
                "TOTAL", "", (unsigned long long) max_tokens,
                t_use ? (double) t_hit / (double) t_use : 0.0,
                (double) t_ls / tok, (double) t_lp / tok,
                t_pp ? (double) t_pu / (double) t_pp : 0.0,
                (double) t_bs / 1024.0 / 1024.0, (double) t_bp / 1024.0 / 1024.0);
    }

}
void ggml_moe_runtime_free(ggml_moe_runtime * sched) {
    if (sched->moe_worker != NULL) {
        for (int i = 0; i < sched->n_backends; i++) { ggml_backend_synchronize(sched->backends[i]); }
        sched->moe_worker->stop.store(true, std::memory_order_relaxed);
        sched->moe_worker->thread.join();
        ggml_moe_runtime_sync(sched, sched->backends[sched->moe_caches[0]->dev_backend_id]);
        delete sched->moe_worker;
        sched->moe_worker = NULL;
    }
    if (sched->moe_stats && sched->n_moe_caches > 0) {
        ggml_moe_runtime_stats_report(sched);
    }
    for (int i = 0; i < sched->n_moe_caches; i++) {
        struct ggml_moe_slot_state * st = sched->moe_caches[i];
        free(st->slot_of_expert);
        free(st->expert_of_slot);
        free(st->freq);
        free(st->pinned);
        free(st->free_slot);
        free(st->free_pass);
        free(st->fill_seq);
        free(st->pub_pass);
        ggml_backend_buffer_free(st->pub_buf);
        free(st);
    }
    sched->n_moe_caches = 0;
    ggml_backend_buffer_free(sched->moe_stage_buf);
    delete sched->moe_substitute_enabled;
    delete sched;
}
