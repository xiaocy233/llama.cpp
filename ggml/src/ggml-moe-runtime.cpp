#include "ggml-moe-runtime.h"
#include "ggml-moe-backend.h"
#include "ggml-moe-cache.h"
#include "ggml-moe-balance.h"
#include "ggml-backend-impl.h"
#include "ggml-moe-substitute.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#ifndef GGML_SCHED_MAX_MOE_CACHES
#define GGML_SCHED_MAX_MOE_CACHES 128
#endif

#define GGML_SCHED_MAX_MOE_PRED 64

#define GGML_SCHED_MOE_STAGE 12

struct ggml_moe_balance_profile {
    ggml_backend_dev_t device;
    std::string path;
    int threads;
    int top_k;
    bool pinned;
    size_t bytes;
    int64_t width;
    ggml_type types[3];
    ggml_moe_cpu_cost cpu_cost;
    double h2d_gbs;
};

struct ggml_moe_cpu_state {
    ggml_moe_cpu_cost cost;
    double h2d_gbs = 0;
    const ggml_tensor * staged_input = nullptr;
    std::atomic<uint32_t> staged_seq{0};
    int reserved_count = 0;
    int64_t join_us = 0;
    std::vector<uint64_t> last_used;
    std::vector<int32_t> ids;
    std::vector<int32_t> gpu_ids;
    std::vector<int> cold;
    uint64_t step = 0;
    uint64_t cpu_assignments = 0;
    uint64_t gpu_assignments = 0;
    uint64_t cold_experts = 0;
    uint64_t avoided_bytes = 0;
    int64_t cpu_us = 0;
    std::atomic<uint32_t> ready_seq{0};
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * input_ids = nullptr;
    ggml_tensor * output = nullptr;
    ggml_backend_buffer_t staging = nullptr;
    size_t staging_size = 0;
    float * source = nullptr;
    int32_t * rows = nullptr;
    uint32_t * complete = nullptr;
    ggml_backend_event_t input_ready = nullptr;
    ggml_backend_event_t output_done = nullptr;
    bool output_pending = false;
    int graph_cold = 0;
};

struct ggml_moe_cpu_worker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable wake;
    ggml_moe_cpu_state * job = nullptr;
    int n_used = 0;
    int64_t * measured_ready_us = nullptr;
    bool pending = false;
    std::atomic<bool> done{false};
    bool stop = false;
    ggml_status status = GGML_STATUS_SUCCESS;
};

struct ggml_moe_slot_state : ggml_moe_cache_policy {
    struct ggml_moe_slot_cache cache;

    ggml_moe_cpu_state * cpu;
    int n_expert_used;
    int cache_policy;
    int decay_interval;
    bool protect_predictions;
    int prefetch_budget;
    bool rank_votes;
    size_t expert_nb[3];        // bytes of one expert in each matrix (host and slots share nb)

    bool loc_dirty;             // device loc_map is behind slot_of_expert

    // Publish buffers stay pinned until the worker stream completes.
    ggml_backend_buffer_t pub_buf;
    int32_t  * loc_pub;         // [n_expert]
    uint32_t * pub_pass;        // [n_expert] worker sync pass that last published the expert

    uint64_t n_hit;
    uint64_t n_miss;
    uint64_t n_tokens;
    uint64_t n_batches;
    uint64_t n_dynamic_slot;
    uint64_t n_dynamic_bulk;

    uint64_t n_load_sync, n_bytes_sync;    // brought in by the blocking fill on the consuming layer
    uint64_t n_load_pf,   n_bytes_pf;      // brought in ahead of use

    uint64_t n_pf_pred;                    // experts predicted for this layer
    uint64_t n_pf_useful;                  // of those, how many the layer then actually used

    int  dev_backend_id;          // backend owning slots, -1 if it could not be determined
    int  gate_slot;               // mailbox index, also the retire order
    uint32_t last_seen;           // last mailbox seq the worker consumed for this layer
    uint32_t head_last_seen;      // last seq consumed from the optional head mailbox

    int32_t  pred_batch[GGML_MOE_GATE_MAX_IDS];
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
    std::mutex moe_mutex;
    int cpu_max_tokens;
    int cpu_min_tokens;
    int cpu_reuse_window;
    ggml_backend_t cpu_backend;
    ggml_threadpool_t cpu_threadpool;
    void (*cpu_threadpool_free)(ggml_threadpool_t);
    ggml_moe_cpu_worker * cpu_worker;
    bool cpu_balance;
    int cpu_threads;
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
    const char * hybrid = getenv("GGML_MOE_HYBRID");
    const bool supported = std::any_of(sched->backends.begin(), sched->backends.end(), [](ggml_backend_t backend) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        return ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cpu_merge") != nullptr;
    });
    if (hybrid && atoi(hybrid) != 0 && supported) {
        auto env_int = [](const char * key, int fallback, int lo, int hi) {
            const char * value = getenv(key);
            return std::max(lo, std::min(hi, value ? atoi(value) : fallback));
        };
        const char * policy = getenv("GGML_MOE_HYBRID_POLICY");
        sched->cpu_balance = policy && strcmp(policy, "balanced") == 0;
        sched->cpu_max_tokens = env_int("GGML_MOE_HYBRID_MAX_TOKENS", 8, 1, 8);
        sched->cpu_min_tokens = env_int("GGML_MOE_HYBRID_MIN_TOKENS", 2, 1, GGML_MOE_GATE_MAX_IDS + 1);
        sched->cpu_reuse_window = env_int("GGML_MOE_HYBRID_REUSE_WINDOW", 1, 0, 1024);
        for (auto backend : sched->backends) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                const auto dev = ggml_backend_get_device(backend);
                sched->cpu_backend = ggml_backend_dev_init(dev, nullptr);
                const auto reg = ggml_backend_dev_backend_reg(dev);
                auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                auto pool_new = (ggml_threadpool_t (*)(ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
                auto set_pool = (void (*)(ggml_backend_t, ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
                sched->cpu_threadpool_free = (void (*)(ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
                GGML_ASSERT(sched->cpu_backend && set_threads && pool_new && set_pool && sched->cpu_threadpool_free);
                const int threads = env_int("GGML_MOE_HYBRID_THREADS", 8, 1, GGML_MAX_N_THREADS);
                sched->cpu_threads = threads;
                auto params = ggml_threadpool_params_default(threads);
                params.poll = 0;
                sched->cpu_threadpool = pool_new(&params);
                GGML_ASSERT(sched->cpu_threadpool);
                set_threads(sched->cpu_backend, threads);
                set_pool(sched->cpu_backend, sched->cpu_threadpool);
                break;
            }
        }
        GGML_ASSERT(sched->cpu_backend != nullptr);

        GGML_LOG_INFO("MoE hybrid-filtered: policy=%s max_tokens=%d min_tokens=%d reuse_window=%d\n",
                sched->cpu_balance ? "balanced" : "legacy",
                sched->cpu_max_tokens, sched->cpu_min_tokens, sched->cpu_reuse_window);
    }
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

struct ggml_moe_slot_state * ggml_moe_runtime_bank_of(
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

bool ggml_moe_runtime_has_expert(const ggml_moe_slot_state * st, int expert) {
    GGML_ASSERT(st == NULL || (expert >= 0 && expert < st->n_expert));
    return st != NULL && st->slot_of_expert[expert] >= 0;
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
    // MMQ reads padded blocks past the last expert row.
    if (*e < st->n_expert) {
        *size += std::min<size_t>(src->nb[2], 512);
    }

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

static void ggml_moe_runtime_cpu_submit(ggml_moe_runtime * sched, ggml_moe_slot_state * st, int n_used = 0, int64_t * measured_ready_us = nullptr);
static bool ggml_moe_runtime_cpu_buffers(ggml_moe_slot_state * st, ggml_backend_t backend, const ggml_tensor * input);

static std::vector<int> ggml_moe_runtime_balance(ggml_moe_slot_state * st, const std::vector<int> & counts) {
    auto & cpu = *st->cpu;
    int hits = 0;
    std::vector<ggml_moe_balance_item> misses;
    for (int e = 0; e < st->n_expert; ++e) {
        if (!counts[e]) continue;
        if (st->slot_of_expert[e] >= 0) hits++;
        else misses.push_back({e, counts[e], cpu.last_used[e]});
    }
    const size_t bytes = st->expert_nb[0] + st->expert_nb[1] + st->expert_nb[2];
    return ggml_moe_balance_select(misses, st->n_slots - hits, cpu.cost, bytes, cpu.h2d_gbs);
}

static int32_t ggml_moe_runtime_cache_victim(
        const ggml_moe_slot_state * st, bool same_token, const uint8_t * protected_ids) {
    if (st->cache_policy == 0 && protected_ids == nullptr) {
        return ggml_moe_cache_victim(st, same_token);
    }
    int32_t victim = -1;
    uint32_t best = UINT32_MAX;
    uint32_t oldest = UINT32_MAX;
    for (int v = 0; v < st->n_slots; ++v) {
        const int e = st->expert_of_slot[v];
        if (e < 0 || st->pinned[v] == st->epoch || (same_token && st->fill_seq[v] >= st->epoch)) {
            continue;
        }
        if (protected_ids && protected_ids[e]) {
            continue;
        }
        const uint32_t recent = std::max(st->pinned[v], st->fill_seq[v]);
        const uint32_t score = st->cache_policy == 1 ? recent : st->freq[e];
        if (victim < 0 || score < best || (st->cache_policy == 2 && score == best && recent < oldest)) {
            victim = v;
            best = score;
            oldest = recent;
        }
    }
    return victim;
}

static void ggml_moe_runtime_prefetch(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * st,
        const int32_t * experts, int n_experts, int n_tokens) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    st->n_pred_batch = 0;
    std::vector<int> counts;
    const bool cpu_filter = st->cpu && n_tokens <= st->cache.cpu_expert_max_tokens;
    if (cpu_filter || (st->rank_votes && st->prefetch_budget != 0)) {
        counts.assign(st->n_expert, 0);
        for (int i = 0; i < n_experts; ++i) {
            if (experts[i] >= 0 && experts[i] < st->n_expert) counts[experts[i]]++;
        }
    }
    std::vector<int> cpu_pred;
    if (st->cache.cpu_expert_early && cpu_filter) cpu_pred = ggml_moe_runtime_balance(st, counts);
    for (int i = 0; i < n_experts; i++) {
        const int32_t e = experts[i];
        if (st->cache.cpu_expert_early) {
            if (std::find(cpu_pred.begin(), cpu_pred.end(), e) != cpu_pred.end()) continue;
        } else if (cpu_filter && e >= 0 && e < st->n_expert && st->slot_of_expert[e] < 0) {
            const auto & cpu = *st->cpu;
            const bool recent = cpu.last_used[e] > 0 && cpu.step + 1 - cpu.last_used[e] <= (uint64_t) sched->cpu_reuse_window;
            if (counts[e] < sched->cpu_min_tokens && !recent) continue;
        }
        if (e >= 0 && e < st->n_expert &&
                std::find(st->pred_batch, st->pred_batch + st->n_pred_batch, e) == st->pred_batch + st->n_pred_batch) {
            st->pred_batch[st->n_pred_batch++] = e;
        }
    }
    if (st->rank_votes && !counts.empty()) {
        std::stable_sort(st->pred_batch, st->pred_batch + st->n_pred_batch,
                [&](int a, int b) { return counts[a] > counts[b]; });
    }
    experts = st->pred_batch;
    n_experts = st->n_pred_batch;
    if (st->prefetch_budget == 0) {
        return;
    }
    std::vector<uint8_t> protected_ids;
    if (st->protect_predictions) {
        protected_ids.resize(st->n_expert, 0);
        for (int i = 0; i < n_experts; ++i) {
            protected_ids[experts[i]] = 1;
        }
    }
    int n_copied = 0;
    for (int i = 0; i < n_experts; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= st->n_expert || st->slot_of_expert[e] >= 0) {
            continue;
        }
        if (st->prefetch_budget >= 0 && n_copied >= st->prefetch_budget) {
            break;
        }
        int32_t v = ggml_moe_cache_take(st, sched->moe_pass);
        if (v < 0) {
            v = ggml_moe_runtime_cache_victim(st, true, protected_ids.empty() ? nullptr : protected_ids.data());
            if (v < 0) {
                continue;
            }
            const int32_t old = st->expert_of_slot[v];
            st->slot_of_expert[old] = -1;
            ggml_moe_runtime_publish_row(sched, st, bk, old);
        }

        ggml_moe_runtime_copy_expert(sched, st, bk, e, v, /* is_pf = */ true);
        st->n_load_pf++;
        n_copied++;
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

static bool ggml_moe_runtime_fill(
        ggml_moe_runtime * sched, ggml_moe_slot_state * st,
        const int32_t * ids, int n_ids, const int32_t * original_ids, int n_tokens = -1) {
    GGML_ASSERT(n_ids >= 0 && (n_tokens >= 0 || (n_ids > 0 && n_ids % st->n_expert_used == 0)));
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    if (st->cache_policy == 2 && st->n_batches > 0 && st->n_batches % st->decay_interval == 0) {
        for (int e = 0; e < st->n_expert; ++e) {
            st->freq[e] >>= 1;
        }
    }
    st->n_tokens += n_tokens >= 0 ? n_tokens : n_ids / st->n_expert_used;
    st->n_batches++;

    if (st->n_pred_batch > 0) {
        st->n_pf_pred += (uint64_t) st->n_pred_batch;
        for (int j = 0; j < st->n_pred_batch; j++) {
            for (int i = 0; i < n_ids; i++) {
                if (original_ids[i] == st->pred_batch[j]) {
                    st->n_pf_useful++;
                    break;
                }
            }
        }
        st->n_pred_batch = 0;
    }

    int n_hit  = 0;
    int n_miss = 0;
    // Count residency before filling, including repeated assignments to the same expert.
    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && e < st->n_expert);
        if (st->slot_of_expert[e] >= 0) {
            st->pinned[st->slot_of_expert[e]] = st->epoch;
            n_hit++;
        } else {
            n_miss++;
        }
    }

    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e < 0 || e >= st->n_expert) {
            continue;
        }

        int32_t v = st->slot_of_expert[e];
        if (v >= 0) {
            st->freq[e]++;
            st->pinned[v] = st->epoch;
            continue;
        }

        v = ggml_moe_cache_take(st, sched->moe_pass);
        if (v < 0) {
            v = ggml_moe_runtime_cache_victim(st, false, nullptr);
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
    }

    st->n_hit += n_hit;
    st->n_miss += n_miss;
    return n_miss > 0;
}

static bool ggml_moe_runtime_select_cpu(ggml_moe_runtime * sched, ggml_moe_slot_state * st, const int32_t * ids, int n_ids, uint32_t seq) {
    auto & cpu = *st->cpu;
    cpu.step++;
    cpu.ids.assign(ids, ids + n_ids);
    cpu.cold.clear();
    std::vector<int> counts(st->n_expert, 0);
    for (int i = 0; i < n_ids; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < st->n_expert);
        counts[ids[i]]++;
    }
    const auto cpu_budget = st->cache.cpu_expert_early ? ggml_moe_runtime_balance(st, counts) : std::vector<int>{};
    std::vector<int> candidates;
    for (int e = 0; e < st->n_expert; ++e) {
        const bool recent = cpu.last_used[e] > 0 && cpu.step - cpu.last_used[e] <= (uint64_t) sched->cpu_reuse_window;
        const bool gpu = st->cache.cpu_expert_early ? std::find(cpu_budget.begin(), cpu_budget.end(), e) == cpu_budget.end() :
                st->slot_of_expert[e] >= 0 || counts[e] >= sched->cpu_min_tokens || recent;
        if (counts[e] && gpu) {
            candidates.push_back(e);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const bool a_cached = st->slot_of_expert[a] >= 0;
        const bool b_cached = st->slot_of_expert[b] >= 0;
        return a_cached != b_cached ? a_cached : counts[a] > counts[b];
    });
    if ((int) candidates.size() > st->n_slots) {
        candidates.resize(st->n_slots);
    }
    std::vector<bool> on_gpu(st->n_expert, false);
    for (int e : candidates) {
        on_gpu[e] = true;
    }
    std::vector<int32_t> hot;
    for (int i = 0; i < n_ids; ++i) {
        if (on_gpu[ids[i]]) {
            hot.push_back(ids[i]);
        } else {
            cpu.cold.push_back(i);
        }
    }
    st->n_pf_pred += st->n_pred_batch;
    for (int i = 0; i < st->n_pred_batch; ++i) {
        st->n_pf_useful += std::find(ids, ids + n_ids, st->pred_batch[i]) != ids + n_ids;
    }
    st->n_pred_batch = 0;
    if (st->cache.cpu_expert_early && !cpu.cold.empty()) {
        GGML_ASSERT(cpu.staged_seq.load(std::memory_order_acquire) == seq);
        GGML_ASSERT(ggml_moe_runtime_cpu_buffers(st, sched->backends[st->dev_backend_id], cpu.staged_input));
        ggml_moe_runtime_cpu_submit(sched, st);
    }
    const bool missed = ggml_moe_runtime_fill(sched, st, hot.data(), (int) hot.size(), ids, n_ids / st->n_expert_used);
    cpu.gpu_ids.resize(n_ids);
    for (int i = 0; i < n_ids; ++i) {
        cpu.gpu_ids[i] = on_gpu[ids[i]] ? st->slot_of_expert[ids[i]] : -1;
    }
    for (int e = 0; e < st->n_expert; ++e) {
        if (counts[e]) {
            cpu.last_used[e] = cpu.step;
            if (!on_gpu[e]) {
                cpu.cold_experts++;
                cpu.avoided_bytes += st->expert_nb[0] + st->expert_nb[1] + st->expert_nb[2];
            }
        }
    }
    cpu.cpu_assignments += cpu.cold.size();
    cpu.gpu_assignments += hot.size();
    memcpy(sched->moe_ch.ring[st->gate_slot].selected_ids, cpu.gpu_ids.data(), n_ids * sizeof(int32_t));
    cpu.ready_seq.store(seq, std::memory_order_release);
    return missed;
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
    const int       n_ids  = (int) box->n_ids;
    const int       n_pred = (int) box->n_pred;
    const uint32_t  seq    = box->seq;

    GGML_ASSERT(n_ids > 0 && n_ids <= GGML_MOE_GATE_MAX_IDS && n_ids % st->n_expert_used == 0);
    GGML_ASSERT(box->mode == GGML_MOE_GATE_MODE_HYBRID || ggml_moe_count_unique(ids, n_ids, st->n_expert) <= st->n_slots);
    GGML_ASSERT(n_ids + n_pred <= GGML_MOE_GATE_MAX_IDS);
    const bool missed = box->mode == GGML_MOE_GATE_MODE_HYBRID ?
        ggml_moe_runtime_select_cpu(sched, st, ids, n_ids, seq) :
        ggml_moe_runtime_fill(sched, st, ids, n_ids, original_ids);
    ggml_moe_runtime_release(sched, st, seq, box->miss_mask != 0 || missed);
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
            ggml_moe_runtime_prefetch(sched, target, ids + n_ids, n_pred, n_ids / st->n_expert_used);
        }
    }
}

static void ggml_moe_runtime_cpu_worker_run(ggml_moe_runtime * sched) {
    auto & worker = *sched->cpu_worker;
    std::unique_lock<std::mutex> lock(worker.mutex);
    for (;;) {
        worker.wake.wait(lock, [&] { return worker.stop || worker.pending; });
        if (worker.stop) return;
        auto & job = *worker.job;
        const int n_used = worker.n_used;
        worker.pending = false;
        lock.unlock();
        ggml_backend_event_synchronize(job.input_ready);
        const int64_t width = job.input->ne[0];
        for (size_t i = 0; i < job.cold.size(); ++i) {
            memcpy((float *) job.input->data + i * width, job.source + (job.cold[i] / n_used) * width, width * sizeof(float));
            ((int32_t *) job.input_ids->data)[i] = job.ids[job.cold[i]];
        }
        const int64_t begin = ggml_time_us();
        const ggml_status status = ggml_backend_graph_compute(sched->cpu_backend, job.graph);
        job.cpu_us += ggml_time_us() - begin;
        lock.lock();
        worker.status = status;
        std::atomic_thread_fence(std::memory_order_release);
        ((volatile uint32_t *) job.complete)[0] = status == GGML_STATUS_SUCCESS ? 1u : 2u;
        if (worker.measured_ready_us) *worker.measured_ready_us = ggml_time_us();
        worker.done = true;
        worker.wake.notify_all();
    }
}

static void ggml_moe_runtime_cpu_submit(ggml_moe_runtime * sched, ggml_moe_slot_state * st, int n_used, int64_t * measured_ready_us) {
    auto & cpu = *st->cpu;
    auto & worker = *sched->cpu_worker;
    std::lock_guard<std::mutex> lock(worker.mutex);
    GGML_ASSERT(worker.job == nullptr);
    worker.job = &cpu;
    worker.n_used = n_used > 0 ? n_used : st->n_expert_used;
    worker.measured_ready_us = measured_ready_us;
    ((volatile uint32_t *) cpu.complete)[0] = 0;
    worker.done = false;
    worker.pending = true;
    worker.wake.notify_all();
}

static bool ggml_moe_runtime_cpu_buffers(ggml_moe_slot_state * st, ggml_backend_t backend, const ggml_tensor * input) {
    auto & cpu = *st->cpu;
    const int count = cpu.cold.size();
    if (cpu.output_pending) {
        ggml_backend_event_synchronize(cpu.output_done);
        cpu.output_pending = false;
    }
    const auto pin = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    GGML_ASSERT(pin);
    const size_t source_size = st->cache.cpu_expert_early ? input->ne[0] * sizeof(float) * st->cache.cpu_expert_max_tokens : ggml_nbytes(input);
    const size_t size = source_size + GGML_MOE_GATE_MAX_IDS * sizeof(int32_t) + sizeof(uint32_t);
    if (size > cpu.staging_size) {
        ggml_backend_buffer_free(cpu.staging);
        cpu.staging = ggml_backend_buft_alloc_buffer(pin, size);
        if (!cpu.staging) return false;
        cpu.staging_size = size;
    }
    cpu.source = (float *) ggml_backend_buffer_get_base(cpu.staging);
    cpu.rows = (int32_t *) ((char *) cpu.source + source_size);
    cpu.complete = (uint32_t *) (cpu.rows + GGML_MOE_GATE_MAX_IDS);
    for (int i = 0; i < count; ++i) cpu.rows[i] = cpu.cold[i];
    if (!cpu.input_ready) {
        cpu.input_ready = ggml_backend_event_new(ggml_backend_get_device(backend));
        cpu.output_done = ggml_backend_event_new(ggml_backend_get_device(backend));
        GGML_ASSERT(cpu.input_ready && cpu.output_done);
    }
    if (cpu.graph_cold != count) {
        ggml_free(cpu.ctx);
        cpu.ctx = ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true});
        if (!cpu.ctx) return false;
        cpu.input = ggml_new_tensor_3d(cpu.ctx, GGML_TYPE_F32, input->ne[0], 1, count);
        cpu.input_ids = ggml_new_tensor_2d(cpu.ctx, GGML_TYPE_I32, 1, count);
        auto * up = ggml_mul_mat_id(cpu.ctx, st->cache.src[1], cpu.input, cpu.input_ids);
        auto * gate = ggml_mul_mat_id(cpu.ctx, st->cache.src[0], cpu.input, cpu.input_ids);
        cpu.output = ggml_mul_mat_id(cpu.ctx, st->cache.src[2], ggml_swiglu_split(cpu.ctx, gate, up), cpu.input_ids);
        ggml_set_input(cpu.input);
        ggml_set_input(cpu.input_ids);
        ggml_set_output(cpu.output);
        cpu.graph = ggml_new_graph_custom(cpu.ctx, 32, false);
        ggml_build_forward_expand(cpu.graph, cpu.output);
        if (!cpu.alloc) cpu.alloc = ggml_gallocr_new(pin);
        if (!ggml_gallocr_alloc_graph(cpu.alloc, cpu.graph)) return false;
        cpu.graph_cold = count;
    }
    return true;
}

static bool ggml_moe_runtime_bandwidth(ggml_moe_runtime * sched, ggml_moe_slot_state * st, ggml_backend_t backend) {
    auto & cpu = *st->cpu;
    const char * path = getenv("GGML_MOE_HYBRID_BW_PROFILE");
    if (!path || !*path) {
        GGML_LOG_ERROR("MoE latency: set GGML_MOE_HYBRID_BW_PROFILE to a file produced by llama-moe-bandwidth\n");
        return false;
    }
    const char * measure_env = getenv("GGML_MOE_HYBRID_BW_MEASURE");
    const bool measure = measure_env && atoi(measure_env) != 0;
    const int count = std::min(st->n_expert_used, st->n_expert);
    const size_t bytes = st->expert_nb[0] + st->expert_nb[1] + st->expert_nb[2];
    const int64_t width = st->cache.src[0]->ne[0];
    static std::mutex profile_mutex;
    static std::vector<ggml_moe_balance_profile> profiles;
    std::lock_guard<std::mutex> profile_lock(profile_mutex);
    const auto device = ggml_backend_get_device(backend);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(device, &props);
    const std::string device_id = props.device_id ? props.device_id : props.name;
    for (const auto & profile : profiles) {
        if (profile.device == device && profile.path == path && profile.threads == sched->cpu_threads && profile.top_k == count && profile.pinned == st->cache.pinned && profile.bytes == bytes && profile.width == width &&
                profile.types[0] == st->cache.src[0]->type && profile.types[1] == st->cache.src[1]->type && profile.types[2] == st->cache.src[2]->type) {
            cpu.cost = profile.cpu_cost;
            cpu.h2d_gbs = profile.h2d_gbs;
            return true;
        }
    }
    if (!measure) {
        std::ifstream file(path);
        std::string version;
        std::getline(file, version);
        if (version != ggml_moe_profile_version) {
            GGML_LOG_ERROR("MoE latency: cannot read v2 profile %s; run llama-moe-bandwidth again (v1 bandwidth profiles are not supported)\n", path);
            return false;
        }
        std::string id, description;
        int threads, top_k, pinned, gate_type, up_type, down_type;
        size_t expert_bytes;
        int64_t input_width;
        ggml_moe_cpu_cost cost;
        double h2d_gbs;
        std::array<double, 3> times_us;
        while (file >> std::quoted(id) >> std::quoted(description) >> threads >> top_k >> pinned >> expert_bytes >> input_width
                    >> gate_type >> up_type >> down_type >> cost.a_us >> cost.b_us >> h2d_gbs >> times_us[0] >> times_us[1] >> times_us[2]) {
            if (id != device_id || description != props.description || threads != sched->cpu_threads || top_k != count ||
                    pinned != (int) st->cache.pinned || expert_bytes != bytes || input_width != width ||
                    gate_type != st->cache.src[0]->type || up_type != st->cache.src[1]->type || down_type != st->cache.src[2]->type) continue;
            ggml_moe_cpu_cost fitted;
            if (threads != ggml_moe_calibration_threads || !cost.valid() || !(h2d_gbs > 0 && std::isfinite(h2d_gbs)) ||
                    !ggml_moe_cpu_cost_fit(times_us, fitted) || std::abs(cost.a_us - fitted.a_us) > 1e-6 * std::max(1.0, fitted.a_us) ||
                    std::abs(cost.b_us - fitted.b_us) > 1e-6 * std::max(1.0, fitted.b_us)) break;
            cpu.cost = cost;
            cpu.h2d_gbs = h2d_gbs;
            profiles.push_back({device, path, sched->cpu_threads, count, st->cache.pinned, bytes, width,
                    {st->cache.src[0]->type, st->cache.src[1]->type, st->cache.src[2]->type}, cost, h2d_gbs});
            GGML_LOG_INFO("MoE latency: loaded CPU=%.3f*x+%.3f us (T(0)=0) H2D=%.3f GB/s from %s\n",
                    cost.a_us, cost.b_us, h2d_gbs, path);
            return true;
        }
        GGML_LOG_ERROR("MoE latency: no valid matching device/thread/weight profile in %s; run llama-moe-bandwidth again\n", path);
        return false;
    }
#if !defined(__SSE2__)
    GGML_LOG_ERROR("MoE latency: cold-cache calibration requires x86 SSE2 cache flushing\n");
    return false;
#else
    if (sched->cpu_threads != ggml_moe_calibration_threads || st->n_expert <= ggml_moe_calibration_counts.back()) {
        GGML_LOG_ERROR("MoE latency: calibration requires 8 threads and more than 10 experts (disjoint CPU/H2D weights)\n");
        return false;
    }
    auto * ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    if (!ctx) return false;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, st->cache.cpu_expert_max_tokens);
    const int reserve = std::max(ggml_moe_calibration_counts.back(), std::min(GGML_MOE_GATE_MAX_IDS, st->cache.cpu_expert_max_tokens * st->n_expert_used));
    cpu.cold.assign(reserve, 0);
    if (!ggml_moe_runtime_cpu_buffers(st, backend, input)) { ggml_free(ctx); return false; }
    cpu.reserved_count = reserve;
    std::fill(cpu.source, cpu.source + width * st->cache.cpu_expert_max_tokens, 0.1f);
    ggml_backend_event_record(cpu.input_ready, backend);
    ggml_backend_synchronize(backend);
    if (!sched->cpu_worker) {
        sched->cpu_worker = new ggml_moe_cpu_worker;
        sched->cpu_worker->thread = std::thread(ggml_moe_runtime_cpu_worker_run, sched);
    }
    const auto pin = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    auto staging = ggml_backend_buft_alloc_buffer(pin, bytes);
    if (!staging) { ggml_free(ctx); return false; }
    char * stage = (char *) ggml_backend_buffer_get_base(staging);
    int copy_expert = ggml_moe_calibration_counts.back();
    auto copy = [&] {
        size_t offset = 0;
        for (int m = 0; m < 3; ++m) {
            const char * source = ggml_moe_runtime_host_ptr(st, m, copy_expert);
            if (!st->cache.pinned) {
                memcpy(stage + offset, source, st->expert_nb[m]);
                source = stage + offset;
            }
            backend->iface.set_tensor_async_stream(backend, sched->transfer_stream, st->cache.slots[m], source, 0, st->expert_nb[m]);
            offset += st->expert_nb[m];
        }
        backend->iface.synchronize_stream(backend, sched->transfer_stream);
        if (++copy_expert == st->n_expert) copy_expert = ggml_moe_calibration_counts.back();
    };
    auto compute = [&](int n, double & elapsed_us) {
        for (int e : cpu.ids) {
            for (int m = 0; m < 3; ++m) {
                const char * source = ggml_moe_runtime_host_ptr(st, m, e);
                for (size_t offset = 0; offset < st->expert_nb[m]; offset += 64) _mm_clflush(source + offset);
                _mm_clflush(source + st->expert_nb[m] - 1);
            }
        }
        _mm_mfence();
        int64_t ready_us = 0;
        const int64_t begin = ggml_time_us();
        ggml_moe_runtime_cpu_submit(sched, st, n, &ready_us);
        auto & worker = *sched->cpu_worker;
        std::unique_lock<std::mutex> lock(worker.mutex);
        worker.wake.wait(lock, [&] { return worker.done.load(std::memory_order_acquire); });
        worker.job = nullptr;
        worker.measured_ready_us = nullptr;
        elapsed_us = ready_us - begin;
        return worker.status == GGML_STATUS_SUCCESS && elapsed_us > 0;
    };
    std::array<double, 3> times_us{};
    int64_t total_copies = 0, total_copy_us = 0;
    bool success = true;
    for (size_t point = 0; point < times_us.size() && success; ++point) {
        const int n = ggml_moe_calibration_counts[point];
        cpu.cold.resize(n);
        cpu.ids.resize(n);
        for (int i = 0; i < n; ++i) cpu.cold[i] = cpu.ids[i] = i;
        if (!ggml_moe_runtime_cpu_buffers(st, backend, input)) { success = false; break; }
        double elapsed_us = 0;
        for (int i = 0; i < 8 && success; ++i) success = compute(n, elapsed_us);
        if (!success) break;
        std::atomic<bool> ready{false}, stop{false};
        int64_t copies = 0, copy_us = 0;
        std::thread transfer([&] {
            copy();
            ready.store(true, std::memory_order_release);
            const int64_t begin = ggml_time_us();
            do { copy(); ++copies; } while (!stop.load(std::memory_order_acquire));
            copy_us = ggml_time_us() - begin;
        });
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        std::array<double, 3> medians{};
        for (size_t repeat = 0; repeat < medians.size() && success; ++repeat) {
            std::vector<double> samples;
            samples.reserve(300);
            for (int i = 0; i < 300; ++i) {
                if (!compute(n, elapsed_us)) { success = false; break; }
                samples.push_back(elapsed_us);
            }
            if (success) {
                std::sort(samples.begin(), samples.end());
                medians[repeat] = (samples[149] + samples[150]) * 0.5;
            }
        }
        stop.store(true, std::memory_order_release);
        transfer.join();
        if (!success || copy_us <= 0) { success = false; break; }
        std::sort(medians.begin(), medians.end());
        times_us[point] = medians[1];
        total_copies += copies;
        total_copy_us += copy_us;
        GGML_LOG_INFO("MoE latency: threads=8 tokens=1 experts=%d submit_to_ready_us=%.3f H2D=%.3f GB/s (cold cache, 3 x 300 samples)\n",
                n, times_us[point], (double) copies * bytes / copy_us / 1000.0);
    }
    ggml_backend_buffer_free(staging);
    ggml_free(ctx);
    cpu.cold.clear();
    cpu.cpu_us = 0;
    if (!success || total_copy_us <= 0 || !ggml_moe_cpu_cost_fit(times_us, cpu.cost)) {
        GGML_LOG_ERROR("MoE latency: calibration failed or did not produce a positive CPU slope\n");
        return false;
    }
    cpu.h2d_gbs = (double) total_copies * bytes / total_copy_us / 1000.0;
    if (!(cpu.h2d_gbs > 0 && std::isfinite(cpu.h2d_gbs))) return false;
    std::ofstream file(path, std::ios::app);
    file << std::setprecision(17) << std::quoted(device_id) << ' ' << std::quoted(props.description) << ' '
         << sched->cpu_threads << ' ' << count << ' ' << (int) st->cache.pinned << ' ' << bytes << ' ' << width << ' '
         << (int) st->cache.src[0]->type << ' ' << (int) st->cache.src[1]->type << ' ' << (int) st->cache.src[2]->type << ' '
         << cpu.cost.a_us << ' ' << cpu.cost.b_us << ' ' << cpu.h2d_gbs;
    for (double value : times_us) file << ' ' << value;
    file << '\n';
    file.close();
    if (!file) { GGML_LOG_ERROR("MoE latency: failed to write %s\n", path); return false; }
    profiles.push_back({device, path, sched->cpu_threads, count, st->cache.pinned, bytes, width, {st->cache.src[0]->type, st->cache.src[1]->type, st->cache.src[2]->type}, cpu.cost, cpu.h2d_gbs});
    double max_error_us = 0;
    for (size_t i = 0; i < times_us.size(); ++i) max_error_us = std::max(max_error_us, std::abs(cpu.cost.time_us(ggml_moe_calibration_counts[i]) - times_us[i]));
    GGML_LOG_INFO("MoE latency: CPU=%.3f*x+%.3f us (T(0)=0) max_fit_error=%.3f us H2D=%.3f GB/s\n",
            cpu.cost.a_us, cpu.cost.b_us, max_error_us, cpu.h2d_gbs);
    return true;
#endif
}

enum ggml_status ggml_moe_runtime_compute_cpu_node(ggml_moe_runtime * sched, ggml_backend_t backend, ggml_tensor * node) {
    GGML_ASSERT(node->op == GGML_OP_MOE_CPU);
    const int layer = ggml_get_op_params_i32(node, 0);
    GGML_ASSERT(layer >= 0 && layer < sched->n_moe_caches);
    auto * st = sched->moe_caches[layer];
    auto & cpu = *st->cpu;
    const int phase = ggml_get_op_params_i32(node, 1);
    if (phase == 2) {
        if (!cpu.reserved_count) {
            const int reserve = std::min(GGML_MOE_GATE_MAX_IDS, st->cache.cpu_expert_max_tokens * st->n_expert_used);
            cpu.cold.assign(reserve, 0);
            if (!ggml_moe_runtime_cpu_buffers(st, backend, node->src[0])) return GGML_STATUS_ALLOC_FAILED;
            cpu.reserved_count = reserve;
            cpu.cold.clear();
        }
        cpu.staged_input = node->src[0];
        ggml_backend_tensor_get_async(backend, node->src[0], cpu.source, 0, ggml_nbytes(node->src[0]));
        ggml_backend_event_record(cpu.input_ready, backend);
        if (!sched->cpu_worker) {
            sched->cpu_worker = new ggml_moe_cpu_worker;
            sched->cpu_worker->thread = std::thread(ggml_moe_runtime_cpu_worker_run, sched);
        }
        cpu.staged_seq.store(sched->moe_seq, std::memory_order_release);
    } else if (phase == 0) {
        const int64_t begin = ggml_time_us();
        while (cpu.ready_seq.load(std::memory_order_acquire) != sched->moe_seq) {
            GGML_ASSERT(ggml_time_us() - begin < 5000000);
            std::this_thread::yield();
        }
        if (cpu.cold.empty()) return GGML_STATUS_SUCCESS;
        if (!ggml_moe_runtime_cpu_buffers(st, backend, node->src[0])) return GGML_STATUS_ALLOC_FAILED;
        ggml_backend_tensor_get_async(backend, node->src[0], cpu.source, 0, ggml_nbytes(node->src[0]));
        ggml_backend_event_record(cpu.input_ready, backend);
        if (!sched->cpu_worker) {
            sched->cpu_worker = new ggml_moe_cpu_worker;
            sched->cpu_worker->thread = std::thread(ggml_moe_runtime_cpu_worker_run, sched);
        }
        ggml_moe_runtime_cpu_submit(sched, st);
    } else {
        const int64_t begin = ggml_time_us();
        while (cpu.ready_seq.load(std::memory_order_acquire) != sched->moe_seq) {
            GGML_ASSERT(ggml_time_us() - begin < 5000000);
            std::this_thread::yield();
        }
        if (cpu.cold.empty()) return GGML_STATUS_SUCCESS;
        auto & worker = *sched->cpu_worker;
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto merge_begin = (ggml_moe_cpu_merge_begin_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cpu_merge_begin");
        const bool queued = merge_begin && merge_begin(backend, node, (const float *) cpu.output->data, cpu.rows, cpu.cold.size(), cpu.complete);
        // Cover a short decode job before falling back to the blocking wait.
        const int64_t spin_end = ggml_time_us() + 500;
        while (!worker.done.load(std::memory_order_acquire) && ggml_time_us() < spin_end) {
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lock(worker.mutex);
        GGML_ASSERT(worker.job == &cpu);
        worker.wake.wait(lock, [&] { return worker.done.load(std::memory_order_acquire); });
        cpu.join_us += ggml_time_us() - begin;
        worker.job = nullptr;
        if (worker.status != GGML_STATUS_SUCCESS) {
            if (queued) {
                ggml_backend_event_record(cpu.output_done, backend);
                cpu.output_pending = true;
            }
            return worker.status;
        }
        if (!queued) {
            auto merge = (ggml_moe_cpu_merge_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cpu_merge");
            GGML_ASSERT(merge);
            merge(backend, node, (const float *) cpu.output->data, cpu.rows, cpu.cold.size());
        }
        ggml_backend_event_record(cpu.output_done, backend);
        cpu.output_pending = true;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_moe_runtime_prefetch_next(
        ggml_moe_runtime * sched, ggml_moe_slot_state * st, const int32_t * ids, int n_ids, int n_tokens) {
    if (n_ids <= 0 || st->cache.pred_target < 0) {
        return;
    }
    for (int c = 0; c < sched->n_moe_caches; ++c) {
        auto * target = sched->moe_caches[c];
        if (target->cache.layer == st->cache.pred_target) {
            ggml_moe_runtime_prefetch(sched, target, ids, n_ids, n_tokens);
            return;
        }
    }
}

bool ggml_moe_runtime_prepare_batch(
        ggml_moe_runtime * sched, ggml_moe_slot_state * st,
        const int32_t * ids, int n_ids, const int32_t * predicted, int n_predicted) {
    GGML_ASSERT(st != nullptr && n_ids > 0 && n_ids % st->n_expert_used == 0);
    GGML_ASSERT(n_predicted >= 0 && n_predicted <= GGML_MOE_GATE_MAX_IDS);
    const int n_unique = ggml_moe_count_unique(ids, n_ids, st->n_expert);
    std::lock_guard<std::mutex> lock(sched->moe_mutex);
    ggml_backend_t bk = sched->backends[st->dev_backend_id];
    ggml_moe_runtime_sync(sched, bk);
    ggml_moe_runtime_publish_loc(st);
    if (ggml_moe_use_bulk_copy(n_unique, st->n_slots)) {
        st->n_dynamic_bulk++;
        st->n_pred_batch = 0;
        return true;
    }
    ggml_moe_runtime_fill(sched, st, ids, n_ids, ids);
    ggml_moe_runtime_sync(sched, bk);
    st->epoch++;
    st->n_dynamic_slot++;
    ggml_moe_runtime_prefetch_next(sched, st, predicted, n_predicted, n_ids / st->n_expert_used);
    return false;
}

static void ggml_moe_runtime_service_head(
        ggml_moe_runtime * sched, struct ggml_moe_slot_state * target,
        const struct ggml_moe_gate_entry * box) {
    GGML_ASSERT(box->n_ids == 0);
    const int n_pred = (int) box->n_pred;
    GGML_ASSERT(n_pred <= GGML_MOE_GATE_MAX_IDS);
    if (n_pred > 0) {
        ggml_moe_runtime_prefetch(sched, target, box->ids, n_pred, n_pred / std::max(1, target->cache.head_n_pred));
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
            std::lock_guard<std::mutex> lock(sched->moe_mutex);
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

            std::lock_guard<std::mutex> lock(sched->moe_mutex);
            if (st->cache.head_gate_slot >= 0 && st->head_last_seen < seq) {
                const auto * head = &sched->moe_ch.ring[st->cache.head_gate_slot];
                if (((volatile const uint32_t *) &head->seq)[0] == seq) {
                    std::atomic_thread_fence(std::memory_order_acquire);
                    struct ggml_moe_gate_entry head_box;
                    memcpy(&head_box, head, sizeof(head_box));
                    ggml_moe_runtime_service_head(sched, st, &head_box);
                }
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
    while (sched->moe_worker != NULL && !sched->moe_worker->idle.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::lock_guard<std::mutex> lock(sched->moe_mutex);
    if (sched->n_moe_caches > 0) {
        ggml_moe_runtime_sync(sched, sched->backends[sched->moe_caches[0]->dev_backend_id]);
    }
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
    const int32_t e = st->expert_of_slot[0] >= 0 ? st->expert_of_slot[0] : 0;

    for (int m = 0; m < 3; m++) {
        if (st->cache.src[m] == NULL) {
            continue;
        }
        const size_t nb  = st->expert_nb[m];
        if (st->cache.src_fd >= 0) {
            const bool ok = bk->iface.set_tensor_async_file(bk, sched->transfer_stream,
                    st->cache.slots[m], st->cache.src_fd, st->cache.src_off[m] + (uint64_t) e * nb, 0, nb);
            GGML_ASSERT(ok);
            continue;
        }
        const char * src = ggml_moe_runtime_host_ptr(st, m, e);
        if (!st->cache.pinned) {
            src = ggml_moe_runtime_stage(sched, bk, src, nb);
        }
        bk->iface.set_tensor_async_stream(bk, sched->transfer_stream, st->cache.slots[m], src, 0, nb);
    }
    ggml_moe_runtime_publish_row(sched, st, bk, e);

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
    if (cache->n_initial < 0 || cache->n_initial > n_slots || (cache->n_initial && !cache->initial_experts)) {
        return false;
    }
    std::vector<bool> initial_seen(n_expert, false);
    for (int i = 0; i < cache->n_initial; ++i) {
        const int32_t e = cache->initial_experts[i];
        if (e < 0 || e >= n_expert || initial_seen[e]) {
            return false;
        }
        initial_seen[e] = true;
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
    const bool hot_init = cache->n_initial > 0;
    const char * policy = getenv("GGML_MOE_CACHE_POLICY");
    if (!policy) {
        policy = hot_init ? "decay" : "lfu";
    }
    GGML_ASSERT(strcmp(policy, "lfu") == 0 || strcmp(policy, "lru") == 0 || strcmp(policy, "decay") == 0);
    st->cache_policy = strcmp(policy, "lru") == 0 ? 1 : strcmp(policy, "decay") == 0 ? 2 : 0;
    const char * decay = getenv("GGML_MOE_CACHE_DECAY");
    st->decay_interval = decay ? atoi(decay) : hot_init ? 8 : 32;
    GGML_ASSERT(st->decay_interval > 0);
    const char * protect = getenv("GGML_MOE_PREFETCH_PROTECT");
    st->protect_predictions = protect ? atoi(protect) != 0 : hot_init;
    const char * budget = getenv("GGML_MOE_PREFETCH_BUDGET");
    st->prefetch_budget = budget ? atoi(budget) : -1;
    GGML_ASSERT(st->prefetch_budget >= -1);
    const char * rank = getenv("GGML_MOE_PREFETCH_RANK");
    GGML_ASSERT(rank == nullptr || strcmp(rank, "original") == 0 || strcmp(rank, "votes") == 0);
    st->rank_votes = rank && strcmp(rank, "votes") == 0;
    GGML_LOG_INFO("MoE cache policy layer=%d policy=%s decay=%d protect=%d budget=%d votes=%d slots=%d\n",
            cache->layer, policy, st->decay_interval, st->protect_predictions,
            st->prefetch_budget, st->rank_votes, n_slots);
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

    const bool hybrid_weights = cache->src[0] && cache->src[1] && cache->src[2] &&
        ggml_is_quantized(cache->src[0]->type) && ggml_is_quantized(cache->src[1]->type) && ggml_is_quantized(cache->src[2]->type);
    if (sched->cpu_max_tokens > 0 && hybrid_weights && cache->gate_seq && cache->src_fd < 0 &&
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cpu_merge")) {
        st->cpu = new ggml_moe_cpu_state;
        st->cpu->last_used.resize(n_expert, 0);
        auto max_tokens = (ggml_moe_cpu_max_tokens_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cpu_max_tokens");
        GGML_ASSERT(max_tokens);
        st->cache.cpu_expert_max_tokens = sched->cpu_max_tokens;
        for (int m = 0; m < 3; ++m) st->cache.cpu_expert_max_tokens = std::min(st->cache.cpu_expert_max_tokens, max_tokens(bk, cache->src[m]->type));
        st->cache.cpu_expert_early = sched->cpu_balance && !sched->moe_host_gate && st->cache.cpu_expert_max_tokens > 0;
        if (st->cache.cpu_expert_early) GGML_ASSERT(ggml_moe_runtime_bandwidth(sched, st, bk));
    } else {
        st->cache.cpu_expert_max_tokens = 0;
    }
    // Calibration can write slot zero. Load the profile after it finishes.
    if (cache->n_initial > 0) {
        std::string ids;
        for (int v = 0; v < cache->n_initial; ++v) {
            const int32_t e = cache->initial_experts[v];
            ggml_moe_runtime_copy_expert(sched, st, bk, e, v, false);
            st->fill_seq[v] = 0;
            ids += (v ? "," : "") + std::to_string(e);
        }
        ggml_moe_runtime_sync(sched, bk);
        st->free_cnt = 0;
        for (int v = cache->n_initial; v < n_slots; ++v) {
            st->free_slot[st->free_cnt++] = v;
        }
        GGML_LOG_INFO("MoE preload: layer=%d slots=%d bytes=%llu experts=[%s]\n", cache->layer, cache->n_initial, (unsigned long long) st->n_bytes_sync, ids.c_str());
        st->n_bytes_sync = 0;
    }
    if (cache->initial_freq) {
        uint64_t total = 0;
        uint32_t maximum = 0;
        for (int e = 0; e < n_expert; ++e) {
            st->freq[e] = cache->initial_freq[e];
            total += st->freq[e];
            maximum = std::max(maximum, st->freq[e]);
        }
        GGML_LOG_INFO("MoE prior: layer=%d experts=%d total=%llu max=%u\n", cache->layer, n_expert, (unsigned long long) total, maximum);
    }
    st->cache.initial_experts = nullptr;
    st->cache.initial_freq = nullptr;
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
    GGML_LOG_DEBUG("MoE slot cache, single-token decode and small batches: %d layers, one load = one expert (up+gate+down)\n",
            sched->n_moe_caches);
    GGML_LOG_DEBUG("  hit: resident token-expert assignments before fill; load_sync/load_pf: unique expert copies per batch; MiB: totals\n");
    GGML_LOG_DEBUG("  pf_prec: unique predicted experts used / unique predicted experts in each batch; tokens include verification inputs and warmup\n");
    GGML_LOG_DEBUG("  %-5s %6s %8s %8s %8s %11s %11s %8s %11s %11s\n",
            "layer", "slots", "batches", "tokens", "hit", "load_sync", "load_pf", "pf_prec", "MiB_sync", "MiB_pf");

    uint64_t t_hit = 0, t_use = 0, t_ls = 0, t_lp = 0, t_bs = 0, t_bp = 0, t_pp = 0, t_pu = 0, max_tokens = 0, max_batches = 0;

    for (int i = 0; i < sched->n_moe_caches; i++) {
        const struct ggml_moe_slot_state * st = sched->moe_caches[i];
        if (st->cpu) {
            const auto & c = *st->cpu;
            GGML_LOG_INFO("MoE hybrid layer=%d batches=%llu cpu_assign=%llu gpu_assign=%llu cold_experts=%llu cold_weight_MiB=%.3f cpu_ms=%.3f cpu_join_ms=%.3f cpu_a_us=%.3f cpu_b_us=%.3f h2d_gbs=%.3f\n",
                    st->cache.layer, (unsigned long long) c.step, (unsigned long long) c.cpu_assignments,
                    (unsigned long long) c.gpu_assignments, (unsigned long long) c.cold_experts,
                    c.avoided_bytes / 1048576.0, c.cpu_us / 1000.0, c.join_us / 1000.0, c.cost.a_us, c.cost.b_us, c.h2d_gbs);
        }
        const double batch = (double) std::max<uint64_t>(1, st->n_batches);
        const uint64_t use = st->n_hit + st->n_miss;
        char precision[32] = "N/A";
        if (st->n_pf_pred > 0) {
            snprintf(precision, sizeof(precision), "%.4f", (double) st->n_pf_useful / (double) st->n_pf_pred);
        }

        GGML_LOG_DEBUG("  %-5d %6d %8llu %8llu %8.4f %11.3f %11.3f %8s %11.1f %11.1f\n",
                st->cache.layer, st->n_slots, (unsigned long long) st->n_batches, (unsigned long long) st->n_tokens,
                use ? (double) st->n_hit / (double) use : 0.0,
                (double) st->n_load_sync / batch, (double) st->n_load_pf / batch, precision,
                (double) st->n_bytes_sync / 1024.0 / 1024.0,
                (double) st->n_bytes_pf   / 1024.0 / 1024.0);

        t_hit += st->n_hit;        t_use += use;
        t_ls  += st->n_load_sync;  t_lp  += st->n_load_pf;
        t_bs  += st->n_bytes_sync; t_bp  += st->n_bytes_pf;
        t_pp  += st->n_pf_pred;    t_pu  += st->n_pf_useful;
        max_tokens = std::max(max_tokens, st->n_tokens);
        max_batches = std::max(max_batches, st->n_batches);
    }

    if (max_tokens > 0) {
        const double batch = (double) max_batches;
        char precision[32] = "N/A";
        if (t_pp > 0) {
            snprintf(precision, sizeof(precision), "%.4f", (double) t_pu / (double) t_pp);
        }
        GGML_LOG_DEBUG("  %-5s %6s %8llu %8llu %8.4f %11.3f %11.3f %8s %11.1f %11.1f\n",
                "TOTAL", "", (unsigned long long) max_batches, (unsigned long long) max_tokens,
                t_use ? (double) t_hit / (double) t_use : 0.0,
                (double) t_ls / batch, (double) t_lp / batch, precision,
                (double) t_bs / 1024.0 / 1024.0, (double) t_bp / 1024.0 / 1024.0);
    }

}
void ggml_moe_runtime_free(ggml_moe_runtime * sched) {
    for (int i = 0; i < sched->n_backends; i++) { ggml_backend_synchronize(sched->backends[i]); }
    if (sched->moe_worker != NULL) {
        sched->moe_worker->stop.store(true, std::memory_order_relaxed);
        sched->moe_worker->thread.join();
        delete sched->moe_worker;
        sched->moe_worker = NULL;
    }
    if (sched->cpu_worker) {
        {
            std::lock_guard<std::mutex> lock(sched->cpu_worker->mutex);
            sched->cpu_worker->stop = true;
            sched->cpu_worker->wake.notify_all();
        }
        sched->cpu_worker->thread.join();
        delete sched->cpu_worker;
    }
    if (sched->n_moe_caches > 0) {
        ggml_moe_runtime_sync(sched, sched->backends[sched->moe_caches[0]->dev_backend_id]);
        ggml_moe_runtime_stats_report(sched);
    }
    for (int i = 0; i < sched->n_moe_caches; i++) {
        struct ggml_moe_slot_state * st = sched->moe_caches[i];
        if (st->cpu) {
            ggml_backend_event_free(st->cpu->input_ready);
            ggml_backend_event_free(st->cpu->output_done);
            ggml_backend_buffer_free(st->cpu->staging);
            ggml_gallocr_free(st->cpu->alloc);
            ggml_free(st->cpu->ctx);
        }
        delete st->cpu;
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
    ggml_backend_free(sched->cpu_backend);
    if (sched->cpu_threadpool) sched->cpu_threadpool_free(sched->cpu_threadpool);
    delete sched->moe_substitute_enabled;
    delete sched;
}
