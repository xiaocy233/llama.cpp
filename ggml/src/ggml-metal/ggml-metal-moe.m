#if defined(__APPLE__)

#import "ggml-metal-moe.h"
#import "ggml-metal-impl.h"

#include "ggml.h"
#include "ggml-impl.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <errno.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

// the gate gives up after this much CPU time when the worker has not released it
#ifndef GGML_METAL_MOE_GATE_TIMEOUT_MS
#define GGML_METAL_MOE_GATE_TIMEOUT_MS 10000
#endif

enum { MOE_TASK_MEMCPY = 0, MOE_TASK_PREAD = 1 };

typedef struct {
    int      kind;
    char *   dst;
    const char * src;   // MEMCPY
    int      fd;        // PREAD
    uint64_t off;       // PREAD
    size_t   size;
    uint64_t dep;       // wait until the completion watermark reaches this before running (0 = none)
    // host-side wait on a GPU event before running (ring-buffer reuse protection)
    ggml_metal_event_t wait_ev;
    uint64_t           wait_val;
    uint64_t seq;
} moe_task;

#define MOE_QUEUE_CAP 2048  // power of two

struct ggml_metal_moe {
    ggml_metal_t    ctx;
    ggml_metal_device_t dev;

    // ---- gate mailbox (shared by the host and device) ----
    struct ggml_moe_gate_channel ch;
    bool ch_init;

    // ---- fill pool ----
    pthread_t       threads[16];
    int             n_threads;
    pthread_mutex_t mu;
    pthread_cond_t  cv_submit;   // queue non-empty
    pthread_cond_t  cv_done;     // completion watermark advanced
    bool            stop;

    // ---- decode gate events (event form): one per gate slot, lazily created ----
    ggml_metal_event_t gate_events[GGML_MOE_GATE_MAX_LAYERS];
    bool gate_events_init;
    void * gate_events_pub[GGML_MOE_GATE_MAX_LAYERS];   // same events, void* view for the channel

    moe_task        queue[MOE_QUEUE_CAP];
    int             q_head;      // next pop
    int             q_tail;      // next push

    _Atomic uint64_t issued;
    _Atomic uint64_t completed;  // in-order watermark: tasks [.., completed] are done

    // per-seq done markers for in-order completion (seqs are dense)
    _Atomic uint64_t done_ring[MOE_QUEUE_CAP];

    // armed events: signal (ev, val) from the CPU once the watermark reaches wm
    struct armed { uint64_t wm; ggml_metal_event_t ev; uint64_t val; } armed[64];
    int n_armed;

    // current ring-reuse wait: captured into tasks submitted while set
    ggml_metal_event_t cur_wait_ev;
    uint64_t           cur_wait_val;

};

static int moe_pool_threads(void) {
    // decode bursts up to 8 misses + 8 predictions in flight at once; 4 threads serialize those
    // into 4 rounds of pread latency and the wait lands in the token time
    const char * val = getenv("GGML_METAL_MOE_POOL_THREADS");
    const int n = val ? atoi(val) : 16;
    return n > 0 && n <= 16 ? n : 16;
}

// ---------- completion watermark ----------

static void moe_mark_done(ggml_metal_moe_t moe, uint64_t seq) {
    atomic_store_explicit(&moe->done_ring[seq % MOE_QUEUE_CAP], seq, memory_order_release);

    pthread_mutex_lock(&moe->mu);

    // advance the in-order watermark as far as the done markers allow
    while (moe->completed < atomic_load_explicit(&moe->issued, memory_order_relaxed) &&
           atomic_load_explicit(&moe->done_ring[(moe->completed + 1) % MOE_QUEUE_CAP], memory_order_acquire) == moe->completed + 1) {
        moe->completed++;
    }

    // fire armed events whose watermark the pool has now passed
    int w = 0;
    for (int i = 0; i < moe->n_armed; i++) {
        if (moe->armed[i].wm <= moe->completed) {
            ggml_metal_event_host_signal(moe->armed[i].ev, moe->armed[i].val);
        } else {
            moe->armed[w++] = moe->armed[i];
        }
    }
    moe->n_armed = w;

    pthread_cond_broadcast(&moe->cv_done);
    pthread_mutex_unlock(&moe->mu);
}

static void moe_wait_watermark(ggml_metal_moe_t moe, uint64_t wm) {
    pthread_mutex_lock(&moe->mu);
    while (atomic_load_explicit(&moe->completed, memory_order_acquire) < wm) {
        pthread_cond_wait(&moe->cv_done, &moe->mu);
    }
    pthread_mutex_unlock(&moe->mu);
}

// ---------- pool ----------

static void * moe_pool_thread(void * arg) {
    ggml_metal_moe_t moe = (ggml_metal_moe_t) arg;

    while (true) {
        pthread_mutex_lock(&moe->mu);
        while (!moe->stop && moe->q_head == moe->q_tail) {
            pthread_cond_wait(&moe->cv_submit, &moe->mu);
        }
        if (moe->stop && moe->q_head == moe->q_tail) {
            pthread_mutex_unlock(&moe->mu);
            return NULL;
        }
        moe_task t = moe->queue[moe->q_head];
        moe->q_head = (moe->q_head + 1) % MOE_QUEUE_CAP;
        pthread_mutex_unlock(&moe->mu);

        if (t.wait_ev != NULL) {
            ggml_metal_event_host_wait(t.wait_ev, t.wait_val);
        }
        if (t.dep != 0) {
            moe_wait_watermark(moe, t.dep);
        }

        if (t.kind == MOE_TASK_MEMCPY) {
            memcpy(t.dst, t.src, t.size);
        } else {
            // pread the full slice; EINTR/short reads must not leave holes
            size_t done = 0;
            while (done < t.size) {
                const ssize_t n = pread(t.fd, t.dst + done, t.size - done, (off_t) (t.off + done));
                if (n <= 0) {
                    if (n < 0 && errno == EINTR) {
                        continue;
                    }
                    GGML_LOG_ERROR("%s: pread failed (fd=%d off=%llu size=%zu): n=%zd errno=%d\n",
                            __func__, t.fd, (unsigned long long) t.off, t.size, n, errno);
                    break;
                }
                done += (size_t) n;
            }
        }

        moe_mark_done(moe, t.seq);
    }
}

static void moe_submit(ggml_metal_moe_t moe, moe_task t) {
    pthread_mutex_lock(&moe->mu);
    // backpressure: a full queue means the pool is behind; wait for room (tasks drain independently)
    while ((moe->q_tail + 1) % MOE_QUEUE_CAP == moe->q_head) {
        pthread_cond_wait(&moe->cv_done, &moe->mu);
    }
    t.seq = atomic_fetch_add_explicit(&moe->issued, 1, memory_order_relaxed) + 1;
    // capture the pending ring-reuse wait, if any
    if (moe->cur_wait_ev != NULL) {
        t.wait_ev  = moe->cur_wait_ev;
        t.wait_val = moe->cur_wait_val;
    }
    moe->queue[moe->q_tail] = t;
    moe->q_tail = (moe->q_tail + 1) % MOE_QUEUE_CAP;
    pthread_cond_signal(&moe->cv_submit);
    pthread_mutex_unlock(&moe->mu);
}

// ---------- interface ----------

ggml_metal_moe_t ggml_metal_moe_get(ggml_metal_t ctx) {
    ggml_metal_moe_t moe = ggml_metal_moe_get_raw(ctx);
    if (moe == NULL) {
        moe = calloc(1, sizeof(struct ggml_metal_moe));
        moe->ctx = ctx;
        moe->dev = ggml_metal_get_dev(ctx);
        pthread_mutex_init(&moe->mu, NULL);
        pthread_cond_init(&moe->cv_submit, NULL);
        pthread_cond_init(&moe->cv_done, NULL);

        moe->n_threads = moe_pool_threads();
        for (int i = 0; i < moe->n_threads; i++) {
            pthread_create(&moe->threads[i], NULL, moe_pool_thread, moe);
        }
        ggml_metal_set_moe(ctx, moe);
        // the op encoders reach the gate events through the device (they have no ctx pointer)
        ggml_metal_device_set_moe_state(moe->dev, moe);
    }
    return moe;
}

void ggml_metal_moe_free(ggml_metal_moe_t moe) {
    if (!moe) {
        return;
    }
    pthread_mutex_lock(&moe->mu);
    moe->stop = true;
    pthread_cond_broadcast(&moe->cv_submit);
    pthread_mutex_unlock(&moe->mu);
    for (int i = 0; i < moe->n_threads; i++) {
        pthread_join(moe->threads[i], NULL);
    }

    pthread_mutex_destroy(&moe->mu);
    pthread_cond_destroy(&moe->cv_submit);
    pthread_cond_destroy(&moe->cv_done);
    // the mailbox MTLBuffer is intentionally not freed: llama_context destroys the backends before
    // the scheduler, and the scheduler's stats report (in ggml_backend_sched_free) still reads the
    // timeout counters through the saved channel pointers. ~54 KB once per process.
    free(moe);
}

bool ggml_metal_moe_gate_channel(ggml_metal_moe_t moe, struct ggml_moe_gate_channel * out) {
    if (!moe->ch_init) {
        // one shared buffer for ring + release + timeout: the host (worker, decode gate) reads and
        // writes it through contents, and the publish kernel (prefill gate) writes it from the GPU.
        // owned by the device, never released - the scheduler's stats report reads it after the
        // backend context is gone
        @autoreleasepool {
            const size_t sz_ring = GGML_MOE_GATE_MAX_LAYERS * sizeof(struct ggml_moe_gate_entry);
            const size_t sz_rel  = GGML_MOE_GATE_MAX_LAYERS * sizeof(uint32_t);
            const size_t sz_to   = GGML_MOE_GATE_MAX_LAYERS * sizeof(int32_t);

            id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(moe->dev);
            id<MTLBuffer> buf = [device newBufferWithLength:sz_ring + sz_rel + sz_to
                                                    options:MTLResourceStorageModeShared];
            if (buf == nil) {
                return false;
            }
            memset(buf.contents, 0, buf.length);

            moe->ch.ring    = (struct ggml_moe_gate_entry *) buf.contents;
            moe->ch.release = (volatile uint32_t *) ((char *) buf.contents + sz_ring);
            moe->ch.timeout = (volatile int32_t *)  ((char *) buf.contents + sz_ring + sz_rel);
            moe->ch.gate_events = NULL;   // filled below, once the channel is first handed out
            ggml_metal_device_set_moe_mailbox(moe->dev, buf);
            moe->ch_init = true;
        }
    }
    *out = moe->ch;
    return true;
}

ggml_metal_event_t ggml_metal_moe_gate_event(ggml_metal_moe_t moe, int slot) {
    if (slot < 0 || slot >= GGML_MOE_GATE_MAX_LAYERS) {
        return NULL;
    }
    if (!moe->gate_events_init) {
        memset(moe->gate_events, 0, sizeof(moe->gate_events));
        memset(moe->gate_events_pub, 0, sizeof(moe->gate_events_pub));
        moe->gate_events_init = true;
    }
    if (moe->gate_events[slot] == NULL) {
        moe->gate_events[slot] = ggml_metal_device_event_init(moe->dev);
        if (moe->gate_events[slot] != NULL) {
            moe->gate_events_pub[slot] = moe->gate_events[slot];
        }
    }
    return moe->gate_events[slot];
}

// worker-side signal for the event form (ggml-backend.cpp): publish the events array through the
// channel and expose the signal entry, so the shared worker needs no Metal-specific iface hook
void ** ggml_metal_moe_gate_events(ggml_metal_moe_t moe) {
    // force-create every slot the scheduler may use (all of them): the worker signals a slot
    // before any gate kernel ever encodes a wait on it
    for (int i = 0; i < GGML_MOE_GATE_MAX_LAYERS; i++) {
        ggml_metal_moe_gate_event(moe, i);
    }
    return moe->gate_events_pub;
}

void ggml_metal_moe_gate_signal(void * event, uint32_t seq) {
    ggml_metal_event_t ev = (ggml_metal_event_t) event;
    if (ev != NULL) {
        ggml_metal_event_host_signal(ev, seq);
    }
}

// signal (ev[slot] = seq) once the pool has finished every task issued so far. The wait itself
// is not optional - a prefetch issued for an earlier entry can still be in flight, and its row
// already says resident - but arming instead of blocking lets the worker go service the next
// layer's entry while the pool drains, which overlaps that layer's fills with the drain.
void ggml_metal_moe_gate_signal_drained(void * moe_p, int slot, uint32_t seq) {
    ggml_metal_moe_t moe = (ggml_metal_moe_t) moe_p;
    ggml_metal_event_t ev = ggml_metal_moe_gate_event(moe, slot);
    if (ev == NULL) {
        return;
    }
    pthread_mutex_lock(&moe->mu);
    const uint64_t wm = atomic_load_explicit(&moe->issued, memory_order_relaxed);
    if (atomic_load_explicit(&moe->completed, memory_order_acquire) >= wm) {
        pthread_mutex_unlock(&moe->mu);
        ggml_metal_event_host_signal(ev, seq);
    } else if (moe->n_armed < (int) (sizeof(moe->armed) / sizeof(moe->armed[0]))) {
        moe->armed[moe->n_armed++] = (struct armed) { .wm = wm, .ev = ev, .val = seq };
        pthread_mutex_unlock(&moe->mu);
    } else {
        pthread_mutex_unlock(&moe->mu);
        GGML_ABORT("MoE gate event queue is full");
    }
}

void ggml_metal_moe_fill_mem(ggml_metal_moe_t moe, void * dst, const void * src, size_t size) {
    moe_submit(moe, (moe_task) { .kind = MOE_TASK_MEMCPY, .dst = dst, .src = src, .size = size });
}

void ggml_metal_moe_fill_file(ggml_metal_moe_t moe, void * dst, int fd, uint64_t off, size_t size) {
    moe_submit(moe, (moe_task) { .kind = MOE_TASK_PREAD, .dst = dst, .fd = fd, .off = off, .size = size });
}

void ggml_metal_moe_row_mem(ggml_metal_moe_t moe, void * dst, const void * src, size_t size) {
    // a row publish must be ordered after every fill issued before it
    moe_submit(moe, (moe_task) { .kind = MOE_TASK_MEMCPY, .dst = dst, .src = src, .size = size,
            .dep = atomic_load_explicit(&moe->issued, memory_order_relaxed) });
}

void ggml_metal_moe_wait(ggml_metal_moe_t moe) {
    moe_wait_watermark(moe, atomic_load_explicit(&moe->issued, memory_order_relaxed));
}

void ggml_metal_moe_ring_begin(ggml_metal_moe_t moe, ggml_metal_event_t ev) {
    pthread_mutex_lock(&moe->mu);
    moe->cur_wait_ev  = ev;
    moe->cur_wait_val = ggml_metal_event_host_await_point(ev);
    pthread_mutex_unlock(&moe->mu);
}

void ggml_metal_moe_ring_end(ggml_metal_moe_t moe, ggml_metal_event_t ev_copy) {
    pthread_mutex_lock(&moe->mu);
    moe->cur_wait_ev = NULL;
    const uint64_t val = ggml_metal_event_host_arm(ev_copy);
    const uint64_t wm  = atomic_load_explicit(&moe->issued, memory_order_relaxed);
    if (atomic_load_explicit(&moe->completed, memory_order_acquire) >= wm) {
        // the pool is already past this point: signal now. arming would never fire without a
        // later task to advance the watermark, and the GPU-side wait would hang the queue
        ggml_metal_event_host_signal(ev_copy, val);
    } else if (moe->n_armed < (int) (sizeof(moe->armed) / sizeof(moe->armed[0]))) {
        moe->armed[moe->n_armed++] = (struct armed) { .wm = wm, .ev = ev_copy, .val = val };
    } else {
        GGML_LOG_ERROR("%s: armed event overflow, signaling late tasks may be dropped\n", __func__);
    }
    pthread_mutex_unlock(&moe->mu);
}

// ---------- host gate ----------

// spin with sched_yield until the event reaches val, bounded by CPU time (a wall-clock bound misfires
// when the process is descheduled; CPU time only advances while we run)
static bool ggml_metal_moe_spin_until(ggml_metal_event_t ev, uint32_t val) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>) ggml_metal_event_obj(ev);
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    const int64_t deadline = (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000 + GGML_METAL_MOE_GATE_TIMEOUT_MS * 1000;

    while (true) {
        if (event.signaledValue >= val) {
            return true;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
        if ((int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000 > deadline) {
            return false;
        }
        sched_yield();
    }
}

bool ggml_metal_moe_host_gate(ggml_metal_moe_t moe, const struct ggml_tensor * node) {
    const struct ggml_tensor * loc_map = node->src[0];
    const struct ggml_tensor * ids_in  = node->src[1];
    const struct ggml_tensor * seq_t   = node->src[2];

    const int layer   = ggml_get_op_params_i32(node, 0);
    const int n_ids   = ggml_get_op_params_i32(node, 1);
    const int n_pred  = ggml_get_op_params_i32(node, 2);
    const int n_slots = ggml_get_op_params_i32(node, 3);

    const int32_t * loc = (const int32_t *) loc_map->data;
    const int32_t * ids = (const int32_t *) ids_in->data;
    const uint32_t  seq = (uint32_t) *(const int32_t *) seq_t->data;
    int32_t * out = (int32_t *) node->data;

    GGML_ASSERT(n_ids > 0);
    if (!moe->ch_init) {
        struct ggml_moe_gate_channel ch;
        if (!ggml_metal_moe_gate_channel(moe, &ch)) {
            return false;
        }
    }

    struct ggml_moe_gate_entry * box = &moe->ch.ring[layer];
    ggml_metal_event_t event = ggml_metal_moe_gate_event(moe, layer);

    // Do not overwrite an entry before the worker releases it.
    {
        const uint32_t prev_seq = *(volatile uint32_t *) &box->seq;
        const bool prev_needs_service = prev_seq != 0;
        if (prev_needs_service) {
            if (!ggml_metal_moe_spin_until(event, prev_seq)) {
                *(volatile int32_t *) &moe->ch.timeout[layer] += 1;
                GGML_LOG_ERROR("%s: gate slot %d prev entry seq %u never serviced; pool issued=%llu completed=%llu q=[%d,%d)\n",
                        __func__, layer, prev_seq,
                        (unsigned long long) atomic_load_explicit(&moe->issued,   memory_order_relaxed),
                        (unsigned long long) atomic_load_explicit(&moe->completed, memory_order_relaxed),
                        moe->q_head, moe->q_tail);
                return false;
            }
        }
    }

    // publish: this layer's ids and the prediction tail, in one entry
    for (int i = 0; i < n_ids + n_pred; i++) {
        box->ids[i] = ids[i];
    }
    box->layer  = (uint32_t) layer;
    box->n_ids  = (uint32_t) n_ids;
    box->n_pred = (uint32_t) n_pred;
    box->mode   = GGML_MOE_GATE_MODE_NORMAL;

    uint32_t mask = 0;
    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && loc[e] >= n_slots) {
            mask |= 1u << i;
        }
    }
    box->miss_mask = mask;
    atomic_thread_fence(memory_order_seq_cst);
    *(volatile uint32_t *) &box->seq = seq;   // seq last: the worker's handshake

    if (!ggml_metal_moe_spin_until(event, seq)) {
        *(volatile int32_t *) &moe->ch.timeout[layer] += 1;
        GGML_LOG_ERROR("%s: gate slot %d seq %u timed out waiting for fills\n", __func__, layer, seq);
        return false;
    }
    atomic_thread_fence(memory_order_seq_cst);

    // resolve: every id is resident by now (the worker filled what was missing)
    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        const int32_t p = e >= 0 ? loc[e] : 0;
        if (e >= 0 && p >= n_slots) {
            // the worker reported the fill but the row still says miss: count it and fail loudly
            *(volatile int32_t *) &moe->ch.timeout[layer] += 1;
            GGML_LOG_ERROR("%s: gate slot %d expert %d still not resident after release (seq %u, n_ids %d)\n",
                    __func__, layer, e, seq, n_ids);
            return false;
        }
        out[i] = p;
    }
    return true;
}

#endif // __APPLE__
