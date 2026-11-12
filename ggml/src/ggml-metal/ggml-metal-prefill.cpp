#if defined(__APPLE__)

#include "ggml-metal-prefill.h"

#include "ggml-alloc.h"
#include "ggml-metal.h"
#include "ggml-impl.h"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>

ggml_metal_prefill_layer::~ggml_metal_prefill_layer() {
    for (auto & pool : pools) {
        ggml_free(pool.prefill_ctx);
    }
    ggml_backend_buffer_free(msg_buf);
    ggml_free(msg_ctx);
    if (event) {
        ggml_backend_metal_event_free(event);
    }
}

ggml_metal_prefill_offload::ggml_metal_prefill_offload(
        ggml_backend_t backend, int n_prefill_buffers)
    : backend(backend), cfg_n_buffers(std::max(1, n_prefill_buffers)) {}

ggml_metal_prefill_offload::~ggml_metal_prefill_offload() {
    stop_sidecar();
    release_prefill_buffers();
    for (auto & entry : group_buffers) {
        ggml_backend_buffer_free(entry.second);
    }
}

void ggml_metal_prefill_offload::add_pool(
        const ggml_tensor * src, int layer_idx, int fd, uint64_t file_offset) {
    if (src == nullptr || src->name[0] == '\0') {
        return;
    }

    std::lock_guard<std::mutex> lock(build_mtx);

    if ((int) layers.size() <= layer_idx) {
        layers.resize((size_t) layer_idx + 1);
    }

    auto & layer_ptr = layers[(size_t) layer_idx];
    if (!layer_ptr) {
        auto layer = std::make_unique<ggml_metal_prefill_layer>();
        layer->layer_idx = layer_idx;
        // pool_to_layer counts matrices (3 per layer), so count bound layers for the order
        int n_bound = 0;
        for (const auto & lp : layers) {
            if (lp != nullptr) {
                n_bound++;
            }
        }
        layer->order = n_bound;
        layer->expected_uses = 0;

        ggml_backend_buffer_type_t shared_buft = ggml_backend_metal_get_shared_buffer_type(backend);

        ggml_init_params msg_params{};
        msg_params.mem_size = ggml_tensor_overhead() + 64;
        msg_params.no_alloc = true;
        layer->msg_ctx = ggml_init(msg_params);
        layer->msg_tensor = ggml_new_tensor_1d(layer->msg_ctx, GGML_TYPE_I8, MOE_MSG_NBYTES);
        char msg_name[64];
        snprintf(msg_name, sizeof(msg_name), "moe_prefill_msg_L%d", layer_idx);
        ggml_set_name(layer->msg_tensor, msg_name);
        layer->msg_buf = ggml_backend_alloc_ctx_tensors_from_buft(layer->msg_ctx, shared_buft);
        GGML_ASSERT(layer->msg_buf);
        layer->mapped = (uint8_t *) layer->msg_tensor->data;
        memset(layer->mapped, 0, MOE_MSG_NBYTES);
        layer->event = ggml_backend_metal_event_new(backend);

        layer_ptr = std::move(layer);
    }

    ggml_metal_prefill_layer & layer = *layer_ptr;
    auto existing = layer.name_to_pool_idx.find(src->name);
    if (existing != layer.name_to_pool_idx.end()) {
        return;
    }

    GGML_ASSERT(!buffers_finalized_.load(std::memory_order_acquire));

    // group key: name suffix past the layer prefix + stride (odd-sized banks get their own ring)
    std::string key = src->name;
    const size_t second = key.find('.', key.find('.') + 1);
    if (second != std::string::npos) {
        key.erase(0, second + 1);
    }
    key += "_" + std::to_string(src->nb[2]);

    ggml_metal_prefill_pool pool;
    pool.fd = fd;
    pool.file_offset = file_offset;
    pool.stride = src->nb[2];
    pool.group_key = key;

    ggml_init_params prefill_params{};
    prefill_params.mem_size = ggml_tensor_overhead() * 2;
    prefill_params.no_alloc = true;
    pool.prefill_ctx = ggml_init(prefill_params);
    pool.prefill_tensor = ggml_new_tensor_3d(pool.prefill_ctx, src->type,
            src->ne[0], src->ne[1], src->ne[2]);
    GGML_ASSERT(pool.prefill_tensor->nb[2] == src->nb[2]);
    ggml_format_name(pool.prefill_tensor, "%s_prefill_ring", src->name);

    layer.expected_uses++;
    layer.name_to_pool_idx[src->name] = layer.pools.size();
    layer.pools.push_back(std::move(pool));

    ggml_metal_prefill_pool & stored = layer.pools.back();
    pool_to_layer[stored.prefill_tensor] = &layer;
}

ggml_tensor * ggml_metal_prefill_offload::bind_pool(const char * name, int layer_idx) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(build_mtx);

    if (layer_idx < 0 || (size_t) layer_idx >= layers.size()) {
        return nullptr;
    }
    const auto & layer_ptr = layers[(size_t) layer_idx];
    if (!layer_ptr) {
        return nullptr;
    }
    auto it = layer_ptr->name_to_pool_idx.find(name);
    if (it == layer_ptr->name_to_pool_idx.end()) {
        return nullptr;
    }
    return layer_ptr->pools[it->second].prefill_tensor;
}

void ggml_metal_prefill_offload::finalize_buffers() {
    std::lock_guard<std::mutex> lock(build_mtx);
    if (buffers_finalized_.load(std::memory_order_relaxed)) {
        return;
    }

    int n_layers = 0;
    for (auto & l : layers) {
        if (l) n_layers++;
    }
    if (n_layers == 0) {
        return;
    }

    cfg_n_buffers = std::min(cfg_n_buffers, n_layers);
    ggml_backend_buffer_type_t buft = ggml_backend_metal_get_shared_buffer_type(backend);

    // one full-layer ring per (suffix, stride) group
    std::map<std::string, size_t> group_experts;  // group -> n_expert (for the buffer size)
    for (auto & layer_ptr : layers) {
        if (!layer_ptr) continue;
        for (auto & pool : layer_ptr->pools) {
            group_experts[pool.group_key] = (size_t) pool.prefill_tensor->ne[2];
        }
    }
    for (auto & [key, n_expert] : group_experts) {
        const size_t stride = [this, &key]() -> size_t {
            for (auto & layer_ptr : layers) {
                if (!layer_ptr) continue;
                for (auto & pool : layer_ptr->pools) {
                    if (pool.group_key == key) return pool.stride;
                }
            }
            return 0;
        }();
        const size_t bytes = (size_t) cfg_n_buffers * n_expert * stride;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, bytes);
        GGML_ASSERT(buf);
        group_buffers[key] = buf;
        group_bases[key] = (uint8_t *) ggml_backend_buffer_get_base(buf);
    }

    for (auto & layer_ptr : layers) {
        if (!layer_ptr) continue;
        const int ring_idx = layer_ptr->order % cfg_n_buffers;
        for (auto & pool : layer_ptr->pools) {
            const size_t n_expert = (size_t) pool.prefill_tensor->ne[2];
            pool.prefill_tensor->data =
                group_bases.at(pool.group_key) + (size_t) ring_idx * n_expert * pool.stride;
            pool.prefill_tensor->buffer = group_buffers.at(pool.group_key);
        }
    }

    ring.assign((size_t) cfg_n_buffers, {});
    last_req.assign((size_t) n_layers, 0);
    signaled.assign((size_t) n_layers, 0);
    passes.assign((size_t) n_layers, 0);
    buffers_finalized_.store(true, std::memory_order_release);
}

void ggml_metal_prefill_offload::prepare_prefill() {
    finalize_buffers();
    if (!buffers_finalized_.load(std::memory_order_acquire)) {
        return;
    }
    if (active_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    rotation_.store(false, std::memory_order_release);
    while (rotation_busy_.load(std::memory_order_acquire)) {
        usleep(50);
    }

    int n = 0;
    for (auto & layer_ptr : layers) {
        if (!layer_ptr) continue;
        auto * req = (std::atomic<uint32_t> *) (layer_ptr->mapped + MOE_OFF_REQ);
        last_req[(size_t) layer_ptr->order] = req->load(std::memory_order_acquire);
        passes[(size_t) layer_ptr->order] = 0;
        signaled[(size_t) layer_ptr->order] = last_req[(size_t) layer_ptr->order];
        layer_ptr->last_src2 = nullptr;
        layer_ptr->last_src2_uses = 0;
        n++;
    }
    for (auto & state : ring) {
        state = {};
    }
    gpu_progress = -1;
    next_load_cumulative = 0;

    const int initial = std::min(cfg_n_buffers, n);
    for (int i = 0; i < initial; ++i) {
        load_layer(i);
    }
    next_load_cumulative = initial;
    rotation_.store(true, std::memory_order_release);

    start_sidecar();
}

void ggml_metal_prefill_offload::release_prefill_buffers() {
    // the group buffers stay allocated (the graphs reference them); this only ends the rotation
    // so a decode phase never waits on a layer that will not be loaded
    active_.store(false, std::memory_order_release);
    rotation_.store(false, std::memory_order_release);
    while (rotation_busy_.load(std::memory_order_acquire)) {
        usleep(50);
    }
    stop_sidecar();
}

void ggml_metal_prefill_offload::load_layer(int64_t cumulative) {
    // find the layer with this order
    const int n_layers = (int) last_req.size();
    const int order = (int) (cumulative % n_layers);
    ggml_metal_prefill_layer * layer = nullptr;
    for (auto & layer_ptr : layers) {
        if (layer_ptr && layer_ptr->order == order) {
            layer = layer_ptr.get();
            break;
        }
    }
    GGML_ASSERT(layer != nullptr);
    const int ring_idx = order % cfg_n_buffers;

    struct task_t { ggml_metal_prefill_pool * pool; int64_t dst; };
    std::vector<task_t> tasks;
    for (auto & pool : layer->pools) {
        const int64_t n_expert = pool.prefill_tensor->ne[2];
        tasks.reserve(tasks.size() + (size_t) n_expert);
        for (int64_t e = 0; e < n_expert; ++e) {
            tasks.push_back({ &pool, e });
        }
    }

    std::vector<uint8_t> failed(tasks.size(), 0);
    uint8_t * failed_data = failed.data();
    dispatch_apply(tasks.size(), DISPATCH_APPLY_AUTO, ^(size_t i) {
        const task_t & t = tasks[i];
        ggml_metal_prefill_pool & pool = *t.pool;
        const uint64_t off = pool.file_offset + (uint64_t) t.dst * pool.stride;
        uint8_t * dst = (uint8_t *) pool.prefill_tensor->data + (size_t) t.dst * pool.stride;

        size_t got = 0;
        while (got < pool.stride) {
            ssize_t nread = pread(pool.fd, dst + got, pool.stride - got, (off_t) (off + got));
            if (nread < 0 && errno == EINTR) {
                continue;
            }
            if (nread <= 0) {
                failed_data[i] = 1;
                break;
            }
            got += (size_t) nread;
        }
    });
    if (std::any_of(failed.begin(), failed.end(), [](uint8_t v) { return v != 0; })) {
        GGML_ABORT("moe prefill: layer pread failed (layer order %d)", order);
    }
    std::atomic_thread_fence(std::memory_order_release);

    ring[(size_t) ring_idx].owner_cumulative = cumulative;
    ring[(size_t) ring_idx].loaded = true;
    service_waits(order);
}

void ggml_metal_prefill_offload::service_waits(int order) {
    const uint32_t req = last_req[(size_t) order];
    if (req == 0 || req == signaled[(size_t) order] || passes[(size_t) order] == 0) {
        return;
    }

    const int64_t cumulative =
        (passes[(size_t) order] - 1) * (int64_t) last_req.size() + order;
    const int ring_idx = order % cfg_n_buffers;
    const ring_state & state = ring[(size_t) ring_idx];
    if (state.loaded && state.owner_cumulative == cumulative) {
        ggml_metal_prefill_layer * layer = nullptr;
        for (auto & layer_ptr : layers) {
            if (layer_ptr && layer_ptr->order == order) {
                layer = layer_ptr.get();
                break;
            }
        }
        GGML_ASSERT(layer != nullptr);
        ggml_backend_metal_event_signal(layer->event, req);
        signaled[(size_t) order] = req;
    }
}

bool ggml_metal_prefill_offload::poll_progress() {
    bool any = false;
    const int n_layers = (int) last_req.size();
    for (int order = 0; order < n_layers; ++order) {
        ggml_metal_prefill_layer * layer = nullptr;
        for (auto & layer_ptr : layers) {
            if (layer_ptr && layer_ptr->order == order) {
                layer = layer_ptr.get();
                break;
            }
        }
        if (layer == nullptr) continue;
        auto * req_ptr = (std::atomic<uint32_t> *) (layer->mapped + MOE_OFF_REQ);
        const uint32_t req = req_ptr->load(std::memory_order_acquire);
        if (req != last_req[(size_t) order]) {
            last_req[(size_t) order] = req;
            passes[(size_t) order]++;
            const int64_t cumulative =
                (passes[(size_t) order] - 1) * (int64_t) n_layers + order;
            gpu_progress = std::max(gpu_progress, cumulative);
            any = true;
        }
        service_waits(order);
    }
    return any;
}

bool ggml_metal_prefill_offload::step() {
    bool any = poll_progress();
    const int n_layers = (int) last_req.size();

    while (next_load_cumulative >= 0) {
        const int order = (int) (next_load_cumulative % n_layers);
        const int ring_idx = order % cfg_n_buffers;
        const ring_state & state = ring[(size_t) ring_idx];
        if (state.loaded && gpu_progress <= state.owner_cumulative) {
            break;   // the GPU may still be reading this ring slot
        }
        load_layer(next_load_cumulative);
        next_load_cumulative++;
        any = true;
    }
    return any;
}

bool ggml_metal_prefill_offload::hook(
        void * user_data, const ggml_tensor * src0, const ggml_tensor * src2,
        ggml_metal_moe_intercept * out) {
    auto * self = (ggml_metal_prefill_offload *) user_data;

    auto it = self->pool_to_layer.find(src0);
    if (it == self->pool_to_layer.end()) {
        return false;
    }
    ggml_metal_prefill_layer * layer = it->second;
    if (!self->active_.load(std::memory_order_acquire)) {
        return false;   // decode phase: the weights are not managed here
    }

    out->msg_tensor = layer->msg_tensor;
    out->event = layer->event;

    if (layer->last_src2 == src2 && layer->last_src2_uses < layer->expected_uses) {
        out->reuse = true;
        out->seq = layer->last_src2_seq;
        out->mode = GGML_METAL_MOE_MODE_WAIT;
        layer->last_src2_uses++;
        return true;
    }

    const uint32_t seq = (uint32_t) (layer->last_src2_seq + 1);
    layer->last_src2 = src2;
    layer->last_src2_seq = seq;
    layer->last_src2_uses = 1;

    out->reuse = false;
    out->seq = seq;
    out->mode = GGML_METAL_MOE_MODE_WAIT;   // beacon + event wait; the ids pass through untouched
    return true;
}

void ggml_metal_prefill_offload::start_sidecar() {
    if (sidecar_run_.exchange(true)) {
        return;
    }
    sidecar_thread = std::thread([this] {
        while (sidecar_run_.load(std::memory_order_relaxed)) {
            bool any = false;
            if (active_.load(std::memory_order_acquire)) {
                rotation_busy_.store(true, std::memory_order_release);
                if (rotation_.load(std::memory_order_acquire)) {
                    any = step();
                }
                rotation_busy_.store(false, std::memory_order_release);
            }
            if (!any) {
                usleep(10);
            }
        }
    });
}

void ggml_metal_prefill_offload::stop_sidecar() {
    if (!sidecar_run_.exchange(false)) {
        return;
    }
    if (sidecar_thread.joinable()) {
        sidecar_thread.join();
    }
}

#endif // __APPLE__

ggml_moe_prefill ggml_backend_metal_moe_prefill_init(ggml_backend_t backend, int n_buffers) {
    auto * state = new ggml_metal_prefill_offload(backend, n_buffers);
    return {
        state,
        [](void * p) { delete (ggml_metal_prefill_offload *) p; },
        [](void * p, const ggml_tensor * src, int layer, int fd, uint64_t offset) { ((ggml_metal_prefill_offload *) p)->add_pool(src, layer, fd, offset); },
        [](void * p, const char * name, int layer) { return ((ggml_metal_prefill_offload *) p)->bind_pool(name, layer); },
        [](void * p) { auto * s = (ggml_metal_prefill_offload *) p; s->finalize_buffers(); s->install_handler(); },
        [](void * p) { ((ggml_metal_prefill_offload *) p)->prepare_prefill(); },
        [](void * p) { ((ggml_metal_prefill_offload *) p)->release_prefill_buffers(); },
        [](const void * p) { return ((const ggml_metal_prefill_offload *) p)->active(); },
    };
}
