#include "llama-model.h"
#include "llama-model-loader.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#if defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#endif

// ---- disk-resident MoE expert banks (params.cache_disk) ----
//
// A marker buffer type: the tensor keeps its metadata, nothing is allocated for data, and the
// expert bytes stay in the GGUF file. Consumers (the prefill layer ring, the decode fill pool)
// pread() them on demand. get_base returns an aligned pseudo-address (the OpenCL backend's trick):
// the allocator derives tensor->data from base + offset and ggml reads data != NULL as
// "allocated", but nothing ever dereferences it - every byte access goes through the fd.

static void llama_moe_disk_buffer_free(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
}

static void * llama_moe_disk_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) (uintptr_t) 32;
}

static void llama_moe_disk_buffer_set_abort(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ABORT("disk-resident MoE bank tensor written directly - the offloader must serve it");
    GGML_UNUSED(buffer); GGML_UNUSED(tensor); GGML_UNUSED(data); GGML_UNUSED(offset); GGML_UNUSED(size);
}

static void llama_moe_disk_buffer_get_abort(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ABORT("disk-resident MoE bank tensor read directly - the offloader must serve it");
    GGML_UNUSED(buffer); GGML_UNUSED(tensor); GGML_UNUSED(data); GGML_UNUSED(offset); GGML_UNUSED(size);
}

static const struct ggml_backend_buffer_i llama_moe_disk_buffer_i = {
    /* .free_buffer     = */ llama_moe_disk_buffer_free,
    /* .get_base        = */ llama_moe_disk_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ llama_moe_disk_buffer_set_abort,
    /* .get_tensor      = */ llama_moe_disk_buffer_get_abort,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ nullptr,
    /* .reset           = */ nullptr,
};

static const char * llama_moe_disk_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "DISK";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t llama_moe_disk_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // metadata-only buffer: the size is kept for accounting, no memory is allocated
    return ggml_backend_buffer_init(buft, llama_moe_disk_buffer_i, nullptr, size);
}

static size_t llama_moe_disk_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static bool llama_moe_disk_buft_is_host(ggml_backend_buffer_type_t buft) {
    // the loaders and the scheduler treat the bank as host-side: its bytes are one pread away
    GGML_UNUSED(buft);
    return true;
}

ggml_backend_buffer_type_t llama_moe_cache_disk_buft(void) {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ llama_moe_disk_buft_get_name,
            /* .alloc_buffer     = */ llama_moe_disk_buft_alloc_buffer,
            /* .get_alignment    = */ llama_moe_disk_buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ llama_moe_disk_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };

    return &buft;
}

// Record shared weight sources. Each context allocates its own slots and location maps.
void llama_model::init_moe_slot_caches(int n_slots, int n_pred, const llama_model_loader & ml) {
    moe_slot_layers.clear();

    if (n_slots <= 0) {
        return;
    }

    // collect the layers that stream: expert tensors present and host-resident (a DISK bank
    // counts as host-resident: its bytes are one pread away)
    struct cand { int il; ggml_tensor * src[3]; };
    std::vector<cand> cands;

    for (int il = 0; il < (int) layers.size(); il++) {
        // MTP/NextN layers (il >= n_layer) keep their experts resident: they are not part of the
        // streamed set regardless of any layer-range override
        if (il >= (int) hparams.n_layer()) {
            continue;
        }
        ggml_tensor * t[3] = {
            layers[il].ffn_gate_exps, layers[il].ffn_up_exps, layers[il].ffn_down_exps,
        };
        if (layers[il].ffn_gate_up_exps || layers[il].ffn_gate_chexps) {
            continue;
        }
        bool any = false;
        bool all_host = true;
        for (int m = 0; m < 3; m++) {
            if (t[m] == nullptr || t[m]->buffer == nullptr) {
                all_host = false;
                break;
            }
            any = true;
            all_host = all_host && ggml_backend_buffer_is_host(t[m]->buffer);
        }
        if (!any || !all_host) {
            continue;
        }

        cands.push_back({ il, { t[0], t[1], t[2] } });
    }

    if (cands.empty()) {
        return;
    }

    const int n_expert = (int) cands[0].src[0]->ne[2];
    if (n_slots >= n_expert) {
        return;
    }

    ggml_backend_buffer_type_t buft_dev = nullptr;
    ggml_backend_buffer_type_t buft_pinned = nullptr;
    for (const auto & d : devices) {
        if (d.is_meta || d.dev == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(d.dev);
        if (buft && !ggml_backend_buft_is_host(buft)) {
            buft_dev = buft;
            buft_pinned = ggml_backend_dev_host_buffer_type(d.dev);
            break;
        }
    }
    if (buft_dev == nullptr) {
        return;
    }

    std::vector<moe_slot_layer> built;
    for (const auto & c : cands) {
        moe_slot_layer L;
        L.layer    = c.il;
        L.n_expert = n_expert;
        L.n_slots  = n_slots;
        bool pinned = buft_pinned != nullptr;

#if defined(__APPLE__)
        // a disk-resident bank: wire the file source; the loader closes its files after load,
        // so the bank outlives them on a dup'ed fd
        if (c.src[0]->buffer != nullptr &&
                ggml_backend_buffer_get_type(c.src[0]->buffer) == llama_moe_cache_disk_buft()) {
            int fd0 = -1;
            uint64_t off0 = 0;
            if (!ml.get_disk_source(ggml_get_name(c.src[0]), &fd0, &off0)) {
                LLAMA_LOG_ERROR("%s: layer %d bank has no file source, not cached\n", __func__, c.il);
                continue;
            }
            int fd_dup = -1;
            // dup once per distinct original fd; the map is local to this build so a fit-context
            // load and a real load (or a model reload) never hand each other a stale fd
            std::map<int, int> fd_map;
            auto it = fd_map.find(fd0);
            if (it != fd_map.end()) {
                fd_dup = it->second;
            } else {
                fd_dup = dup(fd0);
                if (fd_dup < 0) {
                    LLAMA_LOG_ERROR("%s: failed to dup the model file fd: %s\n", __func__, strerror(errno));
                    continue;
                }
                // the disk pages the experts come from are the consumers' problem; the unified
                // buffer cache would just double-buffer them against the resident slots
                if (getenv("GGML_METAL_MOE_CACHE_PAGES") == nullptr) {
                    fcntl(fd_dup, F_NOCACHE, 1);
                }
                fd_map[fd0] = fd_dup;
                moe_disk_fds.push_back(fd_dup);
            }
            L.src_fd = fd_dup;
            bool ok = true;
            for (int m = 0; m < 3; m++) {
                int m_fd = -1;
                uint64_t m_off = 0;
                if (!ml.get_disk_source(ggml_get_name(c.src[m]), &m_fd, &m_off)) {
                    LLAMA_LOG_ERROR("%s: layer %d matrix %d has no file source\n", __func__, c.il, m);
                    ok = false;
                    break;
                }
                L.src_off[m] = m_off;
            }
            if (!ok) {
                continue;
            }
            pinned = false;
        }
#endif

        for (int m = 0; m < 3; m++) {
            L.src[m] = c.src[m];
            GGML_ASSERT(ggml_is_contiguous(L.src[m]));
            pinned = pinned && c.src[m]->buffer != nullptr &&
                    ggml_backend_buffer_get_type(c.src[m]->buffer) == buft_pinned;
        }
        L.pinned  = pinned;
        built.push_back(L);
    }

    if (built.empty()) {
        return;
    }

    if (n_pred > 0) {
        for (size_t i = 0; i + 1 < built.size(); i++) {
            const int il_next = built[i + 1].layer;
            ggml_tensor * w = layers[il_next].ffn_gate_inp;
            if (w == nullptr) {
                continue;
            }
            if (w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
                continue;
            }
            built[i].pred_w      = w;
            built[i].n_pred      = n_pred;
            built[i].pred_target = il_next;
        }

        const int il_target = built.front().layer;
        const int il_source = il_target - 1;
        ggml_tensor * w = il_target >= 0 && il_target < (int) layers.size()
            ? layers[il_target].ffn_gate_inp : nullptr;
        const bool source_is_moe = il_source >= 0 && layers[il_source].ffn_gate_inp != nullptr;
        if (source_is_moe && w != nullptr &&
                (w->buffer == nullptr || !ggml_backend_buffer_is_host(w->buffer))) {
            built.front().head_pred_w      = w;
            built.front().head_n_pred      = n_pred;
            built.front().head_pred_source = il_source;
        }
    }



    moe_slot_buft = buft_dev;
    moe_slot_layers = std::move(built);
}
