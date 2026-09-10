#pragma once

// Metal prefill layer-ring offloader: streams whole expert layers from the GGUF file into a small
// ring of full-layer Shared buffers while the GPU computes on the previous ones (reMOE structure,
// adapted to the moe_slot_layer fact layer of this tree).
//
// Structure:
//   - a ring of N full-layer buffers per matrix group (stride-keyed, so the odd q6_K down matrices
//     get their own group), assigned to streaming layers in round-robin order
//   - a sidecar thread that watches the GPU progress beacons (written by kernel_moe_interceptor
//     at each MUL_MAT_ID) and preloads layer L+N as soon as the GPU has left layer L's ring slot
//   - each MUL_MAT_ID on a pool tensor encodes beacon + encodeWaitForEvent(layer_event), so the
//     weights are guaranteed resident before the kernels that read them

#include "ggml-backend.h"
#include "../ggml-moe-backend.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)

#include "ggml-metal.h"

struct ggml_metal_prefill_pool {
    ggml_tensor *         prefill_tensor = nullptr;  // full-bank view into the ring, data patched
    ggml_context *        prefill_ctx    = nullptr;
    int                   fd            = -1;       // dup'ed model file fd (from the fact layer)
    uint64_t              file_offset   = 0;        // absolute offset of the bank in the file
    size_t                stride        = 0;        // bytes of one expert (nb[2])
    std::string           group_key;                // (suffix, stride) - one ring per group
};

struct ggml_metal_prefill_layer {
    int layer_idx = -1;
    int order     = -1;   // position among the streaming layers (ring assignment + sidecar order)

    std::vector<ggml_metal_prefill_pool> pools;
    std::unordered_map<std::string, size_t> name_to_pool_idx;

    // mailbox + event
    ggml_tensor *         msg_tensor = nullptr;
    ggml_context *        msg_ctx    = nullptr;
    ggml_backend_buffer_t msg_buf    = nullptr;
    uint8_t *             mapped     = nullptr;     // == msg_tensor->data
    void *                event      = nullptr;     // ggml_backend_metal_event_t

    // hook state (same src2 across the 2-3 matrices of a layer = one interception)
    const ggml_tensor * last_src2      = nullptr;
    uint32_t            last_src2_seq  = 0;
    int                 last_src2_uses = 0;
    int                 expected_uses  = 0;

    ~ggml_metal_prefill_layer();
};

class ggml_metal_prefill_offload {
  public:
    ggml_metal_prefill_offload(ggml_backend_t backend, int n_prefill_buffers);
    ~ggml_metal_prefill_offload();

    // create the pool for one disk-resident bank tensor (shape/type/stride taken from src). Has to
    // run for every bank before the first graph build: the pool tensors enter the graphs as
    // weights, and the scheduler sizes its compute buffers against whatever the graphs reference.
    // Creating them lazily during graph builds would leave them buffer-less at reserve time and
    // the allocator would carve full-bank space for each one out of the compute buffer.
    // fd is one of the model's dup'ed fds (moe_disk_fds); this only borrows it.
    void add_pool(const ggml_tensor * src, int layer_idx, int fd, uint64_t file_offset);

    // bind a disk-resident bank tensor to its ring slot; returns the pool tensor to use as
    // MUL_MAT_ID src0, or nullptr when the tensor is not managed here
    ggml_tensor * bind_pool(const char * name, int layer_idx);

    void finalize_buffers();
    void install_handler() {
        ggml_metal_moe_handler handler = { &ggml_metal_prefill_offload::hook, this };
        ggml_backend_metal_set_moe_handler(backend, handler);
    }
    void prepare_prefill();
    void release_prefill_buffers();
    bool active() const { return active_.load(std::memory_order_acquire); }

    static bool hook(void * user_data, const ggml_tensor * src0, const ggml_tensor * src2,
                     ggml_metal_moe_intercept * out);

  private:
    void load_layer(int64_t cumulative);
    void service_waits(int order);
    bool poll_progress();
    bool step();
    void start_sidecar();
    void stop_sidecar();

    ggml_backend_t backend;
    int cfg_n_buffers;

    std::mutex build_mtx;
    std::vector<std::unique_ptr<ggml_metal_prefill_layer>> layers;
    std::unordered_map<const ggml_tensor *, ggml_metal_prefill_layer *> pool_to_layer;
    std::unordered_map<std::string, ggml_backend_buffer_t> group_buffers;  // group_key -> buffer
    std::unordered_map<std::string, uint8_t *> group_bases;

    struct ring_state {
        int64_t owner_cumulative = -1;
        bool    loaded = false;
    };
    std::vector<ring_state> ring;

    // sidecar-visible progress (beacon polling)
    std::vector<uint32_t> last_req;      // per order
    std::vector<uint32_t> signaled;      // per order
    std::vector<int64_t>  passes;        // per order
    int64_t next_load_cumulative = 0;
    int64_t gpu_progress = -1;           // cumulative index the GPU has reached (inclusive)

    std::atomic<bool> active_{false};
    std::atomic<bool> rotation_{false};
    std::atomic<bool> rotation_busy_{false};
    std::atomic<bool> buffers_finalized_{false};
    std::thread sidecar_thread;
    std::atomic<bool> sidecar_run_{false};
};

#endif

ggml_moe_prefill ggml_backend_metal_moe_prefill_init(ggml_backend_t backend, int n_buffers);
