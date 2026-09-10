// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

//
// MoE expert streaming (disk-resident banks)
//

// shared (unified-memory) buffer type of the Metal device, for offloader-owned allocations
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_metal_get_shared_buffer_type(ggml_backend_t backend);

// a Metal shared event, usable from the host and encodable in command buffers
GGML_BACKEND_API void * ggml_backend_metal_event_new(ggml_backend_t backend);                // -> id<MTLSharedEvent>
GGML_BACKEND_API void   ggml_backend_metal_event_free(void * event);
GGML_BACKEND_API void   ggml_backend_metal_event_signal(void * event, uint64_t value);        // host-side signal
GGML_BACKEND_API void * ggml_backend_metal_event_raw(void * event);                          // -> id<MTLSharedEvent>

// mailbox layout shared between the interceptor kernel and the host offloader (bytes).
// The beacon request sequence is the only field today; one cache line keeps it free of
// false sharing with anything else the host may touch.
enum {
    MOE_OFF_REQ      = 0,   // atomic_uint: request seq (GPU writes)
    MOE_MSG_NBYTES   = 64,
};

// how a MUL_MAT_ID's weight bank is served by the MoE handler
enum ggml_metal_moe_mode {
    GGML_METAL_MOE_MODE_WAIT = 1,  // beacon + GPU-side event wait only (prefill layer ring)
};

struct ggml_metal_moe_intercept {
    struct ggml_tensor * msg_tensor;  // the mailbox tensor (device I8[MOE_MSG_NBYTES])
    void * event;                     // ggml_backend_metal_event_t of the owning layer
    bool reuse;                       // this src2 was already intercepted earlier in the graph
    uint32_t seq;                     // request sequence number for this interception
    enum ggml_metal_moe_mode mode;    // how the bank behind src0 is served
};

// called from MUL_MAT_ID encoding when src0 matches a registered pool tensor. return false to
// fall through to the regular path (the weight is resident)
typedef bool (*ggml_metal_moe_query_fn)(void * user_data,
                                        const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src2,
                                        struct ggml_metal_moe_intercept * out);

struct ggml_metal_moe_handler {
    ggml_metal_moe_query_fn fn;
    void * user_data;
};

GGML_BACKEND_API void ggml_backend_metal_set_moe_handler(ggml_backend_t backend, struct ggml_metal_moe_handler handler);

#ifdef __cplusplus
}
#endif
