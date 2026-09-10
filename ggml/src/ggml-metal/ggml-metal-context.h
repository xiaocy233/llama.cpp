#pragma once

#include "ggml-metal-device.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

// MoE host-offload state accessors (ggml-metal-moe.m)
ggml_metal_device_t ggml_metal_get_dev(ggml_metal_t ctx);
void * ggml_metal_moe_get_raw(ggml_metal_t ctx);
void   ggml_metal_set_moe    (ggml_metal_t ctx, void * moe);

// stream routing: stream 0 is the GPU path, the rest are served by the host fill pool
int  ggml_metal_select_stream         (ggml_metal_t ctx, int stream);
void ggml_metal_set_tensor_async_stream(ggml_metal_t ctx, int stream, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_synchronize_stream    (ggml_metal_t ctx, int stream);

// file-sourced fill for disk-resident MoE banks: pread [file_off, file_off+size) of fd into
// tensor at offset. any stream goes to the host fill pool; stream 0 reads synchronously.
bool ggml_metal_set_tensor_async_file(ggml_metal_t ctx, int stream, struct ggml_tensor * tensor, int fd, uint64_t file_off, size_t offset, size_t size);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
