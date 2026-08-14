// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

// select an auxiliary stream for subsequent async operations on this backend
// returns the previously selected stream index, or -1 if the backend does not support it
static int ggml_backend_select_stream(ggml_backend_t backend, int stream) {
    GGML_ASSERT(backend);

    if (backend->iface.select_stream == NULL) {
        return -1;
    }

    return backend->iface.select_stream(backend, stream);
}

bool ggml_backend_moe_gate_channel(ggml_backend_t backend, struct ggml_moe_gate_channel * out) {
    GGML_ASSERT(backend);
    GGML_ASSERT(out);

    if (backend->iface.moe_gate_channel == NULL) {
        return false;
    }

    return backend->iface.moe_gate_channel(backend, out);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

// ring depth for the weight prefetch (ggml_backend_sched_set_weight_prefetch)
// note: must be outside the GGML_SCHED_MAX_COPIES guard, that one is set by CMake
#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS
// one entry is one expert matrix, so a layer's worth of gate/up/down is three entries. 12 leaves
// room to hold several layers ahead without another rebuild.
#define GGML_SCHED_MAX_PREFETCH_SLOTS 12
#endif

// the ring is clamped up to this depth whenever prefetch is enabled at all.
// must stay outside the guard above - that one may be predefined, which would skip this.
#ifndef GGML_SCHED_MIN_PREFETCH_SLOTS
#define GGML_SCHED_MIN_PREFETCH_SLOTS 3
#endif

static_assert(GGML_SCHED_MIN_PREFETCH_SLOTS >= 2,
        "a ring depth below 2 is a data race: the copy for entry k+1 is issued before entry k's compute");
static_assert(GGML_SCHED_MIN_PREFETCH_SLOTS <= GGML_SCHED_MAX_PREFETCH_SLOTS,
        "the prefetch ring minimum cannot exceed the maximum");

// auxiliary stream the prefetch copies are issued on. stream 0 is the compute stream, so anything
// else gives concurrency; backends without an auxiliary stream ignore this and stay serialized.
#define GGML_SCHED_PREFETCH_STREAM_INDEX 1

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

#ifndef GGML_SCHED_MAX_MOE_CACHES
#define GGML_SCHED_MAX_MOE_CACHES 128
#endif

// upper bound on how many experts a layer may predict, i.e. the largest -ncpred that has an effect
#define GGML_SCHED_MAX_MOE_PRED 64

// bounce buffers for pageable expert banks; one expert needs 3, so this holds 4 experts in flight
#define GGML_SCHED_MOE_STAGE 12

// Per-layer state for one MoE slot cache. The device side is the caller's `slots` tensors; this is
// the host-side bookkeeping that decides what goes where.
struct ggml_moe_slot_state {
    struct ggml_moe_slot_cache cache;

    int n_expert;
    int n_slots;                // device cache capacity
    int n_expert_used;
    size_t expert_nb[3];        // bytes of one expert in each matrix (host and slots share nb)

    int32_t  * slot_of_expert;  // [n_expert]  -1 = not on GPU. mirrors loc_map
    int32_t  * expert_of_slot;  // [n_slots]   -1 = empty
    uint32_t * freq;            // [n_expert]  LFU counter
    uint32_t * last_tok;        // [n_expert]  epoch+1 of the last use, 0 = never used
    uint32_t * pinned;          // [n_slots]   epoch of the last hit; a slot pinned this token is
                                //             being read by the GPU and must not be overwritten
    uint32_t epoch;             // bumped once per fill, i.e. once per token for this layer
    bool loc_dirty;             // device loc_map is behind slot_of_expert

    // Free-slot pool. A slot in here holds nothing and no loc_map entry points at it, so a prefetch
    // may write it while the target layer runs. That is the only safe target: the target layer has
    // not passed its gate, so it can resolve any resident slot at any moment. Slots enter the pool
    // by being retired two gates back, a whole token before that layer is read again, which is what
    // makes the invalidate certain to have landed. See probe/GATE-PLAN.md section 4.2.
    int32_t  * free_slot;       // [n_slots]
    int        free_cnt;
    uint32_t * free_pass;       // [n_slots] worker sync pass that retired the slot
    uint32_t * fill_seq;        // [n_slots] token that last wrote the slot, blocks a same-token retire

    // Source of every loc_map row write. Page-locked, because a pageable H2D stalls the whole
    // device on Windows. Page-locked also means the copy reads it whenever it likes, so a second
    // write to the same expert has to wait for the stream - pub_pass is what detects that.
    ggml_backend_buffer_t pub_buf;
    int32_t  * loc_pub;         // [n_expert]
    uint32_t * pub_pass;        // [n_expert] worker sync pass that last published the expert

    // ---- diagnostics (GGML_SCHED_MOE_SLOT_STATS) ----
    // Decode only: the remap node these counters hang off is emitted for one-token graphs only, so
    // prefill never reaches the fill and never contributes here.
    //
    // A "load" is one expert, meaning its up/gate/down slices together. The three slices are three
    // backend calls but one expert load - that is the unit the cache deals in.
    uint64_t n_hit;
    uint64_t n_miss;
    uint64_t n_tokens;

    uint64_t n_load_sync, n_bytes_sync;    // brought in by the blocking fill on the consuming layer
    uint64_t n_load_pf,   n_bytes_pf;      // brought in ahead of use

    // Prediction quality (pf_prec): |predicted ∩ actual| / |predicted|. Counts the router's guess
    // only - not whether a copy was issued, landed, or the expert was already resident.
    uint64_t n_pf_pred;                    // experts predicted for this layer
    uint64_t n_pf_useful;                  // of those, how many the layer then actually used

    // Of the experts this token uses, how many this layer also used inside the last n_slots /
    // n_expert_used tokens. That is the ceiling for a cache that never learns anything about the
    // token it is serving - the hit rate left if the per-layer host round trip goes away.
    uint64_t n_lag_reach;
    uint64_t n_lag_total;

    int  dev_backend_id;          // backend owning slots, -1 if it could not be determined
    int  gate_slot;               // mailbox index, also the retire order
    uint32_t last_seen;           // last mailbox seq the worker consumed for this layer

    // prediction the last gate of this layer published, scored against the next token's ids
    int32_t  pred_batch[GGML_SCHED_MAX_MOE_PRED];
    int      n_pred_batch;

    uint64_t n_pf_try;            // prefetches wanted
    uint64_t n_pf_dry;            // of those, skipped because the target pool was empty
    uint64_t n_retire;
    uint64_t n_lost;              // mailbox entries the worker never saw
    uint64_t n_stage;             // slices copied through the bounce buffer
};

struct ggml_moe_worker {
    std::thread       thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> idle{true};   // a full pass found no new mailbox entry
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // env: GGML_SCHED_FULL_WEIGHT_COPY
    // copy host-resident MoE expert weights in full instead of scanning the routing ids for the
    // used experts. At batch sizes where every expert is selected this copies the same number of
    // bytes, but it removes the host round-trip of the ids tensor from the per-layer critical path.
    bool full_weight_copy;

    // env: GGML_SCHED_EXPERT_STATS
    bool expert_stats;

    // env: GGML_SCHED_PREFETCH_VERIFY
    // read every prefetched weight copy back from the device right before its consumer runs and
    // compare it against the host source. Slow, but it turns a protocol violation into an explicit
    // abort at the offending split instead of subtly wrong output.
    bool prefetch_verify;

    // MoE expert slot caches, registered by the caller before the first reserve.
    // Fixed array rather than a vector: the sched is calloc'd, so no member may need construction.
    struct ggml_moe_slot_state * moe_caches[GGML_SCHED_MAX_MOE_CACHES];
    int  n_moe_caches;
    bool moe_debug;   // env: GGML_SCHED_MOE_SLOT_DEBUG
    bool moe_stats;   // env: GGML_SCHED_MOE_SLOT_STATS

    // decode gate: mailbox, the token counter the gates stamp, and the thread that services them.
    // The worker is behind a pointer because the sched is calloc'd and a thread needs construction.
    struct ggml_moe_gate_channel moe_ch;
    struct ggml_tensor *         moe_seq_t;
    uint32_t                     moe_seq;
    bool                         moe_prerelease; // let the gates of this pass through without waiting
    uint64_t                     moe_probe_uid;  // graph uid the discovery pass last ran for
    uint32_t                     moe_probe_seq;  // its seq, so its parks stay out of the park stats
    uint32_t                     moe_warm_seq;   // seq of the prefill report decode warms from, 0 once used
    int                          moe_warm_max;   // cap on experts warmed per layer
    uint64_t                     n_warm;         // experts the prefill report placed
    uint64_t                     n_probe;        // experts the discovery pass placed
    uint64_t                     w_stale;        // gate used a row the host had already dropped
    uint64_t                     w_unpark;       // worker changed a cache whose gate was not waiting
    uint64_t                     w_victim;       // a fill overwrote a slot this layer's own ids use
    int                          moe_krel_tok;   // decode tokens whose gates the device releases
    uint64_t                     w_krel;         // releases handed to the device
    struct ggml_moe_worker *     moe_worker;
    bool                         has_moe_gate;   // set by split_graph, true on a decode graph
    bool                         has_moe_warm;   // ... true on a prefill graph that reports routing
    uint32_t                     moe_pass;       // bumped on every worker stream sync

    // Bounce buffers for layers whose host bank is not page-locked. Handing the driver a pageable
    // source lets it stage the copy on its own terms, and that staging can wait for the device -
    // against a parked gate, which only this worker can release, that is a deadlock. So the worker
    // stages into page-locked memory itself. Same pass trick as pub_pass: a buffer can be reused
    // once the stream that read it has been waited on.
    ggml_backend_buffer_t        moe_stage_buf;
    char *                       moe_stage[GGML_SCHED_MOE_STAGE];
    uint32_t                     moe_stage_pass[GGML_SCHED_MOE_STAGE];
    size_t                       moe_stage_sz;
    int                          moe_stage_i;

    // Where the worker spends a park (GGML_SCHED_MOE_SLOT_STATS). A park that runs past the gate
    // timeout is a deadlock, not a slow copy, and these say which call it deadlocked in. Three
    // phases: the bounce memcpy, issuing the copies, and waiting for the stream.
    uint64_t w_stage_us, w_issue_us, w_sync_us;   // the park being serviced, reset when it starts
    uint64_t w_n, w_slow;                         // parks serviced, and those over 10 ms
    uint64_t w_tot_us, w_tot_stage, w_tot_issue, w_tot_sync;
    uint64_t w_max_us, w_max_stage, w_max_issue, w_max_sync;
    uint64_t w_call_us;                           // slowest single call of the run
    int      w_call_kind;                         // 0 stage, 1 issue, 2 sync
    int      w_max_layer, w_max_fill;             // where the worst park was, and how much it moved
    bool     w_max_pinned;
    uint64_t w_max_tok;                           // which token of that layer, 1 = the first decode
    uint64_t w_warm_us;                           // first use of the worker stream, paid up front

    // Where a decode step spends its wall time on the host, summed over the splits of one graph.
    // Only steps that ran a fill are counted, so this is decode and never prefill.
    //
    // t_drain is the point of the whole thing: it is measured with an extra synchronize at the end
    // of the step. Large means the device is behind and the host waits are already hidden; near
    // zero means the host paces the step and removing its waits is what would make decode faster.
    uint64_t t_copy_us;    // split input copies, i.e. the ids round trip
    uint64_t t_prod_us;    // of t_copy_us: waiting for the device to produce the input
    uint64_t t_fill_us;    // fill and publish
    uint64_t t_step_us;    // the whole step
    uint64_t t_drain_us;   // waiting for the device after the last split was enqueued
    uint64_t n_step;

    // set through ggml_backend_sched_set_weight_prefetch
    // Overlap host->device MoE expert weight copies with compute: the copy for the next qualifying
    // split is issued on an auxiliary stream while the current split computes, and the consuming
    // split waits on a per-slot event instead of a full device synchronize.
    //
    // Requires:
    //  - n_copies == 1 (multi-GPU pipeline parallelism has its own double buffering)
    //  - the backend implements select_stream, event_record and event_wait
    //  - full_weight_copy, otherwise the copy depends on the routing ids of its own layer and
    //    cannot be issued ahead of the previous split's compute
    // 0 disables the feature (default)
    int prefetch_slots;   // ring depth, 0 = disabled
    bool prefetch_reported;

    // ev_copy[i] signals "entry i mod slots has landed"
    ggml_backend_event_t prefetch_ev_copy[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_PREFETCH_SLOTS];
    // recorded after every split's compute: "all work enqueued so far has completed"
    ggml_backend_event_t prefetch_ev_compute_last[GGML_SCHED_MAX_BACKENDS];

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_WARN("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_WARN("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
// true if this split input is a host-resident MoE expert weight consumed by the split's first
// MUL_MAT_ID. Such an input is the only kind that can be copied ahead of time: its contents are
// constant for the whole graph and it has no producer in the graph.
// note: mirrors the condition of the used-experts copy path in ggml_backend_sched_compute_splits
static bool ggml_backend_sched_is_prefetchable_weight(
        const struct ggml_backend_sched_split * split,
        const struct ggml_tensor * input,
        const struct ggml_tensor * input_cpy) {
    if (split->graph.n_nodes == 0 || input->buffer == NULL) {
        return false;
    }

    if (ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_backend_buffer_is_host(input->buffer)) {
        return false;
    }

    const struct ggml_tensor * node = split->graph.nodes[0];

    if (node->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    // classic Prefill: weight is src[0]; dual Prefill: compact host weight is src[3]
    const bool classic = (node->src[0] == input_cpy);
    const bool dual    = (node->src[3] == input_cpy && node->src[3] != NULL && node->src[4] != NULL);
    if (!classic && !dual) {
        return false;
    }

    // the prefetch copies without looking at the ids, so it must only engage where the used-experts
    // path would have copied (nearly) everything anyway. this mirrors the condition applied to the
    // full-copy shortcut in compute_splits; without it, decode (a handful of assignments out of
    // hundreds of experts) would move the entire tensor per token instead of a few experts.
    //
    // both operands are shapes, known at graph build time, so this needs no ids readback and
    // introduces no dependency on the router.
    const int64_t n_expert = input->ne[2];
    const int64_t n_assign = ggml_nelements(node->src[2]);

    return n_assign >= n_expert;
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // A parking gate means a decode graph, and that is what tells the worker a token is coming. A
    // prefill graph carries publish-only gates (n_ids == 0), which the main thread drains itself.
    sched->has_moe_gate = false;
    sched->has_moe_warm = false;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->op == GGML_OP_MOE_GATE) {
            if (ggml_get_op_params_i32(graph->nodes[i], 1) > 0) {
                sched->has_moe_gate = true;
                break;
            }
            sched->has_moe_warm = true;
        }
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs >= split->inputs_capacity) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    // ---- weight prefetch: hoist the weight copies earlier in the node order ----
    //
    // ggml-alloc gives a tensor its memory when its node position is reached, and hands that memory
    // to other tensors before that point. Prefetching writes the copy before its node position, so
    // the copy would land in memory that ggml-alloc has assigned to something else. Emitting the
    // copy node D entries earlier makes its lifetime start where the prefetch is issued, which is
    // what makes the prefetch legal at all.
    //
    // entry k is issued in compute_splits just before the compute of pf[k - D].split, so that is
    // where its node has to be emitted. Entries below D are emitted at the very start of the graph.
    const int pf_depth = sched->prefetch_slots > 0 ? sched->prefetch_slots - 1 : 0;

    struct pf_entry { int split; int input; };
    std::vector<pf_entry> pf;

    if (pf_depth > 0) {
        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * sp = &sched->splits[i];
            sp->graph = ggml_graph_view(graph, sp->i_start, sp->i_end);

            for (int j = 0; j < sp->n_inputs; j++) {
                struct ggml_tensor * in     = sp->inputs[j];
                struct ggml_tensor * in_cpy = tensor_id_copy(hash_id(in), sp->backend_id, sched->cur_copy);

                if (ggml_backend_sched_is_prefetchable_weight(sp, in, in_cpy)) {
                    pf.push_back({ i, j });
                    break;
                }
            }
        }
    }

    // emit_at_split[i] = index into pf of the entry whose copy node is emitted when split i starts
    std::vector<int> emit_at_split((size_t) sched->n_splits, -1);
    for (size_t k = (size_t) pf_depth; k < pf.size(); k++) {
        emit_at_split[pf[k - pf_depth].split] = (int) k;
    }

    auto emit_copy_node = [&](const pf_entry & e) {
        struct ggml_backend_sched_split * sp = &sched->splits[e.split];
        struct ggml_tensor * in     = sp->inputs[e.input];
        struct ggml_tensor * in_cpy = tensor_id_copy(hash_id(in), sp->backend_id, sched->cur_copy);

        assert(graph_copy->size > graph_copy->n_nodes);
        sched->node_backend_ids[graph_copy->n_nodes] = sp->backend_id;
        graph_copy->nodes[graph_copy->n_nodes++] = in_cpy;
    };

    // the first pf_depth entries have nothing to hide behind, allocate them up front
    for (size_t k = 0; k < pf.size() && k < (size_t) pf_depth; k++) {
        emit_copy_node(pf[k]);
    }

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split.
            // prefetched weight copies are emitted earlier instead (see above), skip them here.
            const bool hoisted = pf_depth > 0 &&
                ggml_backend_sched_is_prefetchable_weight(split, input, input_cpy);

            if (!hoisted) {
                sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
            }
        }

        // emit the copy node of the entry that will be issued during this split's compute
        if (emit_at_split[i] >= 0) {
            emit_copy_node(pf[emit_at_split[i]]);
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static const char * ggml_backend_sched_moe_host_ptr(
        const struct ggml_moe_slot_state * st, int m, int32_t e) {
    GGML_ASSERT(e >= 0 && e < st->n_expert);
    return (const char *) st->cache.src[m]->data + (size_t) e * st->expert_nb[m];
}

// The cache that streams the host expert bank [src], or NULL if [src] is not such a bank.
static const struct ggml_moe_slot_state * ggml_backend_sched_moe_bank_of(
        ggml_backend_sched_t sched, const struct ggml_tensor * src) {
    for (int c = 0; c < sched->n_moe_caches; c++) {
        for (int m = 0; m < 3; m++) {
            if (sched->moe_caches[c]->cache.src[m] == src) {
                return sched->moe_caches[c];
            }
        }
    }

    return NULL;
}

// Walk the ranges of the host expert bank [src] that a Prefill copy must move, starting the scan at
// expert [*e]. Returns false when done. An expert that a slot holds is skipped: loc_map sends it to
// the slot, so its copy in the device bank is never read. That saves n_slots of n_expert bytes.
// [st] = NULL yields the whole tensor, for a bank that is not streamed.
static bool ggml_backend_sched_moe_bank_range(
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

// loc_map[e] = GPU slot index, or n_slots + e to read the host bank. The decode gate reads the
// second form as "not resident"; a Prefill dual-base MUL_MAT_ID reads it as a ring offset.
//
// Blocking, main thread. Prefill only: decode publishes from the worker, one row at a time.
static void ggml_backend_sched_moe_publish_loc(struct ggml_moe_slot_state * st) {
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

// Pick the slot a missing expert goes into: a free one while the cache is still warming, else the
// least frequently used resident. Returns -1 if every slot is protected.
//
// `same_tok` refuses a slot this token already wrote. The retire path needs that: retiring a slot a
// prefetch just filled would throw the copy away and, worse, hand the same slot out twice.
static int32_t ggml_backend_sched_moe_victim(struct ggml_moe_slot_state * st, bool same_tok) {
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

// Take a slot out of the pool, or -1 if none can be used yet.
//
// A slot retired in this pass is not usable. Retiring drops the host side of the mapping at once but
// only queues the invalidate, so a gate can still be reading the old row and resolving to this very
// slot. Waiting one sync is what makes the invalidate certain to have landed first.
static int32_t ggml_backend_sched_moe_pool_take(
        ggml_backend_sched_t sched, struct ggml_moe_slot_state * st) {
    for (int i = st->free_cnt - 1; i >= 0; i--) {
        const int32_t v = st->free_slot[i];
        if (st->free_pass[v] == sched->moe_pass) {
            continue;
        }
        st->free_slot[i] = st->free_slot[--st->free_cnt];
        return v;
    }
    return -1;
}

// ---- MoE gate worker ----
//
// One thread, started with the first cache. It consumes what the gates publish, fills what is
// missing, releases the parked gate, and prefetches what the routers predicted. It is the only
// writer of the slots and of loc_map during decode.
//
// It runs concurrently with the main thread, so it must never touch the backend's current stream.
// Everything it issues goes through set_tensor_async_stream / synchronize_stream on its own stream.

#define GGML_SCHED_MOE_WORKER_STREAM 3

// enough that a layer rarely needs two prefetch slots in one token; the probe measured a 1.8% dry
// rate here and 17.6% at one, see probe/GATE-PLAN.md section 5.3
#define GGML_SCHED_MOE_POOL 2

enum { GGML_SCHED_MOE_W_STAGE = 0, GGML_SCHED_MOE_W_ISSUE = 1, GGML_SCHED_MOE_W_SYNC = 2 };

static void ggml_backend_sched_moe_tick(ggml_backend_sched_t sched, int kind, int64_t t0) {
    const uint64_t dt = (uint64_t) (ggml_time_us() - t0);
    switch (kind) {
        case GGML_SCHED_MOE_W_STAGE: sched->w_stage_us += dt; break;
        case GGML_SCHED_MOE_W_ISSUE: sched->w_issue_us += dt; break;
        default:                     sched->w_sync_us  += dt; break;
    }
    if (dt > sched->w_call_us) {
        sched->w_call_us   = dt;
        sched->w_call_kind = kind;
    }
}

// Everything the worker issues goes to one stream, so one wait covers the payloads and the rows.
static void ggml_backend_sched_moe_sync(ggml_backend_sched_t sched, ggml_backend_t bk) {
    const int64_t t0 = sched->moe_stats ? ggml_time_us() : 0;
    bk->iface.synchronize_stream(bk, GGML_SCHED_MOE_WORKER_STREAM);
    if (sched->moe_stats) {
        ggml_backend_sched_moe_tick(sched, GGML_SCHED_MOE_W_SYNC, t0);
    }
    sched->moe_pass++;
}

// Copy a pageable slice into page-locked memory, so the driver never has to stage it itself.
static const char * ggml_backend_sched_moe_stage(
        ggml_backend_sched_t sched, ggml_backend_t bk, const char * src, size_t nb) {
    if (sched->moe_stage_buf == NULL || nb > sched->moe_stage_sz) {
        return src;
    }
    const int i = sched->moe_stage_i;
    if (sched->moe_stage_pass[i] == sched->moe_pass) {
        // the copy that last read this buffer may still be running
        ggml_backend_sched_moe_sync(sched, bk);
    }
    const int64_t t0 = sched->moe_stats ? ggml_time_us() : 0;
    memcpy(sched->moe_stage[i], src, nb);
    if (sched->moe_stats) {
        ggml_backend_sched_moe_tick(sched, GGML_SCHED_MOE_W_STAGE, t0);
    }
    sched->moe_stage_pass[i] = sched->moe_pass;
    sched->moe_stage_i       = (i + 1) % GGML_SCHED_MOE_STAGE;
    return sched->moe_stage[i];
}

static void ggml_backend_sched_moe_copy_expert(
        ggml_backend_sched_t sched, struct ggml_moe_slot_state * st,
        ggml_backend_t bk, int32_t e, int32_t v, bool is_pf) {
    for (int m = 0; m < 3; m++) {
        if (st->cache.src[m] == NULL) {
            continue;
        }
        const size_t nb  = st->expert_nb[m];
        const char * src = ggml_backend_sched_moe_host_ptr(st, m, e);
        if (!st->cache.pinned) {
            src = ggml_backend_sched_moe_stage(sched, bk, src, nb);
            st->n_stage++;
        }
        const int64_t t0 = sched->moe_stats ? ggml_time_us() : 0;
        bk->iface.set_tensor_async_stream(bk, GGML_SCHED_MOE_WORKER_STREAM,
                st->cache.slots[m], src, (size_t) v * nb, nb);
        if (sched->moe_stats) {
            ggml_backend_sched_moe_tick(sched, GGML_SCHED_MOE_W_ISSUE, t0);
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

// Write one loc_map row. Every change the worker makes goes out this way: a residency change is a
// single expert, and a row write is far cheaper than the whole table.
static void ggml_backend_sched_moe_publish_row(
        ggml_backend_sched_t sched, struct ggml_moe_slot_state * st, ggml_backend_t bk, int32_t e) {
    if (st->pub_pass[e] == sched->moe_pass) {
        // an earlier row write for this expert may not have read the mirror yet
        ggml_backend_sched_moe_sync(sched, bk);
    }
    st->loc_pub[e]  = st->slot_of_expert[e] >= 0 ? st->slot_of_expert[e] : st->n_slots + e;
    st->pub_pass[e] = sched->moe_pass;

    const int64_t t0 = sched->moe_stats ? ggml_time_us() : 0;
    bk->iface.set_tensor_async_stream(bk, GGML_SCHED_MOE_WORKER_STREAM,
            st->cache.loc_map, &st->loc_pub[e], (size_t) e * sizeof(int32_t), sizeof(int32_t));
    if (sched->moe_stats) {
        ggml_backend_sched_moe_tick(sched, GGML_SCHED_MOE_W_ISSUE, t0);
    }
}

// Top up the pool of the layer two gates back.
//
// Two is the conservative bound for "its MoE is certainly done", and it is also the layer whose
// next read is furthest away - a whole token. That distance is what makes the invalidate certain to
// have landed before a prefetch payload can claim the slot, so retire there and nowhere else. Over
// a token this visits every layer exactly once.
static void ggml_backend_sched_moe_retire(ggml_backend_sched_t sched, int gate_slot) {
    const int n = sched->n_moe_caches;
    if (n < 3) {
        return;
    }
    struct ggml_moe_slot_state * st = sched->moe_caches[(gate_slot - 2 + n) % n];
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    while (st->free_cnt < GGML_SCHED_MOE_POOL) {
        const int32_t v = ggml_backend_sched_moe_victim(st, /* same_tok = */ true);
        if (v < 0) {
            break;
        }
        const int32_t e = st->expert_of_slot[v];
        st->slot_of_expert[e]  = -1;
        st->expert_of_slot[v]  = -1;
        st->n_retire++;
        ggml_backend_sched_moe_publish_row(sched, st, bk, e);

        // stamp after the row write, since publishing can sync and move the pass on
        st->free_slot[st->free_cnt++] = v;
        st->free_pass[v]              = sched->moe_pass;
    }
}

// Prefetch the experts the predicting layer guessed for this one.
//
// The target layer has not passed its gate, so it may resolve any resident slot at any moment. The
// only slot with no possible reader is one in the free pool. Out of pool means skip: that expert
// then shows up as a miss on the target's own gate, which is correct, just not hidden.
static void ggml_backend_sched_moe_prefetch(
        ggml_backend_sched_t sched, struct ggml_moe_slot_state * st,
        const int32_t * experts, int n_experts) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    for (int i = 0; i < n_experts; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= st->n_expert || st->slot_of_expert[e] >= 0) {
            continue;
        }
        st->n_pf_try++;
        const int32_t v = ggml_backend_sched_moe_pool_take(sched, st);
        if (v < 0) {
            st->n_pf_dry++;
            continue;
        }

        ggml_backend_sched_moe_copy_expert(sched, st, bk, e, v, /* is_pf = */ true);
        st->n_load_pf++;
        // the payload must land before any reader can be sent here, so order the row after it
        ggml_backend_sched_moe_publish_row(sched, st, bk, e);
    }
}

// Service one mailbox entry: update the LFU, fill the misses, release the gate, then prefetch.
static void ggml_backend_sched_moe_service(
        ggml_backend_sched_t sched, struct ggml_moe_slot_state * st,
        const struct ggml_moe_gate_entry * box) {
    const int32_t * ids    = box->ids;
    const int       n_ids  = std::min<int>(box->n_ids, st->n_expert_used);
    const int       n_pred = std::min<int>(box->n_pred, GGML_SCHED_MAX_MOE_PRED);
    const uint32_t  seq    = box->seq;

    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    const bool probe = (seq == sched->moe_probe_seq);

    const int64_t t_park = sched->moe_stats ? ggml_time_us() : 0;
    if (sched->moe_stats) {
        sched->w_stage_us = 0;
        sched->w_issue_us = 0;
        sched->w_sync_us  = 0;
    }

    st->n_tokens++;

    if (st->n_pred_batch > 0) {
        if (sched->moe_stats) {
            st->n_pf_pred += (uint64_t) st->n_pred_batch;
            for (int j = 0; j < st->n_pred_batch; j++) {
                for (int i = 0; i < n_ids; i++) {
                    if (ids[i] == st->pred_batch[j]) {
                        st->n_pf_useful++;
                        break;
                    }
                }
            }
        }
        st->n_pred_batch = 0;
    }

    const uint32_t lag_window = (uint32_t) (st->n_slots / (st->n_expert_used > 0 ? st->n_expert_used : 1));

    // Pin what this layer already has before filling anything. Pinning as the loop walks the ids would
    // leave the ones behind the cursor open to being evicted by a fill for the ones in front of it.
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

        if (sched->moe_stats) {
            st->n_lag_total++;
            const uint32_t seen = st->last_tok[e];
            if (seen != 0 && st->epoch + 1 - seen <= lag_window) {
                st->n_lag_reach++;
            }
        }

        int32_t v = st->slot_of_expert[e];
        if (v >= 0) {
            // the gate saw it missing but a prefetch has since claimed it: the copy is on this same
            // stream, so the synchronize below already covers it. Do not copy it again.
            st->freq[e]++;
            st->pinned[v] = st->epoch;
            n_hit++;
            continue;
        }

        // The gate read a row for this expert and the host has since dropped it. The gate is then not
        // waiting for anything and will resolve from whatever the row says by the time it looks again.
        if (!probe && i < 32 && (box->miss_mask & (1u << i)) == 0) {
            sched->w_stale++;
        }

        // Prefer the pool, so a fill costs the cache nothing. Out of pool, overwrite in place: this
        // layer is parked in its gate, so no kernel can be reading its slots right now.
        v = ggml_backend_sched_moe_pool_take(sched, st);
        if (v < 0) {
            v = ggml_backend_sched_moe_victim(st, /* same_tok = */ false);
            GGML_ASSERT(v >= 0);
            const int32_t old = st->expert_of_slot[v];
            if (!probe && box->miss_mask == 0) {
                // pinned only covers the ids walked so far, so a later one can still be evicted here.
                // A waiting gate survives that - the loop refills it and the wait covers the refill -
                // so only count it for a gate already free to run its matmuls on this very slot.
                for (int j = i + 1; j < n_ids; j++) {
                    if (ids[j] == old) {
                        sched->w_victim++;
                        break;
                    }
                }
            }
            st->slot_of_expert[old] = -1;
            // the evicted expert must stop pointing here, or the next gate reads the wrong weights
            ggml_backend_sched_moe_publish_row(sched, st, bk, old);
        }

        ggml_backend_sched_moe_copy_expert(sched, st, bk, e, v, /* is_pf = */ false);
        ggml_backend_sched_moe_publish_row(sched, st, bk, e);
        st->n_load_sync++;
        st->freq[e]++;
        st->pinned[v] = st->epoch;
        n_miss++;
    }

    // The gate only waits when it found a miss itself. Anything the worker does here for a gate that
    // is not waiting races with that layer's own matmuls, which are already free to run.
    if (!probe && box->miss_mask == 0 && n_miss > 0) {
        sched->w_unpark++;
    }

    // Everything above went to one stream in order, so this one wait covers the payloads, the row
    // writes, and any prefetch of this layer still in flight.
    //
    // A fill the gate did not ask for still has to be flushed here. The gate is then waiting on the
    // row rather than on the release, and leaving it for the next park would hold it a whole layer.
    if (box->miss_mask != 0 || n_miss > 0) {
        // Hand the release to the device before waiting for it. The first decode graph goes to the
        // driver node by node, so a parked gate stops the device and the host then blocks with the
        // queue full - the wait below can never finish. A release the device reaches on its own
        // breaks that circle. The host still stores it after the wait, for backends with no kernel.
        if (st->n_tokens <= (uint64_t) sched->moe_krel_tok && bk->iface.moe_gate_release_stream != NULL) {
            const int64_t t0 = sched->moe_stats ? ggml_time_us() : 0;
            bk->iface.moe_gate_release_stream(bk, GGML_SCHED_MOE_WORKER_STREAM, st->gate_slot, seq);
            if (sched->moe_stats) {
                ggml_backend_sched_moe_tick(sched, GGML_SCHED_MOE_W_ISSUE, t0);
                sched->w_krel++;
            }
        }

        ggml_backend_sched_moe_sync(sched, bk);
        sched->moe_ch.release[st->gate_slot] = seq;

        // the gate is still parked until this point, so this is exactly what the device waited for.
        // The probe pass is the exception: its gates were released up front, so its wait is only the
        // worker sitting behind a graph nobody is blocked on.
        if (seq == sched->moe_probe_seq) {
            sched->n_probe += (uint64_t) n_miss;
        }
        if (sched->moe_stats && seq != sched->moe_probe_seq) {
            const uint64_t dt = (uint64_t) (ggml_time_us() - t_park);
            sched->w_n++;
            sched->w_tot_us    += dt;
            sched->w_tot_stage += sched->w_stage_us;
            sched->w_tot_issue += sched->w_issue_us;
            sched->w_tot_sync  += sched->w_sync_us;
            sched->w_slow      += dt > 10000 ? 1 : 0;
            if (dt > sched->w_max_us) {
                sched->w_max_us    = dt;
                sched->w_max_stage = sched->w_stage_us;
                sched->w_max_issue = sched->w_issue_us;
                sched->w_max_sync  = sched->w_sync_us;
                sched->w_max_layer = st->cache.layer;   // model layer, to match the timeout report
                sched->w_max_pinned = st->cache.pinned;
                sched->w_max_tok   = st->n_tokens;
                sched->w_max_fill  = n_miss;
            }
        }
    }

    for (int i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && e < st->n_expert) {
            st->last_tok[e] = st->epoch + 1;
        }
    }

    st->n_hit  += n_hit;
    st->n_miss += n_miss;
    st->epoch++;

    if (sched->moe_debug) {
        char buf[512];
        int  off = snprintf(buf, sizeof(buf), "MOE decode t%-4u L%-2d hit=%d load=%d ids=",
                st->epoch, st->cache.layer, n_hit, n_miss);
        for (int i = 0; i < n_ids && off < (int) sizeof(buf) - 24; i++) {
            off += snprintf(buf + off, sizeof(buf) - off, "%d%s", ids[i], i + 1 < n_ids ? "," : "");
        }
        off += snprintf(buf + off, sizeof(buf) - off, " pred=");
        for (int i = 0; i < n_pred && off < (int) sizeof(buf) - 24; i++) {
            off += snprintf(buf + off, sizeof(buf) - off, "%d%s", ids[n_ids + i], i + 1 < n_pred ? "," : "");
        }
        GGML_LOG_INFO("%s\n", buf);
    }

    // The gate is running again from here. Retiring and prefetching are for layers that are not,
    // so they cost this layer nothing.
    ggml_backend_sched_moe_retire(sched, st->gate_slot);

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
            ggml_backend_sched_moe_prefetch(sched, target, target->pred_batch, n_pred);
        }
    }
}

static void ggml_backend_sched_moe_worker(ggml_backend_sched_t sched) {
    struct ggml_moe_gate_entry box;

    while (!sched->moe_worker->stop.load(std::memory_order_relaxed)) {
        bool did_work = false;

        for (int c = 0; c < sched->n_moe_caches; c++) {
            struct ggml_moe_slot_state * st = sched->moe_caches[c];

            const struct ggml_moe_gate_entry * live = &sched->moe_ch.ring[st->gate_slot];

            const uint32_t seq = ((volatile const uint32_t *) &live->seq)[0];
            if (seq == st->last_seen) {
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);

            memcpy(&box, live, sizeof(box));
            std::atomic_thread_fence(std::memory_order_acquire);
            if (((volatile const uint32_t *) &live->seq)[0] != seq) {
                continue;   // the device moved on mid-read, take it next round
            }
            box.seq = seq;

            if (seq > st->last_seen + 1) {
                st->n_lost += seq - st->last_seen - 1;
            }
            st->last_seen = seq;

            sched->moe_worker->idle.store(false, std::memory_order_release);
            ggml_backend_sched_moe_service(sched, st, &box);
            did_work = true;
        }

        if (!did_work) {
            sched->moe_worker->idle.store(true, std::memory_order_release);
            std::this_thread::yield();
        }
    }
}

// Wait until the worker has nothing left in flight, then make its copies visible to the compute
// stream. A Prefill graph reads slot_of_expert and loc_map directly on the main thread, and those
// are the worker's during decode, so a Prefill has to come through here first.
static void ggml_backend_sched_moe_quiesce(ggml_backend_sched_t sched) {
    if (sched->moe_worker == NULL) {
        return;
    }

    while (!sched->moe_worker->idle.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    ggml_backend_sched_moe_sync(sched, sched->backends[sched->moe_caches[0]->dev_backend_id]);
}

// Publish the location table of every dual-base node in this split, the last point before the node
// is enqueued. Decode leaves the table to the worker, so a Prefill must catch up here.
static void ggml_backend_sched_moe_publish_split(
        ggml_backend_sched_t sched, struct ggml_backend_sched_split * split) {
    if (sched->has_moe_gate) {
        return;   // decode, the worker owns the table
    }

    for (int i = 0; i < split->graph.n_nodes; i++) {
        struct ggml_tensor * node = split->graph.nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[4] == NULL) {
            continue;
        }

        for (int c = 0; c < sched->n_moe_caches; c++) {
            struct ggml_moe_slot_state * st = sched->moe_caches[c];
            if (st->cache.loc_map != node->src[4]) {
                continue;
            }
            ggml_backend_sched_moe_publish_loc(st);
            break;
        }
    }
}

// Pay the first use of the worker stream here, on the main thread, before any decode graph is in
// flight. Under WDDM that first use makes the driver build the queue and map the memory, and the
// measured cost of doing it later is the whole rest of the token: once a gate is parked the device
// never goes idle, so the wait only ends when the graph does. Slot 0 is still in the free pool and
// nothing points at it, so the bytes written here do not matter.
static void ggml_backend_sched_moe_warm(ggml_backend_sched_t sched, struct ggml_moe_slot_state * st) {
    ggml_backend_t bk = sched->backends[st->dev_backend_id];

    for (int m = 0; m < 3; m++) {
        if (st->cache.src[m] == NULL) {
            continue;
        }
        const size_t nb  = st->expert_nb[m];
        const char * src = ggml_backend_sched_moe_host_ptr(st, m, 0);
        if (!st->cache.pinned) {
            src = ggml_backend_sched_moe_stage(sched, bk, src, nb);
        }
        bk->iface.set_tensor_async_stream(bk, GGML_SCHED_MOE_WORKER_STREAM, st->cache.slots[m], src, 0, nb);
    }
    ggml_backend_sched_moe_publish_row(sched, st, bk, 0);

    bk->iface.synchronize_stream(bk, GGML_SCHED_MOE_WORKER_STREAM);
    sched->moe_pass++;
}

// Fill the slot caches from what the last prefill ubatch reported it routed to.
//
// Runs once, on the main thread, against an idle device and with no gate parked anywhere. That is
// the whole point: without it the same bytes move during the first decode token, while a gate spins
// on them, and that is where the second-long stalls come from.
//
// The last tokens of the prompt predict the first token of the answer well - a layer reuses about
// four fifths of the experts it used a few tokens back - so most of the first token's misses go.
// The mailbox keeps the last report, so nothing has to be redone per ubatch.
static void ggml_backend_sched_moe_drain_warm(ggml_backend_sched_t sched) {
    if (sched->moe_warm_seq == 0) {
        return;
    }

    ggml_backend_t bk0 = sched->backends[sched->moe_caches[0]->dev_backend_id];

    for (int c = 0; c < sched->n_moe_caches; c++) {
        struct ggml_moe_slot_state * st = sched->moe_caches[c];
        const struct ggml_moe_gate_entry * box = &sched->moe_ch.ring[st->gate_slot];

        // the worker starts right after this, so hand it a clean slate: a prefill report is not a
        // token and must not be serviced as one
        st->last_seen = box->seq;

        if (box->seq != sched->moe_warm_seq || box->n_ids != 0) {
            continue;
        }
        const int n = std::min<int>(box->n_pred, sched->moe_warm_max);
        const uint64_t before = st->n_load_pf;
        ggml_backend_sched_moe_prefetch(sched, st, box->ids, n);
        sched->n_warm += st->n_load_pf - before;
    }

    ggml_backend_sched_moe_sync(sched, bk0);
    sched->moe_warm_seq = 0;
}

// Bump the token counter the gates stamp their entries with, and hand it to the device. One small
// H2D per decode step, on the compute stream, in front of the graph.
//
// Also where the worker starts: by the first decode every cache is registered, so the worker never
// sees the registry grow under it.
static void ggml_backend_sched_moe_new_token(ggml_backend_sched_t sched) {
    if ((!sched->has_moe_gate && !sched->has_moe_warm) || sched->moe_seq_t == NULL) {
        return;
    }

    if (sched->has_moe_gate && sched->moe_worker == NULL) {
        const int64_t t0 = ggml_time_us();

        // the point is to do this against an idle device, so make sure it is one
        ggml_backend_synchronize(sched->backends[sched->moe_caches[0]->dev_backend_id]);

        // a page-locked bank and a pageable one reach the driver differently, so warm one of each
        ggml_backend_sched_moe_warm(sched, sched->moe_caches[0]);
        for (int c = 0; c < sched->n_moe_caches; c++) {
            if (!sched->moe_caches[c]->cache.pinned) {
                ggml_backend_sched_moe_warm(sched, sched->moe_caches[c]);
                break;
            }
        }
        ggml_backend_sched_moe_drain_warm(sched);
        sched->w_warm_us = (uint64_t) (ggml_time_us() - t0);

        // the warm went through the same helpers, so drop what it left in the park counters
        sched->w_call_us = 0;
        sched->w_call_kind = 0;

        sched->moe_worker = new ggml_moe_worker();
        sched->moe_worker->thread = std::thread(ggml_backend_sched_moe_worker, sched);
    }

    sched->moe_seq++;
    const int32_t v = (int32_t) sched->moe_seq;
    ggml_backend_tensor_set(sched->moe_seq_t, &v, 0, sizeof(v));

    if (sched->moe_prerelease) {
        // release every gate before the graph starts, so a miss reports itself and walks straight on
        sched->moe_probe_seq = sched->moe_seq;
        for (int c = 0; c < sched->n_moe_caches; c++) {
            sched->moe_ch.release[sched->moe_caches[c]->gate_slot] = sched->moe_seq;
        }
    }

    if (sched->has_moe_warm) {
        sched->moe_warm_seq = sched->moe_seq;   // this graph's report is the one decode will use
    }
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
    std::vector<uint8_t> verify_buf;   // GGML_SCHED_PREFETCH_VERIFY only

    // ---- weight prefetch (ggml_backend_sched_set_weight_prefetch) ----
    // The entry list must match the one ggml_backend_sched_split_graph built, because that is where
    // the copy nodes were hoisted to. Both are derived from the same predicate over the same splits,
    // so they agree by construction.
    //
    // Schedule for entry k (D = pf_depth = prefetch_slots - 1):
    //   issued  just before the compute of split pf[k - D].split   (entries < D are issued up front)
    //   waited  by the compute of split pf[k].split
    // The copy target's previous owner is some tensor whose work was enqueued before the issue
    // point, so the copy waits on ev_compute_last, which is recorded after every split's compute.
    const int pf_depth = sched->prefetch_slots > 0 ? sched->prefetch_slots - 1 : 0;

    struct pf_entry { int split; int input; };
    std::vector<pf_entry> pf;
    std::vector<int> pf_of_split;     // split -> entry index, -1 if none
    std::vector<int> issue_at_split;  // split -> entry index to issue here, -1 if none
    size_t n_issued = 0;

    if (sched->prefetch_slots > 0 && getenv("GGML_SCHED_PREFETCH_DEBUG")) {
        GGML_LOG_WARN("PFDBG n_splits=%d prefetch_slots=%d\n", sched->n_splits, sched->prefetch_slots);
    }
    if (pf_depth > 0) {
        pf_of_split.assign(sched->n_splits, -1);
        issue_at_split.assign(sched->n_splits, -1);

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * sp = &splits[i];

            for (int j = 0; j < sp->n_inputs; j++) {
                struct ggml_tensor * in     = sp->inputs[j];
                struct ggml_tensor * in_cpy = tensor_copy(in, sp->backend_id, sched->cur_copy);

                if (ggml_backend_sched_is_prefetchable_weight(sp, in, in_cpy)) {
                    pf_of_split[i] = (int) pf.size();
                    pf.push_back({ i, j });
                    break;
                }
            }
        }

        // events are only created for backends that implement event_record / event_wait
        if (!pf.empty() && sched->prefetch_ev_copy[splits[pf[0].split].backend_id][0] == NULL) {
            pf.clear();
            pf_of_split.assign(sched->n_splits, -1);
        }

        for (size_t k = (size_t) pf_depth; k < pf.size(); k++) {
            issue_at_split[pf[k - pf_depth].split] = (int) k;
        }

        // seed ev_compute_last so the first wait does not block on an unrecorded event
        if (!pf.empty()) {
            const int bid = splits[pf[0].split].backend_id;
            ggml_backend_event_record(sched->prefetch_ev_compute_last[bid], sched->backends[bid]);
        }
    }

    // issue the copy of entry k on the auxiliary stream
    auto prefetch_issue = [&](size_t k) {
        const pf_entry & e = pf[k];
        struct ggml_backend_sched_split * sp = &splits[e.split];
        const int slot = (int) (k % (size_t) sched->prefetch_slots);
        const int bid  = sp->backend_id;

        ggml_backend_t bk = sched->backends[bid];

        struct ggml_tensor * in     = sp->inputs[e.input];
        struct ggml_tensor * in_cpy = tensor_copy(in, bid, sched->cur_copy);

        if (getenv("GGML_SCHED_PREFETCH_DEBUG")) {
            GGML_LOG_WARN("PFDBG k=%3zu split=%3d slot=%d in=%-24s in_cpy->data=%p nbytes=%zu\n",
                    k, e.split, slot, in->name, (void *) in_cpy->data, ggml_nbytes(in));
        }

        const int prev_stream = ggml_backend_select_stream(bk, GGML_SCHED_PREFETCH_STREAM_INDEX);

        // report once whether the copies actually landed on a separate stream. without it the
        // prefetch is correct but serialized, i.e. no speedup is possible.
        if (!sched->prefetch_reported) {
            sched->prefetch_reported = true;
            if (prev_stream < 0) {
                GGML_LOG_WARN("%s: backend has no auxiliary stream, weight copies stay on the compute "
                              "stream: correct but NOT concurrent (no speedup expected)\n", __func__);
            } else {
                GGML_LOG_WARN("%s: weight copies issued on stream %d, compute on stream %d\n",
                        __func__, GGML_SCHED_PREFETCH_STREAM_INDEX, prev_stream);
            }
        }

        // the target memory belonged to another tensor until this point in the node order; that
        // tensor's work was enqueued before the issue point, so waiting for the last recorded
        // compute is sufficient to know it is done reading
        ggml_backend_event_wait(bk, sched->prefetch_ev_compute_last[bid]);

        // no dependency on this layer's routing ids, so it can be issued early
        const struct ggml_moe_slot_state * st = ggml_backend_sched_moe_bank_of(sched, in);

        int    ie  = 0;
        size_t off = 0;
        size_t len = 0;
        while (ggml_backend_sched_moe_bank_range(st, in, &ie, &off, &len)) {
            ggml_backend_tensor_set_async(bk, in_cpy, (const char *) in->data + off, off, len);
        }

        ggml_backend_event_record(sched->prefetch_ev_copy[bid][slot], bk);

        if (prev_stream >= 0) {
            ggml_backend_select_stream(bk, prev_stream);
        }
    };

    // the first pf_depth entries have nothing to hide behind, issue them up front
    for (; n_issued < pf.size() && n_issued < (size_t) pf_depth; n_issued++) {
        prefetch_issue(n_issued);
    }

    if (sched->n_moe_caches > 0) {
        if (!sched->has_moe_gate) {
            ggml_backend_sched_moe_quiesce(sched);
        }
        ggml_backend_sched_moe_new_token(sched);
    }

    const bool    timing = sched->moe_stats && sched->n_moe_caches > 0;
    const int64_t t_begin = timing ? ggml_time_us() : 0;
    int64_t t_copy = 0;
    int64_t t_prod = 0;
    int64_t t_fill = 0;
    const bool did_fill = sched->has_moe_gate;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        const int64_t t_copy0 = timing ? ggml_time_us() : 0;

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // this weight was already copied ahead of time on the auxiliary stream: only wait for
                // the copy to land, no device synchronize and no ids round-trip
                if (pf_depth > 0 && pf_of_split[split_id] >= 0 &&
                        pf[pf_of_split[split_id]].input == input_id) {
                    const size_t k    = (size_t) pf_of_split[split_id];
                    const int    slot = (int) (k % (size_t) sched->prefetch_slots);
                    ggml_backend_event_wait(split_backend, sched->prefetch_ev_copy[split_backend_id][slot]);

                    if (sched->prefetch_verify) {
                        // the copy has landed by now; the device content must equal the host source.
                        // a mismatch means the prefetch protocol let something overwrite this buffer.
                        // only the copied ranges hold live data, the rest is left stale on purpose.
                        const size_t nb = ggml_nbytes(input);

                        const struct ggml_moe_slot_state * st =
                            ggml_backend_sched_moe_bank_of(sched, input);

                        ggml_backend_synchronize(split_backend);
                        verify_buf.resize(nb);
                        ggml_backend_tensor_get(input_cpy, verify_buf.data(), 0, nb);

                        int    ie  = 0;
                        size_t off = 0;
                        size_t len = 0;
                        while (ggml_backend_sched_moe_bank_range(st, input, &ie, &off, &len)) {
                            const uint8_t * dev = verify_buf.data() + off;
                            const uint8_t * src = (const uint8_t *) input->data + off;
                            if (memcmp(dev, src, len) == 0) {
                                continue;
                            }

                            size_t i = 0;
                            while (i < len && dev[i] == src[i]) {
                                i++;
                            }
                            GGML_LOG_ERROR("%s: PREFETCH VERIFY FAILED\n", __func__);
                            GGML_LOG_ERROR("  split=%d entry=%zu slot=%d tensor=%s\n",
                                    split_id, k, slot, input->name);
                            GGML_LOG_ERROR("  nbytes=%zu first mismatch at byte %zu (device=0x%02x host=0x%02x)\n",
                                    nb, off + i, dev[i], src[i]);
                            GGML_ABORT("weight prefetch corrupted a copy");
                        }
                    }

                    continue;
                }

                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                const bool moe_classic = split->graph.n_nodes > 0 &&
                    node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy;
                const bool moe_dual = split->graph.n_nodes > 0 &&
                    node->op == GGML_OP_MUL_MAT_ID && node->src[3] == input_cpy && node->src[4] != NULL;

                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (moe_classic || moe_dual)) {

                    // dual Prefill: the ids index the logical experts and loc_map - not this copy -
                    // decides which of them are read here, so the used-experts scan does not apply
                    if (moe_dual) {
                        const struct ggml_moe_slot_state * st =
                            ggml_backend_sched_moe_bank_of(sched, input);

                        int    ie  = 0;
                        size_t off = 0;
                        size_t len = 0;
                        while (ggml_backend_sched_moe_bank_range(st, input, &ie, &off, &len)) {
                            ggml_backend_tensor_set_async(split_backend, input_cpy,
                                    (const char *) input->data + off, off, len);
                        }
                        continue;
                    }

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    // full_weight_copy, set together with the prefetch ring
                    //
                    // Determining which experts are used requires reading the ids tensor back to the host,
                    // which serializes on the router of this very layer. That data dependency makes it
                    // impossible to issue this copy before the previous layer's compute has finished.
                    //
                    // When the batch is large enough that every expert is statistically certain to be
                    // selected, the used-experts scan copies the whole tensor anyway, so skipping it costs
                    // no extra bytes and removes the dependency.
                    //
                    // The n_assign >= n_expert gate is what keeps decode on the used-experts path: with
                    // one token there are only n_expert_used assignments, so copying whole tensors
                    // would move far more bytes than needed.
                    if (sched->full_weight_copy && node->op == GGML_OP_MUL_MAT_ID) {
                        ggml_tensor * ids = node->src[2];

                        // n_tokens * n_expert_used, i.e. the number of (token, expert) assignments
                        const int64_t n_assign = ggml_nelements(ids);

                        if (n_assign >= n_expert) {
                            ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                            continue;
                        }
                    }

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // env: GGML_SCHED_EXPERT_STATS
                    // report how many experts the router actually selected. This decides whether a
                    // full-tensor copy costs extra bytes over the used-experts copy, which is the
                    // premise of removing the ids dependency. Model/prompt property, not hardware.
                    if (sched->expert_stats) {
                        int64_t n_used = 0;
                        for (int64_t e = 0; e < n_expert; ++e) {
                            n_used += ggml_bitset_get(used_ids.data(), e) ? 1 : 0;
                        }
                        GGML_LOG_WARN("EXPERT_STATS n_tokens=%" PRId64 " n_assign=%" PRId64
                                      " n_expert=%" PRId64 " n_used=%" PRId64 " ratio=%.4f\n",
                                ids_tensor->ne[1], ggml_nelements(ids_tensor), n_expert, n_used,
                                (double) n_used / (double) n_expert);
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        const int64_t t_prod0 = timing ? ggml_time_us() : 0;
                        ggml_backend_synchronize(input_backend);
                        if (timing) {
                            t_prod += ggml_time_us() - t_prod0;
                        }
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        // Host->device: prefer the compute stream. A blocking set_tensor uses
                        // cudaStreamPerThread and its Synchronize waits out any earlier H2D on the
                        // copy engine (including MoE slot prefetch), which serializes MoE launch.
                        if (ggml_backend_buffer_is_host(input->buffer) &&
                                split_backend->iface.set_tensor_async != NULL) {
                            ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                        } else {
                            ggml_backend_tensor_copy(input, input_cpy);
                        }
                    }
                }
            }
        }

        if (timing) {
            t_copy += ggml_time_us() - t_copy0;
        }

        // issue this split's scheduled entry before its compute, so the DMA runs alongside it.
        // exactly once per entry: the schedule is precomputed, there is no forward search.
        if (issue_at_split.size() > (size_t) split_id && issue_at_split[split_id] >= 0) {
            const size_t k = (size_t) issue_at_split[split_id];
            GGML_ASSERT(k == n_issued);
            prefetch_issue(k);
            n_issued++;
        }

        // A Prefill dual-base node reads the location table straight from the graph, so it needs the
        // table caught up here. Decode does not come through this: its gate publishes to the worker
        // and the worker writes the table, both without the main thread.
        if (sched->n_moe_caches > 0) {
            const int64_t t_fill0 = timing ? ggml_time_us() : 0;
            ggml_backend_sched_moe_publish_split(sched, split);
            if (timing) {
                t_fill += ggml_time_us() - t_fill0;
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }

        // publish "everything enqueued up to here has been computed", which is what the next
        // prefetch waits on before overwriting memory that used to belong to another tensor
        // only on backends that own prefetch events; CPU-type backends have no event interface
        if (pf_depth > 0 && !pf.empty() && sched->prefetch_ev_compute_last[split_backend_id] != NULL) {
            ggml_backend_event_record(sched->prefetch_ev_compute_last[split_backend_id], split_backend);
        }
    }

    if (timing && did_fill) {
        // the caller synchronizes right after this to read the logits, so the wait is not extra work,
        // only moved a few microseconds earlier
        const int64_t t_drain0 = ggml_time_us();
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
        const int64_t t_end = ggml_time_us();

        sched->t_copy_us  += t_copy;
        sched->t_prod_us  += t_prod;
        sched->t_fill_us  += t_fill;
        sched->t_drain_us += t_end - t_drain0;
        sched->t_step_us  += t_end - t_begin;
        sched->n_step++;
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    // Diagnostics only. The feature itself is turned on through
    // ggml_backend_sched_set_weight_prefetch, so that it is a property of the caller's
    // configuration rather than of the environment the process happens to run in.
    const char * GGML_SCHED_EXPERT_STATS = getenv("GGML_SCHED_EXPERT_STATS");
    sched->expert_stats = GGML_SCHED_EXPERT_STATS ? (atoi(GGML_SCHED_EXPERT_STATS) != 0) : false;

    const char * GGML_SCHED_PREFETCH_VERIFY = getenv("GGML_SCHED_PREFETCH_VERIFY");
    sched->prefetch_verify = GGML_SCHED_PREFETCH_VERIFY ? (atoi(GGML_SCHED_PREFETCH_VERIFY) != 0) : false;

    const char * GGML_SCHED_MOE_SLOT_DEBUG = getenv("GGML_SCHED_MOE_SLOT_DEBUG");
    const char * GGML_SCHED_MOE_SLOT_STATS = getenv("GGML_SCHED_MOE_SLOT_STATS");

    sched->moe_debug = GGML_SCHED_MOE_SLOT_DEBUG ? (atoi(GGML_SCHED_MOE_SLOT_DEBUG) != 0) : false;
    sched->moe_stats = GGML_SCHED_MOE_SLOT_STATS ? (atoi(GGML_SCHED_MOE_SLOT_STATS) != 0) : false;
    sched->moe_pass  = 1;   // pub_pass starts at 0, so no expert looks published yet

    // how many experts per layer the prefill report may warm. Lower it to trade first token latency
    // for a colder cache.
    const char * GGML_SCHED_MOE_WARM = getenv("GGML_SCHED_MOE_WARM");
    sched->moe_warm_max = GGML_SCHED_MOE_WARM ? atoi(GGML_SCHED_MOE_WARM) : GGML_MOE_GATE_MAX_IDS;

    // For how many decode tokens the device releases its own parked gate, instead of the host doing
    // it after waiting for the stream.
    //
    // Off by default. It cannot help the first decode graph: that one goes to the driver node by
    // node, and measurement says nothing the worker gives the device runs until the graph is done -
    // the release kernel included. It only made the deadlock start 14 layers earlier.
    const char * GGML_SCHED_MOE_KREL = getenv("GGML_SCHED_MOE_KREL");
    sched->moe_krel_tok = GGML_SCHED_MOE_KREL ? atoi(GGML_SCHED_MOE_KREL) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }

    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

bool ggml_backend_sched_add_moe_slot_cache(
        ggml_backend_sched_t sched, const struct ggml_moe_slot_cache * cache, int n_expert_used) {
    GGML_ASSERT(sched != NULL);
    GGML_ASSERT(cache != NULL);

    if (sched->n_moe_caches >= GGML_SCHED_MAX_MOE_CACHES) {
        GGML_LOG_WARN("%s: registry full (%d), layer %d not cached\n",
                __func__, GGML_SCHED_MAX_MOE_CACHES, cache->layer);
        return false;
    }

    if (sched->n_moe_caches >= GGML_MOE_GATE_MAX_LAYERS) {
        GGML_LOG_WARN("%s: the gate mailbox holds %d layers, layer %d not cached\n",
                __func__, GGML_MOE_GATE_MAX_LAYERS, cache->layer);
        return false;
    }

    const int n_expert = cache->n_expert;
    const int n_slots  = cache->n_slots;
    if (n_expert <= 0 || n_slots <= 0) {
        GGML_LOG_WARN("%s: layer %d invalid n_expert=%d n_slots=%d\n", __func__, cache->layer, n_expert, n_slots);
        return false;
    }

    size_t nb[3] = { 0, 0, 0 };

    for (int m = 0; m < 3; m++) {
        if (cache->src[m] == NULL) {
            if (cache->slots[m] != NULL) {
                GGML_LOG_WARN("%s: layer %d matrix %d has slots but no src\n", __func__, cache->layer, m);
                return false;
            }
            continue;
        }
        if (cache->slots[m] == NULL) {
            GGML_LOG_WARN("%s: layer %d matrix %d has src but no slots\n", __func__, cache->layer, m);
            return false;
        }
        if (cache->src[m]->ne[0] != cache->slots[m]->ne[0] ||
            cache->src[m]->ne[1] != cache->slots[m]->ne[1] ||
            cache->src[m]->type  != cache->slots[m]->type) {
            GGML_LOG_WARN("%s: layer %d matrix %d shape/type mismatch between src and slots\n",
                    __func__, cache->layer, m);
            return false;
        }
        if (cache->src[m]->buffer == NULL || !ggml_backend_buffer_is_host(cache->src[m]->buffer)) {
            GGML_LOG_WARN("%s: layer %d matrix %d src must live in host memory\n", __func__, cache->layer, m);
            return false;
        }
        if ((int) cache->src[m]->ne[2] != n_expert) {
            GGML_LOG_WARN("%s: layer %d matrix %d host ne[2]=%d expected %d\n",
                    __func__, cache->layer, m, (int) cache->src[m]->ne[2], n_expert);
            return false;
        }
        if ((int) cache->slots[m]->ne[2] != n_slots) {
            GGML_LOG_WARN("%s: layer %d matrix %d slots ne[2]=%d expected %d\n",
                    __func__, cache->layer, m, (int) cache->slots[m]->ne[2], n_slots);
            return false;
        }
        nb[m] = cache->src[m]->nb[2];
    }

    if (nb[0] == 0 && nb[1] == 0 && nb[2] == 0) {
        GGML_LOG_WARN("%s: layer %d has no matrices\n", __func__, cache->layer);
        return false;
    }
    if (n_slots < n_expert_used || n_slots > n_expert) {
        GGML_LOG_WARN("%s: layer %d has %d slots, must be within [%d, %d]\n",
                __func__, cache->layer, n_slots, n_expert_used, n_expert);
        return false;
    }
    if (cache->loc_map == NULL || cache->loc_map->type != GGML_TYPE_I32 ||
            (int) ggml_nelements(cache->loc_map) != n_expert) {
        GGML_LOG_WARN("%s: layer %d loc_map must be I32[%d]\n", __func__, cache->layer, n_expert);
        return false;
    }
    // the pool is what keeps a prefetch off a slot a reader may point at, so the cache has to be
    // wide enough to hold it on top of a full token
    if (n_slots < n_expert_used + GGML_SCHED_MOE_POOL) {
        GGML_LOG_WARN("%s: layer %d has %d slots, needs at least %d\n",
                __func__, cache->layer, n_slots, n_expert_used + GGML_SCHED_MOE_POOL);
        return false;
    }
    if (cache->n_pred > GGML_SCHED_MAX_MOE_PRED) {
        GGML_LOG_WARN("%s: layer %d predicts %d experts, at most %d fit\n",
                __func__, cache->layer, cache->n_pred, GGML_SCHED_MAX_MOE_PRED);
        return false;
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
    st->last_tok       = (uint32_t *) calloc(n_expert, sizeof(uint32_t));
    st->pinned         = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));
    st->free_slot      = (int32_t  *) malloc(sizeof(int32_t)  * n_slots);
    st->free_pass      = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));
    st->fill_seq       = (uint32_t *) calloc(n_slots,  sizeof(uint32_t));

    // Start empty with every slot in the pool. An empty slot outside the pool would be unreachable:
    // the miss path only takes from the pool or evicts a resident, and a retire only turns a
    // resident into a pool entry. The pool drains as the cache warms and settles at the water mark.
    for (int e = 0; e < n_expert; e++) { st->slot_of_expert[e] = -1; }
    for (int v = 0; v < n_slots;  v++) { st->expert_of_slot[v] = -1; st->free_slot[v] = v; }
    st->free_cnt = n_slots;

    st->epoch     = 1;   // 0 means "never pinned", so start above it
    st->loc_dirty = true;
    st->gate_slot = sched->n_moe_caches;
    st->cache.gate_slot = st->gate_slot;   // the graph builder reads it back through find()

    st->dev_backend_id = -1;
    for (int b = 0; b < sched->n_backends; b++) {
        if (cache->slots[0] && cache->slots[0]->buffer &&
            ggml_backend_supports_buft(sched->backends[b], cache->slots[0]->buffer->buft)) {
            st->dev_backend_id = b;
            break;
        }
    }
    if (st->dev_backend_id < 0) {
        GGML_LOG_WARN("%s: layer %d slots belong to no registered backend\n", __func__, cache->layer);
        free(st);
        return false;
    }

    ggml_backend_t bk = sched->backends[st->dev_backend_id];
    if (bk->iface.set_tensor_async_stream == NULL || bk->iface.synchronize_stream == NULL ||
            bk->iface.moe_gate_channel == NULL) {
        GGML_LOG_WARN("%s: backend %s has no gate mailbox, layer %d not cached\n",
                __func__, ggml_backend_name(bk), cache->layer);
        free(st);
        return false;
    }
    if (sched->n_moe_caches == 0 && !bk->iface.moe_gate_channel(bk, &sched->moe_ch)) {
        GGML_LOG_WARN("%s: could not allocate the gate mailbox, layer %d not cached\n",
                __func__, cache->layer);
        free(st);
        return false;
    }

    ggml_backend_buffer_type_t buft_pin = ggml_backend_dev_host_buffer_type(bk->device);
    st->pub_buf = buft_pin ? ggml_backend_buft_alloc_buffer(buft_pin, (size_t) n_expert * sizeof(int32_t)) : NULL;
    if (st->pub_buf == NULL) {
        GGML_LOG_WARN("%s: no page-locked host memory for layer %d, not cached\n", __func__, cache->layer);
        free(st);
        return false;
    }
    st->loc_pub  = (int32_t  *) ggml_backend_buffer_get_base(st->pub_buf);
    st->pub_pass = (uint32_t *) calloc(n_expert, sizeof(uint32_t));

    if (!cache->pinned) {
        const size_t need = std::max(std::max(nb[0], nb[1]), nb[2]);
        if (need > sched->moe_stage_sz) {
            // one slice each, so several experts can be in flight before the ring has to wait
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft_pin, need * GGML_SCHED_MOE_STAGE);
            if (buf == NULL) {
                GGML_LOG_WARN("%s: no page-locked staging for layer %d, not cached\n", __func__, cache->layer);
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

    ggml_backend_sched_moe_publish_loc(st);

    sched->moe_caches[sched->n_moe_caches++] = st;

    if (sched->moe_debug || sched->n_moe_caches == 1) {
        GGML_LOG_INFO("%s: layer %2d registered, %d experts -> %d slots, %.2f MiB per matrix slice\n",
                __func__, cache->layer, n_expert, n_slots,
                (double) nb[0] / 1024.0 / 1024.0);
    }
    return true;
}

void ggml_backend_sched_set_moe_gate_seq(ggml_backend_sched_t sched, struct ggml_tensor * seq) {
    GGML_ASSERT(sched != NULL);
    sched->moe_seq_t = seq;
}

const struct ggml_moe_slot_cache * ggml_backend_sched_find_moe_slot_cache(
        ggml_backend_sched_t sched, const struct ggml_tensor * src) {
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

void ggml_backend_sched_set_weight_prefetch(ggml_backend_sched_t sched, int ring_depth) {
    GGML_ASSERT(sched != NULL);
    GGML_ASSERT(sched->galloc != NULL);   // must be called before the first reserve

    if (ring_depth <= 0) {
        sched->prefetch_slots   = 0;
        sched->full_weight_copy = false;
        return;
    }

    if (ring_depth > GGML_SCHED_MAX_PREFETCH_SLOTS) {
        ring_depth = GGML_SCHED_MAX_PREFETCH_SLOTS;
    }
    // A single slot is a data race: the copy for the next split is issued before the current split's
    // compute is launched, so with one slot they would touch the same memory. Depth 2 is correct but
    // leaves the copy stream only one entry ahead, so any jitter stalls the consumer - depth 3
    // measured 1810 t/s against 1574 at depth 2 on an RTX 5060 over an 8188-token prefill, for the
    // cost of 18 MiB. So the ring is raised to the minimum rather than left where the caller put it.
    if (ring_depth < GGML_SCHED_MIN_PREFETCH_SLOTS) {
        GGML_LOG_WARN("%s: ring depth %d is below the minimum of %d, raising it\n",
                __func__, ring_depth, GGML_SCHED_MIN_PREFETCH_SLOTS);
        ring_depth = GGML_SCHED_MIN_PREFETCH_SLOTS;
    }

    // Pipeline parallelism already double buffers the split inputs across graph invocations, and its
    // events use the same slots, so the two schemes cannot both own them.
    if (sched->n_copies > 1) {
        GGML_LOG_WARN("%s: weight prefetch is not compatible with pipeline parallelism, disabling\n", __func__);
        sched->prefetch_slots   = 0;
        sched->full_weight_copy = false;
        return;
    }

    sched->prefetch_slots = ring_depth;

    // The prefetch copies whole expert tensors: reading the routing ids back to pick out the used
    // experts would serialize on the router of the very layer being prefetched, which is exactly the
    // dependency that makes prefetching impossible. So the two go together.
    sched->full_weight_copy = true;

    for (int b = 0; b < sched->n_backends; b++) {
        // Events are mandatory - they carry the slot-filled and slot-free dependencies. select_stream
        // is optional: without it the copies stay on the compute stream, which is still correct
        // (stream order is stronger than event order) but yields no concurrency. That keeps the ring
        // and the event protocol exercisable on backends that have no auxiliary stream.
        ggml_backend_t backend = sched->backends[b];

        if (backend->iface.event_record == NULL || backend->iface.event_wait == NULL) {
            continue;
        }
        for (int i = 0; i < sched->prefetch_slots; i++) {
            if (sched->prefetch_ev_copy[b][i] == NULL) {
                sched->prefetch_ev_copy[b][i] = ggml_backend_event_new(backend->device);
            }
        }
        if (sched->prefetch_ev_compute_last[b] == NULL) {
            sched->prefetch_ev_compute_last[b] = ggml_backend_event_new(backend->device);
        }
    }

    GGML_LOG_INFO("%s: weight prefetch enabled, ring depth = %d, aux stream = %d\n",
            __func__, sched->prefetch_slots, GGML_SCHED_PREFETCH_STREAM_INDEX);
    GGML_LOG_INFO("%s: the used-experts scan is skipped, whole expert tensors are copied\n", __func__);
    GGML_LOG_INFO("%s: this grows the compute buffer by (ring depth - 1) x the largest host-resident "
                  "expert tensor; an out-of-memory here means the ring is too deep\n", __func__);

    if (sched->prefetch_verify) {
        GGML_LOG_WARN("%s: prefetch verification enabled - every copy is read back and compared, "
                      "this is very slow and must be off for performance runs\n", __func__);
    }
}

// Per-layer slot cache accounting, plus a total normalised per token. Printed once at free.
//
// Decode only, by construction: the remap this hangs off is emitted for one-token graphs, so a
// prefill ubatch never reaches the fill. tokens is therefore a count of decode steps.
//
// hit   = P(an activated expert is already in a slot) - the cache hit rate
// load  = experts moved host -> device per token. One expert means its up/gate/down slices
//         together; those are three backend calls but one load.
// _sync = fetched by the blocking fill on the layer that needs it
// _pf   = fetched ahead of use
// pf_prec = how many predicted experts the layer really used - router guess quality, not transfer success
static void ggml_backend_sched_moe_stats_report(ggml_backend_sched_t sched) {
    GGML_LOG_INFO("\n");
    GGML_LOG_INFO("MoE slot cache, decode only: %d layers, one load = one expert (up+gate+down)\n",
            sched->n_moe_caches);
    GGML_LOG_INFO("  %-5s %6s %8s %8s %11s %11s %8s %11s %11s\n",
            "layer", "slots", "tokens", "hit", "load_sync", "load_pf", "pf_prec", "MiB_sync", "MiB_pf");

    uint64_t t_hit = 0, t_use = 0, t_ls = 0, t_lp = 0, t_bs = 0, t_bp = 0, t_pp = 0, t_pu = 0, max_tokens = 0;
    uint64_t t_lr = 0, t_lt = 0;

    for (int i = 0; i < sched->n_moe_caches; i++) {
        const struct ggml_moe_slot_state * st = sched->moe_caches[i];
        if (st->n_tokens == 0) {
            continue;
        }
        const double   tok = (double) st->n_tokens;
        const uint64_t use = st->n_hit + st->n_miss;

        GGML_LOG_INFO("  %-5d %6d %8llu %8.4f %11.3f %11.3f %8.4f %11.1f %11.1f\n",
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
        t_lr  += st->n_lag_reach;  t_lt  += st->n_lag_total;
        max_tokens = std::max(max_tokens, st->n_tokens);
    }

    // the total is per decode token across all cached layers: what one step actually costs
    if (max_tokens > 0) {
        const double tok = (double) max_tokens;
        GGML_LOG_INFO("  %-5s %6s %8llu %8.4f %11.3f %11.3f %8.4f %11.1f %11.1f\n",
                "TOTAL", "", (unsigned long long) max_tokens,
                t_use ? (double) t_hit / (double) t_use : 0.0,
                (double) t_ls / tok, (double) t_lp / tok,
                t_pp ? (double) t_pu / (double) t_pp : 0.0,
                (double) t_bs / 1024.0 / 1024.0, (double) t_bp / 1024.0 / 1024.0);
    }

    if (t_lt > 0) {
        GGML_LOG_INFO("  lag_reach %.4f: of the experts a token uses, this many the layer also used in the\n",
                (double) t_lr / (double) t_lt);
        GGML_LOG_INFO("  previous slots/used tokens. That is the hit rate a cache can reach knowing nothing\n");
        GGML_LOG_INFO("  about the token it serves, i.e. without the per-layer host round trip.\n");
    }

    if (t_pp == 0) {
        GGML_LOG_INFO("  pf_prec is zero: prediction is off (-ncpred 0) or nothing could be chained\n");
    } else {
        GGML_LOG_INFO("  pf_prec = predicted and used, over predicted; ignores transfers and residency\n");
    }
    if (t_lp == 0 && t_pp > 0) {
        GGML_LOG_INFO("  load_pf is zero: every predicted expert was already resident or could not be fetched\n");
    }

    {
        uint64_t t_try = 0, t_dry = 0, t_ret = 0, t_lost = 0, t_to = 0, t_stage = 0;
        int n_page = 0;
        for (int i = 0; i < sched->n_moe_caches; i++) {
            t_try   += sched->moe_caches[i]->n_pf_try;
            t_dry   += sched->moe_caches[i]->n_pf_dry;
            t_ret   += sched->moe_caches[i]->n_retire;
            t_lost  += sched->moe_caches[i]->n_lost;
            t_stage += sched->moe_caches[i]->n_stage;
            n_page  += sched->moe_caches[i]->cache.pinned ? 0 : 1;
        }
        GGML_LOG_INFO("  gate: pool %d/%d slots, prefetch wanted %llu, skipped %llu (%.1f%% pool dry), retired %llu\n",
                GGML_SCHED_MOE_POOL, sched->n_moe_caches > 0 ? sched->moe_caches[0]->n_slots : 0,
                (unsigned long long) t_try, (unsigned long long) t_dry,
                t_try ? 100.0 * (double) t_dry / (double) t_try : 0.0,
                (unsigned long long) t_ret);
        GGML_LOG_INFO("  gate: %d/%d layers have a pageable bank, %llu slices staged through page-locked memory\n",
                n_page, sched->n_moe_caches, (unsigned long long) t_stage);

        // Which layers gave up waiting for an expert. The gate aborts the run when it does, so seeing
        // this printed at all means the abort path was reached and the numbers below are what it saw.
        if (sched->moe_ch.timeout != NULL) {
            char buf[256] = "";
            int  n = 0;
            for (int i = 0; i < sched->n_moe_caches; i++) {
                const int32_t c = sched->moe_ch.timeout[sched->moe_caches[i]->gate_slot];
                if (c <= 0) {
                    continue;
                }
                t_to += (uint64_t) c;
                if (n < (int) sizeof(buf) - 24) {
                    n += snprintf(buf + n, sizeof(buf) - n, " %d%s(%d)", sched->moe_caches[i]->cache.layer,
                            sched->moe_caches[i]->cache.pinned ? "" : "P", c);
                }
            }
            if (t_to > 0) {
                GGML_LOG_ERROR("  gate: %llu give-ups by layer, P = pageable bank:%s\n",
                        (unsigned long long) t_to, buf);
            }
        }
        if (t_lost > 0 || t_to > 0) {
            GGML_LOG_ERROR("  gate: %llu mailbox entries lost, %llu give-ups - both must be zero\n",
                    (unsigned long long) t_lost, (unsigned long long) t_to);
        }
    }

    // What the device waited for during a park, split by the only three things the worker does.
    // The gate gives up after a fixed time, so a park that ran long shows up here as one phase
    // holding almost all of max - that phase is where it deadlocked.
    if (sched->w_n > 0) {
        static const char * kind[] = { "bounce memcpy", "issue copy", "wait stream" };
        const double n = (double) sched->w_n;

        GGML_LOG_INFO("  gate worker, us per park over %llu parks, mean / max:\n",
                (unsigned long long) sched->w_n);
        GGML_LOG_INFO("    park %.0f / %llu = bounce %.0f / %llu + issue %.0f / %llu + wait %.0f / %llu\n",
                (double) sched->w_tot_us / n,    (unsigned long long) sched->w_max_us,
                (double) sched->w_tot_stage / n, (unsigned long long) sched->w_max_stage,
                (double) sched->w_tot_issue / n, (unsigned long long) sched->w_max_issue,
                (double) sched->w_tot_sync / n,  (unsigned long long) sched->w_max_sync);
        GGML_LOG_INFO("    the max column is one single park, so its four numbers add up\n");
        GGML_LOG_INFO("    that park was layer %d%s, token %llu of that layer, filling %d experts\n",
                sched->w_max_layer, sched->w_max_pinned ? "" : "P",
                (unsigned long long) sched->w_max_tok, sched->w_max_fill);
        GGML_LOG_INFO("    %llu parks over 10 ms; slowest single call of the run: %s, %llu us\n",
                (unsigned long long) sched->w_slow, kind[sched->w_call_kind],
                (unsigned long long) sched->w_call_us);
        GGML_LOG_INFO("    warm up front %llu us, placing %llu experts from the prefill report\n",
                (unsigned long long) sched->w_warm_us, (unsigned long long) sched->n_warm);
        GGML_LOG_INFO("    the discovery pass placed %llu experts\n", (unsigned long long) sched->n_probe);
        GGML_LOG_INFO("    races: %llu stale rows, %llu unparked writes, %llu live victims\n",
                (unsigned long long) sched->w_stale, (unsigned long long) sched->w_unpark,
                (unsigned long long) sched->w_victim);
    }

    if (sched->n_step > 0) {
        const double n = (double) sched->n_step;
        const double step = (double) sched->t_step_us / n;
        const double copy = (double) sched->t_copy_us / n;
        const double prod = (double) sched->t_prod_us / n;
        const double fill = (double) sched->t_fill_us / n;
        const double drain = (double) sched->t_drain_us / n;

        GGML_LOG_INFO("  host per decode step (%llu steps, us): step %.0f = wait-input %.0f + move-input %.0f + fill %.0f + rest %.0f\n",
                (unsigned long long) sched->n_step, step, prod, copy - prod, fill,
                step - copy - fill);
        GGML_LOG_INFO("  of which drain %.0f us (%.0f%%): the device is this far behind once every split is enqueued\n",
                drain, 100.0 * drain / step);
    }

    GGML_LOG_INFO("\n");
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    if (sched->moe_worker != NULL) {
        sched->moe_worker->stop.store(true, std::memory_order_relaxed);
        sched->moe_worker->thread.join();
        delete sched->moe_worker;
        sched->moe_worker = NULL;
    }
    if (sched->moe_stats && sched->n_moe_caches > 0) {
        ggml_backend_sched_moe_stats_report(sched);
    }
    for (int i = 0; i < sched->n_moe_caches; i++) {
        struct ggml_moe_slot_state * st = sched->moe_caches[i];
        free(st->slot_of_expert);
        free(st->expert_of_slot);
        free(st->freq);
        free(st->last_tok);
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
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            ggml_backend_event_free(sched->prefetch_ev_copy[b][i]);
        }
        ggml_backend_event_free(sched->prefetch_ev_compute_last[b]);
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    // A freshly split graph reaches the driver node by node - a backend needs one run of it before it
    // can capture. A gate parked in that run deadlocks: the device stops, the host blocks with the
    // queue full, and nothing the worker gives the device runs until the graph is over. Measured at
    // up to 1.2 s.
    //
    // So spend that run on a discovery pass, whose gates report every expert as missing and wait for
    // nothing. Its output is wrong and thrown away; what it is for is the ids it publishes. Fill the
    // cache from those, then run the real pass, which hits everywhere and is a captured graph by now.
    //
    // Every split gives the graph a new uid, and that is exactly what sends a backend back to running
    // it node by node, so discover once per uid. A real pass is then always a captured replay, which
    // is what lets a gate wait as long as it has to without ever deadlocking.
    if (sched->has_moe_gate && sched->moe_probe_uid != graph->uid) {
        sched->moe_probe_uid  = graph->uid;
        sched->moe_prerelease = true;

        const enum ggml_status err = ggml_backend_sched_compute_splits(sched);

        sched->moe_prerelease = false;
        if (err != GGML_STATUS_SUCCESS) {
            return err;
        }

        // the gates publish from the graph, so it has to be done before the worker can be drained
        ggml_backend_sched_synchronize(sched);
        ggml_backend_sched_moe_quiesce(sched);
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
