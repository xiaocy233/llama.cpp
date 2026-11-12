#pragma once

// Metal host-side MoE offload support: gate mailbox, host-executed fill pool, host gate barrier.
//
// Why host-side: an experiment (see spin-test/ in the workspace root) showed that a running Metal
// kernel cannot reliably observe CPU writes to shared memory mid-execution (the release direction
// is flaky), while CPU reads of GPU-written shared memory work but with ms-scale jitter. All
// synchronization here therefore happens at command-buffer boundaries, where Metal guarantees
// coherence.
//
// The worker/mailbox protocol itself is the scheduler's (ggml-backend.cpp), shared with CUDA:
// the only difference is who publishes the mailbox entry (a host gate instead of a device kernel)
// and how fills move bytes (pool memcpy / pread instead of H2D on a stream).

#include "ggml-metal-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_moe * ggml_metal_moe_t;

// the state hangs off the backend context and is created on first use
ggml_metal_moe_t ggml_metal_moe_get(ggml_metal_t ctx);
void             ggml_metal_moe_free(ggml_metal_moe_t moe);

// gate mailbox, allocated on first call; false only on allocation failure
bool ggml_metal_moe_gate_channel(ggml_metal_moe_t moe, struct ggml_moe_gate_channel * out);

// decode event form: one shared event per gate slot. The resolving gate kernel publishes the
// mailbox entry; the encoder places encodeWaitForEvent(ev, seq) behind it; the worker fills the
// misses and signals the event. Waits with the event already at [seq] pass in hardware.
//   gate_event:  lazily create one slot's event (NULL only on allocation failure)
//   gate_events: force-create every slot, return the void* array for the shared channel
//   gate_signal: host-side signal entry the shared worker calls through the channel
ggml_metal_event_t ggml_metal_moe_gate_event (ggml_metal_moe_t moe, int slot);
void **            ggml_metal_moe_gate_events(ggml_metal_moe_t moe);
void               ggml_metal_moe_gate_signal (void * event, uint32_t seq);
void               ggml_metal_moe_gate_signal_drained(void * moe, int slot, uint32_t seq);

// worker-stream fills (called on the scheduler's MoE worker thread):
//   fill_mem/file: payload copies, executed by the pool, no ordering vs earlier tasks
//   row_mem:       loc_map row publishes, ordered after every fill issued before them
//   wait:          drain the pool (synchronize_stream for the worker stream)
void ggml_metal_moe_fill_mem (ggml_metal_moe_t moe, void * dst, const void * src, size_t size);
void ggml_metal_moe_fill_file(ggml_metal_moe_t moe, void * dst, int fd, uint64_t off, size_t size);
void ggml_metal_moe_row_mem  (ggml_metal_moe_t moe, void * dst, const void * src, size_t size);
void ggml_metal_moe_wait     (ggml_metal_moe_t moe);

// prefetch-ring fills (called on the main thread with the prefetch stream selected):
//   begin: the ring buffer is about to be reused - the pool tasks must first wait (host-side)
//          until [ev] reaches its current value (i.e. the GPU's last recorded compute is done)
//   end:   arm [ev_copy] so the GPU-side wait on it passes once everything issued so far landed
void ggml_metal_moe_ring_begin(ggml_metal_moe_t moe, ggml_metal_event_t ev);
void ggml_metal_moe_ring_end  (ggml_metal_moe_t moe, ggml_metal_event_t ev_copy);

// run the gate on the host: publish the mailbox entry, wait for the worker to fill the misses,
// resolve ids to slot indices into the gate's output tensor. Returns false on timeout.
// [node] is a GGML_OP_MOE_GATE tensor; all of its tensors must live in shared Metal buffers.
bool ggml_metal_moe_host_gate(ggml_metal_moe_t moe, const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
