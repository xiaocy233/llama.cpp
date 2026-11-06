#include "freetoken.cuh"
#include "ggml-backend-impl.h"

#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

std::mutex ft_registration_mutex;
std::map<std::pair<uintptr_t, uintptr_t>, size_t> ft_registrations;

struct ft_bank {
    const uint8_t * host;
    uint8_t * device;
    size_t stride;
    size_t cache_stride;
};

struct ft_group_device {
    int32_t n_expert, n_slots;
    int32_t * map;
    int32_t * keys;
    unsigned long long * ages;
    unsigned long long * clock;
    int32_t * ready;
    ggml_freetoken_stats * stats;
};

struct ft_work {
    int32_t n_fill;
    int32_t invalid;
    int32_t * fill_experts;
    int32_t * fill_slots;
    int32_t * seen;
    int32_t * misses;
    int32_t * victims;
};

__device__ bool ft_older(ft_group_device g, int a, int b) {
    return g.ages[a] < g.ages[b] || (g.ages[a] == g.ages[b] && a < b);
}

__device__ void ft_heap_down(ft_group_device g, int * heap, int n, int p) {
    for (;;) {
        int child = 2*p + 1;
        if (child >= n) {
            return;
        }
        if (child + 1 < n && ft_older(g, heap[child], heap[child + 1])) {
            ++child;
        }
        if (!ft_older(g, heap[p], heap[child])) {
            return;
        }
        const int tmp = heap[p]; heap[p] = heap[child]; heap[child] = tmp;
        p = child;
    }
}

// One pass over slots finds all victims; selected hits cannot be evicted.
__global__ void ft_plan(ft_group_device g, ft_work * work, int layer, const int32_t * ids,
                        int count, int n_used, float ratio, bool cpu_only, int32_t * slots, int32_t * cpu_ids) {
    if (threadIdx.x != 0) {
        return;
    }
    ft_work & w = *work;
    w.n_fill = 0;
    w.invalid = 0;
    g.ready[0] = g.ready[1] = -1;
    for (int e = 0; e < g.n_expert; ++e) {
        w.seen[e] = 0;
    }
    for (int i = 0; i < count; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= g.n_expert) {
            ++w.invalid;
        } else {
            w.seen[e] = 1;
        }
    }
    const auto time = atomicAdd(g.clock, 1ULL) + 1;
    int n_miss = 0;
    for (int e = 0; e < g.n_expert; ++e) {
        if (!w.seen[e]) {
            continue;
        }
        const int s = g.map[layer*g.n_expert + e];
        if (s >= 0 && !cpu_only) {
            g.ages[s] = time;
            ++g.stats->hits;
        } else {
            w.misses[n_miss++] = e;
        }
    }
    int q = cpu_only ? 0 : min(n_miss, max(1, int(floorf(n_miss*ratio + 0.5f))));
    int found = 0;
    for (int s = 0; s < g.n_slots && q > 0; ++s) {
        const int key = g.keys[s];
        if (key >= 0 && key/g.n_expert == layer && w.seen[key % g.n_expert]) {
            continue;
        }
        if (found < q) {
            int p = found++;
            w.victims[p] = s;
            while (p > 0) {
                const int parent = (p - 1)/2;
                if (!ft_older(g, w.victims[parent], w.victims[p])) {
                    break;
                }
                const int tmp = w.victims[p]; w.victims[p] = w.victims[parent]; w.victims[parent] = tmp;
                p = parent;
            }
        } else if (ft_older(g, s, w.victims[0])) {
            w.victims[0] = s;
            ft_heap_down(g, w.victims, found, 0);
        }
    }
    q = min(q, found);
    for (int i = 0; i < q; ++i) {
        const int e = w.misses[i];
        const int s = w.victims[i];
        const int old = g.keys[s];
        if (old >= 0) {
            g.map[old] = -1;
        }
        g.map[layer*g.n_expert + e] = s;
        g.keys[s] = layer*g.n_expert + e;
        g.ages[s] = time;
        w.fill_experts[i] = e;
        w.fill_slots[i] = s;
    }
    w.n_fill = q;
    for (int i = 0; i < count; ++i) {
        const int e = ids[i];
        const int s = e >= 0 && e < g.n_expert && !cpu_only ? g.map[layer*g.n_expert + e] : -1;
        // Each route needs a distinct zero expert; native grouped GEMM assumes unique IDs per token.
        slots[i] = s >= 0 ? s : g.n_slots + i % n_used;
        cpu_ids[i] = s >= 0 ? -1 : e;
    }
    g.stats->misses += n_miss;
    g.stats->fills += q;
    g.stats->cpu += n_miss - q;
    g.stats->invalid_ids += w.invalid;
}

// The same work list addresses every bank of a complete expert.
__global__ void ft_fill(const ft_bank * banks, int n_banks, const ft_work * work) {
    const int i = blockIdx.y;
    if (i >= work->n_fill) {
        return;
    }
    for (int b = 0; b < n_banks; ++b) {
        const ft_bank bank = banks[b];
        const auto * src = reinterpret_cast<const uint4 *>(bank.host + work->fill_experts[i]*bank.stride);
        auto * dst = reinterpret_cast<uint4 *>(bank.device + work->fill_slots[i]*bank.cache_stride);
        for (size_t j = blockIdx.x*blockDim.x + threadIdx.x; j < bank.cache_stride/sizeof(uint4); j += gridDim.x*blockDim.x) {
            dst[j] = j < bank.stride/sizeof(uint4) ? src[j] : make_uint4(0, 0, 0, 0);
        }
    }
}

__global__ void ft_full_copy(ft_group_device g, const ft_bank * banks, int n_banks, int layer, int half) {
    if (g.ready[half] == layer) {
        return;
    }
    for (int b = 0; b < n_banks; ++b) {
        const ft_bank bank = banks[b];
        const size_t n = bank.cache_stride*g.n_expert/sizeof(uint4);
        for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
            const size_t e = i/(bank.cache_stride/sizeof(uint4));
            const size_t j = i % (bank.cache_stride/sizeof(uint4));
            const auto * src = reinterpret_cast<const uint4 *>(bank.host + e*bank.stride);
            auto * dst = reinterpret_cast<uint4 *>(bank.device + (half*g.n_expert + e)*bank.cache_stride);
            dst[j] = j < bank.stride/sizeof(uint4) ? src[j] : make_uint4(0, 0, 0, 0);
        }
    }
}

__global__ void ft_full_install(ft_group_device g, int layer, int half, bool prefetch) {
    if (threadIdx.x != 0 || g.ready[half] == layer) {
        return;
    }
    const auto time = atomicAdd(g.clock, 1ULL) + 1;
    for (int e = 0; e < g.n_expert; ++e) {
        const int s = half*g.n_expert + e;
        const int old = g.keys[s];
        if (old >= 0 && g.map[old] == s) {
            g.map[old] = -1;
        }
        const int key = layer*g.n_expert + e;
        const int previous = g.map[key];
        if (previous >= 0 && previous != s) {
            g.keys[previous] = -1;
            g.ages[previous] = 0;
        }
        g.keys[s] = key;
        g.map[key] = s;
        g.ages[s] = time;
    }
    g.ready[half] = layer;
    if (prefetch) {
        ++g.stats->prefetch_layers;
    } else {
        ++g.stats->prefill_layers;
    }
}

__global__ void ft_lookup(ft_group_device g, int layer, const int32_t * ids, int32_t * slots, int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        const int e = ids[i];
        assert(e >= 0 && e < g.n_expert);
        const int s = g.map[layer*g.n_expert + e];
        assert(s >= 0 && s < g.n_slots);
        slots[i] = s;
    }
}

__global__ void ft_add(const float * gpu, const float * cpu, float * out, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = gpu[i] + cpu[i];
    }
}

struct ft_runtime;
struct ft_job {
    ft_runtime * owner = nullptr;
    ggml_freetoken_job cpu{};
    int32_t * cpu_ids_device = nullptr;
    float * output_device = nullptr;
    int32_t * invalid = nullptr;
    void * host = nullptr;
    ~ft_job() {
        if (host) { cudaFreeHost(host); }
        if (cpu_ids_device) { cudaFree(cpu_ids_device); }
        if (output_device) { cudaFree(output_device); }
    }
};

struct ft_group {
    ggml_freetoken_group cfg{};
    ft_group_device device{};
    ft_work * work = nullptr;
    std::vector<int> layers;
};

struct ft_runtime {
    ggml_backend_t backend;
    int device;
    double ratio;
    double pcie_gbps = 0, host_gbps = 0;
    bool pinned = true;
    ggml_freetoken_cpu cpu;
    std::vector<ft_group> groups;
    std::vector<ggml_freetoken_layer> layers;
    std::vector<ft_bank *> banks;
    std::vector<void *> allocations;
    std::vector<std::pair<uintptr_t, uintptr_t>> registrations;
    std::map<std::tuple<int,int,int>, std::unique_ptr<ft_job>> jobs;
    cudaStream_t cpu_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t fork = nullptr, join_cpu = nullptr, join_transfer = nullptr;

    template<typename T> T * allocate(size_t n) {
        if (n > SIZE_MAX/sizeof(T)) {
            throw std::overflow_error("FreeToken allocation overflow");
        }
        T * p = nullptr;
        const auto err = cudaMalloc((void **) &p, n*sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        allocations.push_back(p);
        CUDA_CHECK(cudaMemset(p, 0, n*sizeof(T)));
        return p;
    }

    ft_runtime(ggml_backend_t backend, ggml_freetoken_cpu cpu) : backend(backend), cpu(cpu) {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        device = ctx->device;
        ggml_cuda_set_device(device);
    }

    ~ft_runtime() {
        ggml_cuda_set_device(device);
        ggml_backend_synchronize(backend);
        if (cpu_stream) { cudaStreamSynchronize(cpu_stream); }
        if (transfer_stream) { cudaStreamSynchronize(transfer_stream); }
#ifdef USE_CUDA_GRAPH
        static_cast<ggml_backend_cuda_context *>(backend->context)->cuda_graphs.clear();
#endif
        jobs.clear();
        for (void * p : allocations) { cudaFree(p); }
        {
            std::lock_guard<std::mutex> lock(ft_registration_mutex);
            for (auto range : registrations) {
                auto it = ft_registrations.find(range);
                if (--it->second == 0) {
                    cudaHostUnregister(reinterpret_cast<void *>(range.first));
                    ft_registrations.erase(it);
                }
            }
        }
        if (cpu_stream) { cudaStreamDestroy(cpu_stream); }
        if (transfer_stream) { cudaStreamDestroy(transfer_stream); }
        if (fork) { cudaEventDestroy(fork); }
        if (join_cpu) { cudaEventDestroy(join_cpu); }
        if (join_transfer) { cudaEventDestroy(join_transfer); }
    }

    ft_job * job(int il, int nt, int k) {
        const auto key = std::make_tuple(il, nt, k);
        auto it = jobs.find(key);
        if (it != jobs.end()) {
            return it->second.get();
        }
        const auto & l = layers[il];
        const auto & g = groups[l.group];
        const auto * down = l.weights[g.cfg.n_banks - 1];
        const size_t input_bytes = size_t(nt)*l.weights[0]->ne[0]*sizeof(float);
        const size_t output_bytes = size_t(nt)*down->ne[1]*sizeof(float);
        const size_t route_bytes = size_t(nt)*k*sizeof(int32_t);
        auto j = std::unique_ptr<ft_job>(new ft_job);
        j->owner = this;
        CUDA_CHECK(cudaHostAlloc(&j->host, input_bytes + output_bytes + 2*route_bytes + sizeof(int32_t), cudaHostAllocPortable));
        auto * p = static_cast<char *>(j->host);
        j->cpu = {il, nt, k, (float *) p, (float *) (p + input_bytes),
                  (int32_t *) (p + input_bytes + route_bytes), (float *) (p + input_bytes + 2*route_bytes)};
        j->invalid = (int32_t *) (p + input_bytes + 2*route_bytes + output_bytes);
        CUDA_CHECK(cudaMalloc((void **) &j->cpu_ids_device, route_bytes));
        CUDA_CHECK(cudaMalloc((void **) &j->output_device, output_bytes));
        auto * result = j.get();
        jobs.emplace(key, std::move(j));
        return result;
    }
};

static void CUDART_CB ft_submit(void * data) {
    auto * job = static_cast<ft_job *>(data);
    GGML_ASSERT(*job->invalid == 0);
    job->owner->cpu.submit(job->owner->cpu.userdata, job->cpu);
}

static void CUDART_CB ft_wait(void * data) {
    auto * job = static_cast<ft_job *>(data);
    job->owner->cpu.wait(job->owner->cpu.userdata);
}

// These markers are rejected by CPU execution: graph construction pins them to CUDA.
static void ft_plan_marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken plan requires CUDA"); }
static void ft_merge_marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken merge requires CUDA"); }

struct ft_node {
    ft_runtime * runtime;
    int32_t layer;
    int32_t n_tokens;
    int32_t n_used;
};

// Keep op descriptors with the runtime, not the transient graph tensor.
struct ft_handle : ft_runtime {
    using ft_runtime::ft_runtime;
    std::map<std::tuple<int,int,int>, ft_node> nodes;
};

static void * ft_create(ggml_backend_t backend, const ggml_freetoken_group * config, int32_t ng,
                       const ggml_freetoken_layer * layers, int32_t nl, ggml_freetoken_cpu cpu,
                       double bp, double bh) {
    if (!config || !layers || ng != 1 || nl < 1 || !cpu.submit || !cpu.wait || (bh == 0 && !cpu.profile)) {
        throw std::invalid_argument("invalid FreeToken cache configuration");
    }
    const auto & cfg = config[0];
    if (cfg.n_expert < 1 || cfg.n_slots < 1 || cfg.n_dummy < 1 || cfg.n_dummy > cfg.n_expert ||
        (cfg.n_banks != 2 && cfg.n_banks != 3) || int64_t(nl)*cfg.n_expert > INT32_MAX ||
        int64_t(cfg.n_slots) + cfg.n_dummy + 1 > INT32_MAX) {
        throw std::invalid_argument("invalid FreeToken expert geometry");
    }
    for (int b = 0; b < cfg.n_banks; ++b) {
        if (!cfg.banks[b] || !cfg.banks[b]->data || !cfg.strides[b] || cfg.strides[b] % 16 ||
            uint64_t(cfg.n_slots + cfg.n_dummy + 1) > ggml_nbytes(cfg.banks[b])/cfg.strides[b]) {
            throw std::invalid_argument("invalid FreeToken cache bank");
        }
    }
    for (int il = 0; il < nl; ++il) {
        if (layers[il].group < 0) { continue; }
        if (layers[il].group != 0) { throw std::invalid_argument("invalid FreeToken layer group"); }
        for (int b = 0; b < cfg.n_banks; ++b) {
            const auto * w = layers[il].weights[b];
            if (!w || !ggml_is_contiguous(w) || w->ne[2] != cfg.n_expert || w->ne[3] != 1 || w->nb[2] > cfg.strides[b]) {
                throw std::invalid_argument("invalid FreeToken host bank");
            }
        }
    }
    auto r = std::unique_ptr<ft_handle>(new ft_handle(backend, cpu));
    r->layers.assign(layers, layers + nl);
    r->banks.resize(nl, nullptr);
    r->groups.resize(ng);
    CUDA_CHECK(cudaStreamCreateWithFlags(&r->cpu_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&r->transfer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&r->fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&r->join_cpu, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&r->join_transfer, cudaEventDisableTiming));
    for (int i = 0; i < ng; ++i) {
        auto & g = r->groups[i];
        g.cfg = config[i];
        if (g.cfg.n_slots == 0) {
            continue;
        }
        g.device.n_expert = g.cfg.n_expert;
        g.device.n_slots = g.cfg.n_slots;
        g.device.map = r->allocate<int32_t>(size_t(nl)*g.cfg.n_expert);
        g.device.keys = r->allocate<int32_t>(g.cfg.n_slots);
        g.device.ages = r->allocate<unsigned long long>(g.cfg.n_slots);
        g.device.clock = r->allocate<unsigned long long>(1);
        g.device.ready = r->allocate<int32_t>(2);
        g.device.stats = r->allocate<ggml_freetoken_stats>(1);
        CUDA_CHECK(cudaMemset(g.device.map, 0xff, size_t(nl)*g.cfg.n_expert*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(g.device.keys, 0xff, g.cfg.n_slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(g.device.ready, 0xff, 2*sizeof(int32_t)));
        ft_work w{};
        w.fill_experts = r->allocate<int32_t>(g.cfg.n_expert);
        w.fill_slots = r->allocate<int32_t>(g.cfg.n_expert);
        w.seen = r->allocate<int32_t>(g.cfg.n_expert);
        w.misses = r->allocate<int32_t>(g.cfg.n_expert);
        w.victims = r->allocate<int32_t>(g.cfg.n_expert);
        g.work = r->allocate<ft_work>(1);
        CUDA_CHECK(cudaMemcpy(g.work, &w, sizeof(w), cudaMemcpyHostToDevice));
    }

    // Merge page ranges before registering: adjacent GGUF tensors can share a page.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    std::map<uintptr_t, bool> read_only;
    for (int il = 0; il < nl; ++il) {
        if (layers[il].group < 0) {
            continue;
        }
        auto & g = r->groups[layers[il].group];
        g.layers.push_back(il);
        for (int b = 0; b < g.cfg.n_banks; ++b) {
            auto * w = layers[il].weights[b];
            const uintptr_t begin = reinterpret_cast<uintptr_t>(w->data);
            if (!w->data || begin > UINTPTR_MAX - ggml_nbytes(w) - 4095 || w->nb[2] % 16 || begin % 16) {
                throw std::invalid_argument("FreeToken requires aligned contiguous host experts");
            }
            const uintptr_t page = begin & ~uintptr_t(4095);
            const bool mapped = std::strcmp(ggml_backend_buffer_name(w->buffer), "CPU_Mapped") == 0;
            const auto found = read_only.emplace(page, mapped);
            if (!found.second) { found.first->second &= mapped; }
            ranges.emplace_back(page, (begin + ggml_nbytes(w) + 4095) & ~uintptr_t(4095));
        }
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (auto range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second && read_only[range.first] == read_only[merged.back().first]) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    for (auto range : merged) {
        std::lock_guard<std::mutex> lock(ft_registration_mutex);
        const auto existing = ft_registrations.find(range);
        if (existing != ft_registrations.end()) {
            r->registrations.push_back(range);
            ++existing->second;
            continue;
        }
        void * ptr = reinterpret_cast<void *>(range.first);
        cudaPointerAttributes attr{};
        auto err = cudaPointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess) { (void) cudaGetLastError(); }
        if (err == cudaSuccess && attr.type == cudaMemoryTypeHost) {
            continue;
        }
        // Heap boundary pages can contain writable neighbors; mapped GGUF pages can be read-only.
        const unsigned flags = cudaHostRegisterPortable | cudaHostRegisterMapped | (read_only[range.first] ? cudaHostRegisterReadOnly : 0);
        err = cudaHostRegister(ptr, range.second - range.first, flags);
        if (err != cudaSuccess) {
            GGML_LOG_WARN("FreeToken: host registration failed (%s); using CPU experts\n", cudaGetErrorString(err));
            (void) cudaGetLastError();
            r->pinned = false;
            break;
        }
        ft_registrations[range] = 1;
        r->registrations.push_back(range);
    }
    for (int il = 0; il < nl; ++il) {
        if (layers[il].group < 0) { continue; }
        auto & g = r->groups[layers[il].group];
        std::vector<ft_bank> banks;
        for (int b = 0; b < g.cfg.n_banks; ++b) {
            void * mapped = nullptr;
            if (r->pinned) {
                CUDA_CHECK(cudaHostGetDevicePointer(&mapped, layers[il].weights[b]->data, 0));
            }
            banks.push_back({(const uint8_t *) mapped, (uint8_t *) g.cfg.banks[b]->data, layers[il].weights[b]->nb[2], g.cfg.strides[b]});
        }
        r->banks[il] = r->allocate<ft_bank>(banks.size());
        CUDA_CHECK(cudaMemcpy(r->banks[il], banks.data(), banks.size()*sizeof(ft_bank), cudaMemcpyHostToDevice));
    }
    if (bh == 0) { bh = cpu.profile(cpu.userdata); }
    if (bp == 0 && r->pinned) {
        // Profile the deployed bank gather, including all matrices of each expert.
        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        double bytes = 0, total_ms = 0;
        for (int il = 0; il < nl; ++il) {
            if (layers[il].group < 0) { continue; }
            auto & g = r->groups[layers[il].group];
            const int n = std::min(g.cfg.n_slots, g.cfg.n_expert);
            ft_work host{};
            CUDA_CHECK(cudaMemcpy(&host, g.work, sizeof(host), cudaMemcpyDeviceToHost));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) { ids[i] = i; }
            CUDA_CHECK(cudaMemcpy(host.fill_experts, ids.data(), n*sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(host.fill_slots, ids.data(), n*sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(&g.work->n_fill, &n, sizeof(n), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start, r->transfer_stream));
            ft_fill<<<dim3(32, n), 256, 0, r->transfer_stream>>>(r->banks[il], g.cfg.n_banks, g.work);
            CUDA_CHECK(cudaEventRecord(end, r->transfer_stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
            total_ms += ms;
            for (int b = 0; b < g.cfg.n_banks; ++b) { bytes += n*double(layers[il].weights[b]->nb[2]); }
            if (bytes >= 512*1024*1024) { break; }
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        bp = bytes/(total_ms*1e6);
    }
    if (!std::isfinite(bh) || bh <= 0 || (r->pinned && (!std::isfinite(bp) || bp <= 0))) {
        throw std::invalid_argument("FreeToken bandwidths must be positive finite GB/s");
    }
    r->pcie_gbps = bp;
    r->host_gbps = bh;
    r->ratio = r->pinned ? std::min(1.0, bp/bh) : 0.0;
    GGML_LOG_INFO("FreeToken: B_P=%.3f GB/s B_H=%.3f GB/s fill ratio=%.5f; device LRU and graph CPU branch\n", bp, bh, r->ratio);
    return r.release();
}

static ggml_tensor * ft_build_plan(void * handle, ggml_context * ctx, int32_t il,
                                  ggml_tensor * input, ggml_tensor * ids, ggml_tensor * gates) {
    auto & r = *static_cast<ft_handle *>(handle);
    if (il < 0 || il >= int(r.layers.size()) || r.layers[il].group < 0) { return nullptr; }
    auto & g = r.groups[r.layers[il].group];
    if (ggml_is_empty(ids)) { return nullptr; }
    const int64_t nt = ids->ne[1], k = ids->ne[0];
    if (nt < 1 || k < 1 || nt > 65536 || k > g.cfg.n_expert || k > g.cfg.n_dummy) {
        throw std::invalid_argument("invalid FreeToken routing shape");
    }
    if (nt > 8 && (!r.pinned || g.cfg.n_slots < 2LL*g.cfg.n_expert)) {
        return nullptr; // standard on-demand prefill never allocates two extra layers
    }
    if (ggml_nelements(ids) != nt*k || ggml_nelements(input) != nt*r.layers[il].weights[0]->ne[0] ||
        ggml_nelements(gates) != nt*k || nt*k > INT32_MAX) {
        throw std::invalid_argument("invalid FreeToken input shape");
    }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(gates->type == GGML_TYPE_F32 && ggml_is_contiguous(gates));
    if (nt <= 8 || !r.pinned) { r.job(il, nt, k); }
    auto & node = r.nodes[std::make_tuple(il, nt, k)];
    node = {&r, il, int32_t(nt), int32_t(k)};
    ggml_tensor * args[] = {ids, input, gates};
    return ggml_custom_4d(ctx, GGML_TYPE_I32, k, nt, 1, 1, args, 3, ft_plan_marker, 1, &node);
}

static ggml_tensor * ft_build_merge(void * handle, ggml_context * ctx, int32_t il, ggml_tensor * gpu, ggml_tensor * plan) {
    auto & r = *static_cast<ft_handle *>(handle);
    auto & node = r.nodes.at(std::make_tuple(il, int(plan->ne[1]), int(plan->ne[0])));
    // Keep D2H sources alive until the CPU branch joins.
    ggml_tensor * args[] = {gpu, plan, plan->src[1], plan->src[2]};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, gpu->ne[0], gpu->ne[1], gpu->ne[2], gpu->ne[3], args, 4, ft_merge_marker, 1, &node);
}

static ggml_freetoken_stats ft_stats(void * handle) {
    auto & r = *static_cast<ft_handle *>(handle);
    ggml_backend_synchronize(r.backend);
    ggml_freetoken_stats result{};
    for (auto & g : r.groups) {
        if (!g.cfg.n_slots) { continue; }
        ggml_freetoken_stats s{};
        CUDA_CHECK(cudaMemcpy(&s, g.device.stats, sizeof(s), cudaMemcpyDeviceToHost));
        result.hits += s.hits; result.misses += s.misses; result.fills += s.fills;
        result.cpu += s.cpu; result.prefill_layers += s.prefill_layers;
        result.prefetch_layers += s.prefetch_layers; result.invalid_ids += s.invalid_ids;
    }
    return result;
}

}

bool ggml_cuda_freetoken_is_op(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) { return false; }
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    return p.fun == ft_plan_marker || p.fun == ft_merge_marker;
}

void ggml_cuda_freetoken_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    const auto & node = *static_cast<ft_node *>(p.userdata);
    auto & r = *node.runtime;
    GGML_ASSERT(ctx.device == r.device);
    auto & g = r.groups[r.layers[node.layer].group];
    cudaStream_t main = ctx.stream();
    const bool hybrid = node.n_tokens <= 8 || !r.pinned;
    const int count = node.n_tokens*node.n_used;
    const auto it = std::find(g.layers.begin(), g.layers.end(), node.layer);
    const int rank = int(it - g.layers.begin());
    if (p.fun == ft_plan_marker) {
        if (hybrid) {
            auto * job = r.job(node.layer, node.n_tokens, node.n_used);
            ft_plan<<<1, 1, 0, main>>>(g.device, g.work, node.layer, (int32_t *) op->src[0]->data,
                count, node.n_used, r.ratio, !r.pinned, (int32_t *) op->data, job->cpu_ids_device);
            CUDA_CHECK(cudaEventRecord(r.fork, main));
            CUDA_CHECK(cudaStreamWaitEvent(r.cpu_stream, r.fork, 0));
            CUDA_CHECK(cudaMemcpyAsync((void *) job->cpu.input, op->src[1]->data, ggml_nbytes(op->src[1]), cudaMemcpyDeviceToHost, r.cpu_stream));
            CUDA_CHECK(cudaMemcpyAsync((void *) job->cpu.gates, op->src[2]->data, count*sizeof(float), cudaMemcpyDeviceToHost, r.cpu_stream));
            CUDA_CHECK(cudaMemcpyAsync((void *) job->cpu.ids, job->cpu_ids_device, count*sizeof(int32_t), cudaMemcpyDeviceToHost, r.cpu_stream));
            CUDA_CHECK(cudaMemcpyAsync(job->invalid, &g.work->invalid, sizeof(int32_t), cudaMemcpyDeviceToHost, r.cpu_stream));
            CUDA_CHECK(cudaLaunchHostFunc(r.cpu_stream, ft_submit, job));
            if (r.pinned) {
                ft_fill<<<dim3(32, std::min(count, g.cfg.n_expert)), 256, 0, main>>>(r.banks[node.layer], g.cfg.n_banks, g.work);
            }
        } else {
            const int half = rank % 2;
            ft_full_copy<<<256, 256, 0, main>>>(g.device, r.banks[node.layer], g.cfg.n_banks, node.layer, half);
            ft_full_install<<<1, 1, 0, main>>>(g.device, node.layer, half, false);
            ft_lookup<<<(count + 255)/256, 256, 0, main>>>(g.device, node.layer, (int32_t *) op->src[0]->data, (int32_t *) op->data, count);
            CUDA_CHECK(cudaEventRecord(r.fork, main));
            CUDA_CHECK(cudaStreamWaitEvent(r.transfer_stream, r.fork, 0));
            if (rank + 1 < int(g.layers.size())) {
                const int next = g.layers[rank + 1];
                ft_full_copy<<<256, 256, 0, r.transfer_stream>>>(g.device, r.banks[next], g.cfg.n_banks, next, 1 - half);
                ft_full_install<<<1, 1, 0, r.transfer_stream>>>(g.device, next, 1 - half, true);
            }
        }
    } else {
        GGML_ASSERT(p.fun == ft_merge_marker);
        if (hybrid) {
            auto * job = r.job(node.layer, node.n_tokens, node.n_used);
            CUDA_CHECK(cudaLaunchHostFunc(r.cpu_stream, ft_wait, job));
            CUDA_CHECK(cudaMemcpyAsync(job->output_device, job->cpu.output, ggml_nbytes(op), cudaMemcpyHostToDevice, r.cpu_stream));
            CUDA_CHECK(cudaEventRecord(r.join_cpu, r.cpu_stream));
            CUDA_CHECK(cudaStreamWaitEvent(main, r.join_cpu, 0));
            ft_add<<<(ggml_nelements(op) + 255)/256, 256, 0, main>>>((float *) op->src[0]->data, job->output_device, (float *) op->data, ggml_nelements(op));
        } else {
            CUDA_CHECK(cudaEventRecord(r.join_transfer, r.transfer_stream));
            CUDA_CHECK(cudaStreamWaitEvent(main, r.join_transfer, 0));
            CUDA_CHECK(cudaMemcpyAsync(op->data, op->src[0]->data, ggml_nbytes(op), cudaMemcpyDeviceToDevice, main));
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

const ggml_freetoken_api * ggml_cuda_freetoken_get_api() {
    static const ggml_freetoken_api api{ft_create, [](void * p) { delete static_cast<ft_handle *>(p); }, ft_build_plan, ft_build_merge, ft_stats,
        [](void * p, double * bp, double * bh) {
            const auto & r = *static_cast<ft_handle *>(p);
            *bp = r.pcie_gbps; *bh = r.host_gbps;
        }};
    return &api;
}

#else
bool ggml_cuda_freetoken_is_op(const ggml_tensor *) { return false; }
void ggml_cuda_freetoken_compute(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("FreeToken requires NVIDIA CUDA"); }
const ggml_freetoken_api * ggml_cuda_freetoken_get_api() { return nullptr; }
#endif
