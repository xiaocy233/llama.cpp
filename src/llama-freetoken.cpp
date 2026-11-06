#include "llama-freetoken.h"

#include "ggml-cpu.h"
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

struct llama_freetoken_cpu::impl {
    std::vector<ggml_freetoken_layer> layers;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable done;
    bool stop = false;
    uint64_t generation = 0;
    size_t pending = 0;
    std::atomic<int32_t> next{0};
    ggml_freetoken_job job{};
    std::vector<float> partial;

    struct scratch {
        std::vector<float> gate, up, hidden;
        std::vector<uint8_t> quant;
    };

    static void matvec(const ggml_tensor * w, int32_t expert, const float * x, float * y,
                       int64_t row, int64_t count, scratch & s) {
        const auto * traits = ggml_get_type_traits_cpu(w->type);
        const auto * qtraits = ggml_get_type_traits_cpu(traits->vec_dot_type);
        s.quant.resize(ggml_row_size(traits->vec_dot_type, w->ne[0]));
        qtraits->from_float(x, s.quant.data(), w->ne[0]);
        const char * data = (const char *) w->data + expert*w->nb[2];
        for (int64_t r = 0; r < count; ++r) {
            traits->vec_dot(w->ne[0], y + r, 0, data + (r + row)*w->nb[1], 0, s.quant.data(), 0, 1);
        }
    }

    void expert(int32_t i, scratch & s) {
        const int32_t e = job.ids[i];
        if (e < 0) {
            return;
        }
        const auto & l = layers[job.layer];
        const bool fused = l.weights[2] == nullptr;
        const ggml_tensor * gate = l.weights[0];
        const ggml_tensor * up   = fused ? l.weights[0] : l.weights[1];
        const ggml_tensor * down = fused ? l.weights[1] : l.weights[2];
        const int64_t n_ff = down->ne[0];
        const int64_t n_embd = down->ne[1];
        GGML_ASSERT(e < gate->ne[2]);
        s.gate.resize(n_ff);
        s.up.resize(n_ff);
        s.hidden.resize(n_ff);
        const float * input = job.input + (i/job.n_used)*gate->ne[0];
        matvec(gate, e, input, s.gate.data(), 0, n_ff, s);
        matvec(up, e, input, s.up.data(), fused ? n_ff : 0, n_ff, s);
        for (int64_t r = 0; r < n_ff; ++r) {
            s.hidden[r] = (s.gate[r]/(1.0f + std::exp(-s.gate[r])))*s.up[r];
        }
        float * output = partial.data() + i*n_embd;
        matvec(down, e, s.hidden.data(), output, 0, n_embd, s);
        for (int64_t r = 0; r < n_embd; ++r) {
            output[r] *= job.gates[i];
        }
    }

    impl(const std::vector<ggml_freetoken_layer> & layers, int32_t n_threads) : layers(layers) {
        if (n_threads < 1 || n_threads > 1024) {
            throw std::invalid_argument("FreeToken requires 1..1024 CPU threads");
        }
        for (const auto & l : layers) {
            if (l.group < 0) {
                continue;
            }
            const bool fused = l.weights[2] == nullptr;
            const auto * gate = l.weights[0];
            const auto * up = fused ? gate : l.weights[1];
            const auto * down = l.weights[fused ? 1 : 2];
            if (!gate || !up || !down || l.weights[3] || gate->ne[0] != down->ne[1] ||
                up->ne[0] != gate->ne[0] || gate->ne[1] != down->ne[0]*(fused ? 2 : 1) ||
                up->ne[1] != gate->ne[1] || up->ne[2] != gate->ne[2] || down->ne[2] != gate->ne[2]) {
                throw std::invalid_argument("invalid FreeToken SwiGLU geometry");
            }
            for (const ggml_tensor * w : l.weights) {
                if (!w) {
                    continue;
                }
                const auto * tr = ggml_get_type_traits_cpu(w->type);
                if (!tr->vec_dot || !ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float) {
                    throw std::invalid_argument("FreeToken CPU kernel does not support this expert type");
                }
            }
        }
        std::vector<int> cores;
#if defined(__linux__)
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
            for (int i = 0; i < CPU_SETSIZE; ++i) {
                if (CPU_ISSET(i, &allowed)) {
                    cores.push_back(i);
                }
            }
        }
#endif
        try {
            for (int32_t i = 0; i < n_threads; ++i) {
                workers.emplace_back([this, i, cores] {
#if defined(__linux__)
                    if (!cores.empty()) {
                        cpu_set_t mask;
                        CPU_ZERO(&mask);
                        CPU_SET(cores[i % cores.size()], &mask);
                        if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
                            std::fprintf(stderr, "FreeToken: could not pin worker %d\n", i);
                        }
                    }
#endif
                    scratch s;
                    uint64_t seen = 0;
                    std::unique_lock<std::mutex> lock(mutex);
                    for (;;) {
                        ready.wait(lock, [&] { return stop || generation != seen; });
                        if (stop) {
                            return;
                        }
                        seen = generation;
                        lock.unlock();
                        int32_t item;
                        while ((item = next.fetch_add(1)) < job.n_tokens*job.n_used) {
                            expert(item, s);
                        }
                        lock.lock();
                        if (--pending == 0) {
                            done.notify_one();
                        }
                    }
                });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    ~impl() { shutdown(); }

    void submit(const ggml_freetoken_job & value) {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(pending == 0);
        job = value;
        const auto & l = layers[job.layer];
        const ggml_tensor * down = l.weights[l.weights[2] ? 2 : 1];
        partial.assign(size_t(job.n_tokens)*job.n_used*down->ne[1], 0.0f);
        next.store(0);
        pending = workers.size();
        ++generation;
        ready.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return pending == 0; });
        const auto & l = layers[job.layer];
        const ggml_tensor * down = l.weights[l.weights[2] ? 2 : 1];
        const int64_t n = down->ne[1];
        std::fill(job.output, job.output + n*job.n_tokens, 0.0f);
        for (int32_t t = 0; t < job.n_tokens; ++t) {
            for (int32_t k = 0; k < job.n_used; ++k) {
                for (int64_t r = 0; r < n; ++r) {
                    job.output[t*n + r] += partial[(t*job.n_used + k)*n + r];
                }
            }
        }
    }
};

llama_freetoken_cpu::llama_freetoken_cpu(const std::vector<ggml_freetoken_layer> & layers, int32_t n_threads) :
    pimpl(new impl(layers, n_threads)) {}
llama_freetoken_cpu::~llama_freetoken_cpu() = default;

ggml_freetoken_cpu llama_freetoken_cpu::callbacks() {
    return {this,
        [](void * p, const ggml_freetoken_job & job) { static_cast<llama_freetoken_cpu *>(p)->pimpl->submit(job); },
        [](void * p) { static_cast<llama_freetoken_cpu *>(p)->pimpl->wait(); },
        [](void * p) { return static_cast<llama_freetoken_cpu *>(p)->profile(); }};
}

double llama_freetoken_cpu::profile() {
    double bytes = 0;
    int64_t elapsed = 0;
    // Sweep distinct experts with the deployed kernels and worker count, beyond a small CPU cache.
    for (size_t il = 0; il < pimpl->layers.size(); ++il) {
        const auto & l = pimpl->layers[il];
        if (l.group < 0) {
            continue;
        }
        const ggml_tensor * down = l.weights[l.weights[2] ? 2 : 1];
        const int32_t count = std::min<int64_t>(l.weights[0]->ne[2], std::max<size_t>(8, pimpl->workers.size()*2));
        std::vector<float> input(l.weights[0]->ne[0], 0.125f), gates(count, 1.0f/count), output(down->ne[1]);
        std::vector<int32_t> ids(count);
        for (int i = 0; i < count; ++i) {
            ids[i] = i;
        }
        const ggml_freetoken_job job{int32_t(il), 1, count, input.data(), gates.data(), ids.data(), output.data()};
        const int64_t start = ggml_time_us();
        pimpl->submit(job);
        pimpl->wait();
        elapsed += ggml_time_us() - start;
        for (const ggml_tensor * w : l.weights) {
            if (w) {
                bytes += double(count)*w->nb[2];
            }
        }
        if (elapsed > 1000000) {
            break;
        }
    }
    if (elapsed <= 0 || bytes <= 0) {
        throw std::runtime_error("FreeToken CPU bandwidth calibration failed");
    }
    return bytes/(elapsed*1000.0); // decimal GB/s
}
