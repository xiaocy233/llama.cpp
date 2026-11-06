#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "llama-freetoken.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static ggml_context_ptr context() {
    return ggml_context_ptr(ggml_init({2*1024*1024, nullptr, true}));
}

static ggml_tensor * ffn(ggml_context * ctx, ggml_tensor ** w, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * gates) {
    const int nt = ids->ne[1], k = ids->ne[0], d = x->ne[0];
    x = ggml_reshape_3d(ctx, x, d, 1, nt);
    auto * gate = ggml_mul_mat_id(ctx, w[0], x, ids);
    auto * up = ggml_mul_mat_id(ctx, w[1], x, ids);
    ggml_prec_set_acc(gate, GGML_PREC_F32);
    ggml_prec_set_acc(up, GGML_PREC_F32);
    auto * hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    auto * down = ggml_mul_mat_id(ctx, w[2], hidden, ids);
    ggml_prec_set_acc(down, GGML_PREC_F32);
    auto * y = ggml_mul(ctx, down, ggml_reshape_3d(ctx, gates, 1, k, nt));
    ggml_tensor * result = nullptr;
    for (int i = 0; i < k; ++i) {
        auto * v = ggml_view_2d(ctx, y, d, nt, y->nb[2], i*y->nb[1]);
        result = result ? ggml_add(ctx, result, v) : v;
    }
    return ggml_cont(ctx, result);
}

struct graph {
    ggml_context_ptr ctx = context();
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * gf;
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * gates;
    ggml_tensor * output;
    graph(ggml_backend_t backend, ggml_tensor ** w, int nt, int k, const ggml_freetoken_api * api = nullptr, void * handle = nullptr, int il = 0) {
        input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, w[0]->ne[0], nt);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k, nt);
        gates = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, nt);
        ggml_set_input(input); ggml_set_input(ids); ggml_set_input(gates);
        auto * plan = api ? api->plan(handle, ctx.get(), il, input, ids, gates) : nullptr;
        if (api && !plan) { throw std::runtime_error("unexpected fallback"); }
        output = ffn(ctx.get(), w, input, plan ? plan : ids, gates);
        if (plan) { output = api->merge(handle, ctx.get(), il, output, plan); }
        ggml_set_output(output);
        gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer) { throw std::runtime_error("graph allocation failed"); }
    }
    std::vector<float> run(ggml_backend_t backend, const std::vector<float> & x, const std::vector<int32_t> & route, const std::vector<float> & weight) {
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ids, route.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(gates, weight.data(), 0, ggml_nbytes(gates));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { throw std::runtime_error("compute failed"); }
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
        return result;
    }
};

static void run_case(ggml_backend_t gpu, ggml_backend_t cpu, const ggml_freetoken_api * api, int mode, int slots, double ratio) {
    constexpr int nl = 3, ne = 8, k = 3;
    const bool quant = mode != 0;
    const int d = mode == 2 ? 2560 : 256, h = mode == 2 ? 640 : 256;
    auto host = context(), device = context(), native = context();
    std::vector<ggml_freetoken_layer> layers(nl);
    ggml_freetoken_group group{};
    group.n_expert = ne; group.n_slots = slots; group.n_banks = 3; group.n_dummy = k;
    const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    for (int il = 0; il < nl; ++il) {
        layers[il].group = 0;
        for (int b = 0; b < 3; ++b) {
            const auto type = mode == 2 ? (b == 2 ? (il % 2 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q5_1) : types[il % 2]) : (quant ? types[(il + b) % 3] : GGML_TYPE_F32);
            layers[il].weights[b] = ggml_new_tensor_3d(host.get(), type, b == 2 ? h : d, b == 2 ? d : h, ne);
        }
    }
    ggml_backend_buffer_ptr hb(ggml_backend_alloc_ctx_tensors(host.get(), cpu));
    std::mt19937 rng(193);
    std::uniform_real_distribution<float> uniform(-0.1f, 0.1f);
    for (auto & layer : layers) {
        for (int b = 0; b < 3; ++b) {
            auto * w = layer.weights[b];
            std::vector<float> values(ggml_nelements(w));
            for (auto & v : values) { v = uniform(rng); }
            if (quant) {
                std::vector<uint8_t> data(ggml_nbytes(w));
                ggml_quantize_chunk(w->type, values.data(), data.data(), 0, w->ne[1]*ne, w->ne[0], nullptr);
                ggml_backend_tensor_set(w, data.data(), 0, data.size());
            } else { ggml_backend_tensor_set(w, values.data(), 0, ggml_nbytes(w)); }
        }
    }
    std::vector<std::vector<ggml_tensor *>> native_weights(nl, std::vector<ggml_tensor *>(3));
    for (int il = 0; il < nl; ++il) {
        for (int b = 0; b < 3; ++b) { native_weights[il][b] = ggml_dup_tensor(native.get(), layers[il].weights[b]); }
    }
    ggml_backend_buffer_ptr nb(ggml_backend_alloc_ctx_tensors(native.get(), gpu));
    ggml_backend_buffer_set_usage(nb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il = 0; il < nl; ++il) {
        for (int b = 0; b < 3; ++b) { ggml_backend_tensor_copy(layers[il].weights[b], native_weights[il][b]); }
    }
    std::vector<std::vector<ggml_tensor *>> cached(nl, std::vector<ggml_tensor *>(3));
    for (int b = 0; b < 3; ++b) {
        size_t align = 16, stride = 0;
        for (auto & layer : layers) {
            align = std::lcm(align, ggml_type_size(layer.weights[b]->type));
            stride = std::max(stride, layer.weights[b]->nb[2]);
        }
        stride = (stride + align - 1)/align*align;
        group.strides[b] = stride;
        group.banks[b] = ggml_new_tensor_1d(device.get(), GGML_TYPE_I8, stride*(slots + k + 1));
        for (int il = 0; il < nl; ++il) {
            auto * w = layers[il].weights[b];
            auto * v = ggml_view_3d(device.get(), group.banks[b], w->ne[0], w->ne[1], slots + k, w->nb[1], stride, 0);
            v->type = w->type; v->nb[0] = w->nb[0];
            cached[il][b] = v;
        }
    }
    ggml_backend_buffer_ptr db(ggml_backend_alloc_ctx_tensors(device.get(), gpu));
    ggml_backend_buffer_clear(db.get(), 0);
    ggml_backend_buffer_set_usage(db.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    llama_freetoken_cpu executor(layers, 3);
    void * handle = api->create(gpu, &group, 1, layers.data(), nl, executor.callbacks(), ratio, 1.0);
    {
        auto c = context();
        auto * x = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, d, 0);
        auto * ids = ggml_new_tensor_2d(c.get(), GGML_TYPE_I32, k, 0);
        auto * weights = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, k, 0);
        if (api->plan(handle, c.get(), 0, x, ids, weights)) { throw std::runtime_error("empty output layer must be skipped"); }
    }
    int comparisons = 0;
    double worst = 0, native_worst = 0, split_worst = 0;
    // Reuse the same graph while changing routes, including repeats and full eviction.
    for (int nt : {1, 4, 8, 16, 1}) {
        if (nt > 8 && slots < 2*ne) {
            auto c = context();
            auto * x = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, d, nt);
            auto * ids = ggml_new_tensor_2d(c.get(), GGML_TYPE_I32, k, nt);
            auto * weights = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, k, nt);
            if (api->plan(handle, c.get(), 0, x, ids, weights)) { throw std::runtime_error("small-cache prefill must fall back"); }
            continue;
        }
        std::vector<std::unique_ptr<graph>> actual, reference, native_reference;
        for (int il = 0; il < nl; ++il) {
            actual.emplace_back(new graph(gpu, cached[il].data(), nt, k, api, handle, il));
            reference.emplace_back(new graph(cpu, layers[il].weights, nt, k));
            native_reference.emplace_back(new graph(gpu, native_weights[il].data(), nt, k));
        }
        for (int step = 0; step < 8; ++step) {
            std::vector<float> x(d*nt), gates(k*nt);
            std::vector<int32_t> ids(k*nt);
            for (auto & v : x) { v = uniform(rng)*5; }
            for (int i = 0; i < k*nt; ++i) {
                ids[i] = (step % 3 == 0 ? i % k : i + step) % ne;
                gates[i] = float(i % k + 1)/6;
            }
            for (int il = 0; il < nl; ++il) {
                auto a = actual[il]->run(gpu, x, ids, gates);
                auto b = reference[il]->run(cpu, x, ids, gates);
                if (comparisons == 0) {
                    const auto first = api->stats(handle);
                    const int q = std::min(k, std::max(1, int(std::floor(k*ratio + 0.5))));
                    if (first.hits != 0 || first.misses != k || first.fills != uint64_t(q) || first.cpu != uint64_t(k - q)) {
                        throw std::runtime_error("cold-route bandwidth split mismatch");
                    }
                }
                auto n = native_reference[il]->run(gpu, x, ids, gates);
                auto cpu_gates = gates, gpu_gates = gates;
                std::vector<int32_t> physical(k*nt);
                ggml_backend_tensor_get(actual[il]->output->src[1], physical.data(), 0, physical.size()*sizeof(int32_t));
                for (int i = 0; i < k*nt; ++i) {
                    if (physical[i] < slots) { cpu_gates[i] = 0; } else { gpu_gates[i] = 0; }
                }
                auto part_cpu = reference[il]->run(cpu, x, ids, cpu_gates);
                auto part_gpu = native_reference[il]->run(gpu, x, ids, gpu_gates);
                double native_error = 0, split_error = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    native_error += std::pow(n[i] - b[i], 2);
                    split_error += std::pow(a[i] - part_cpu[i] - part_gpu[i], 2);
                }
                double error = 0, scale = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { throw std::runtime_error("non-finite output"); }
                    error += (a[i] - b[i])*double(a[i] - b[i]);
                    scale += b[i]*double(b[i]);
                }
                const double relative = std::sqrt(error/std::max(1e-30, scale));
                worst = std::max(worst, relative);
                native_worst = std::max(native_worst, std::sqrt(native_error/std::max(scale, 1e-30)));
                const double split_relative = std::sqrt(split_error/std::max(scale, 1e-30));
                split_worst = std::max(split_worst, split_relative);
                if (relative > (quant ? 0.04 : 0.01) || split_relative > (quant ? 0.001 : 0.00001)) {
                    std::vector<float> check(d*nt);
                    const auto cb = executor.callbacks();
                    cb.submit(cb.userdata, {il, nt, k, x.data(), gates.data(), ids.data(), check.data()}); cb.wait(cb.userdata);
                    double ec = 0, nc = 0;
                    for (size_t j = 0; j < b.size(); ++j) { ec += std::pow(check[j] - b[j], 2); nc += b[j]*double(b[j]); }
                    std::vector<int32_t> slots_read(k*nt);
                    ggml_backend_tensor_get(actual[il]->output->src[1], slots_read.data(), 0, slots_read.size()*sizeof(int32_t));
                    std::vector<float> gpu_only(a.size()), expected_cpu(a.size());
                    ggml_backend_tensor_get(actual[il]->output->src[0], gpu_only.data(), 0, gpu_only.size()*sizeof(float));
                    auto cpu_route = ids;
                    for (size_t j = 0; j < ids.size(); ++j) { if (slots_read[j] < slots) { cpu_route[j] = -1; } }
                    cb.submit(cb.userdata, {il, nt, k, x.data(), gates.data(), cpu_route.data(), expected_cpu.data()}); cb.wait(cb.userdata);
                    double cpu_error = 0, gpu_error = 0;
                    for (size_t j = 0; j < a.size(); ++j) {
                        cpu_error += std::pow(a[j] - gpu_only[j] - expected_cpu[j], 2);
                        gpu_error += std::pow(gpu_only[j] + expected_cpu[j] - b[j], 2);
                    }
                    std::fprintf(stderr, "split cpu error=%g gpu error=%g\n", std::sqrt(cpu_error/nc), std::sqrt(gpu_error/nc));
                    std::fprintf(stderr, "cpu direct delta=%g route:", std::sqrt(ec/nc));
                    for (size_t j = 0; j < ids.size(); ++j) { std::fprintf(stderr, " %d:%d", ids[j], slots_read[j]); }
                    std::fprintf(stderr, "\n");
                    for (int bank = 0; bank < 3; ++bank) {
                        std::vector<uint8_t> values(ggml_nbytes(group.banks[bank]));
                        ggml_backend_tensor_get(group.banks[bank], values.data(), 0, values.size());
                        for (int j = 0; j < k*nt; ++j) {
                            if (slots_read[j] >= slots) { continue; }
                            const auto * w = layers[il].weights[bank];
                            const auto * expected = (const uint8_t *) w->data + ids[j]*w->nb[2];
                            size_t diff = 0;
                            for (size_t a = 0; a < w->nb[2]; ++a) { diff += expected[a] != values[slots_read[j]*group.strides[bank] + a]; }
                            std::fprintf(stderr, "bank %d expert %d differing bytes=%zu\n", bank, ids[j], diff);
                        }
                    }
                    std::fprintf(stderr, "mismatch quant=%d slots=%d ratio=%g nt=%d step=%d layer=%d error=%g split=%g native=%g\n", quant, slots, ratio, nt, step, il, relative, split_relative, std::sqrt(native_error/scale));
                    std::abort();
                }
                ++comparisons;
            }
        }
    }
    const auto stats = api->stats(handle);
    if (!stats.fills || stats.invalid_ids || stats.fills + stats.cpu != stats.misses || (ratio < 1 && !stats.cpu) || (slots >= 16 && !stats.prefetch_layers)) {
        throw std::runtime_error("expected execution branches were not covered");
    }
    std::printf("PASS mode=%d slots=%d ratio=%.2f comparisons=%d worst_relative_l2=%g native_relative_l2=%g split_relative_l2=%g hits=%llu fills=%llu cpu=%llu prefetch=%llu\n",
        mode, slots, ratio, comparisons, worst, native_worst, split_worst, (unsigned long long) stats.hits, (unsigned long long) stats.fills,
        (unsigned long long) stats.cpu, (unsigned long long) stats.prefetch_layers);
    void * second = api->create(gpu, &group, 1, layers.data(), nl, executor.callbacks(), 1.0, 1.0);
    api->destroy(handle);
    graph survivor(gpu, cached[0].data(), 1, k, api, second, 0);
    graph control(cpu, layers[0].weights, 1, k);
    const std::vector<float> x(d, 0.125f), gates(k, 1.0f/k);
    const std::vector<int32_t> route{0, 1, 2};
    const auto a = survivor.run(gpu, x, route, gates), b = control.run(cpu, x, route, gates);
    double error = 0, scale = 0;
    for (size_t i = 0; i < a.size(); ++i) { error += std::pow(a[i] - b[i], 2); scale += double(b[i])*b[i]; }
    if (!std::isfinite(error) || error > 0.04*0.04*scale) { throw std::runtime_error("shared host registration lifetime mismatch"); }
    api->destroy(second);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    ggml_backend_ptr gpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!gpu || !cpu) { std::fprintf(stderr, "CUDA device required\n"); return 77; }
    ggml_backend_cpu_set_n_threads(cpu.get(), 3);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
    auto get_api = (ggml_freetoken_get_api_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_freetoken_get_api");
    if (!get_api || !get_api()) { std::fprintf(stderr, "FreeToken CUDA backend required\n"); return 77; }
    for (int mode : {0, 1, 2}) {
        for (int slots : {3, 16}) {
            for (double ratio : {0.25, 1.0}) { run_case(gpu.get(), cpu.get(), get_api(), mode, slots, ratio); }
        }
    }
    return 0;
}
