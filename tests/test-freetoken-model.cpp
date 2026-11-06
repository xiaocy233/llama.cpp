#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    if (!params.freetoken || !params.moe_cache_size) {
        std::fprintf(stderr, "requires --freetoken, --cpu-moe, --moe-cache-mib and a MoE GGUF\n");
        return 1;
    }
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    if (!ctx) { return 1; }
    const auto * vocab = llama_model_get_vocab(init->model());
    auto tokens = common_tokenize(vocab, "The capital of France is Paris. Explain why a cache helps a computer run faster.", true, true);
    auto eval = [&](llama_context * context, llama_token * data, int n, int pos) {
        auto batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = data[i]; batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i == n - 1;
        }
        const int status = llama_decode(context, batch);
        llama_batch_free(batch);
        if (status) { throw std::runtime_error("decode failed"); }
        llama_synchronize(context);
        const auto * logits = llama_get_logits_ith(context, -1);
        return std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
    };
    const int n = tokens.size() - 1;
    eval(ctx, tokens.data(), n, 0);
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, 0, 0));
    if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, 0) != state.size()) { return 2; }
    const auto reference = eval(ctx, tokens.data() + n, 1, n);
    bool matched = true;
    for (size_t bytes : {params.moe_cache_size/2, size_t(0), params.moe_cache_size}) {
        if (!llama_freetoken_resize_cache(ctx, bytes)) { throw std::runtime_error("resize failed"); }
        if (llama_state_seq_set_data_ext(ctx, state.data(), state.size(), 0, 0) != state.size()) { return 3; }
        const auto result = eval(ctx, tokens.data() + n, 1, n);
        auto expected = reference;
        if (bytes == 0) {
            auto cp = common_context_params_to_llama(params);
            cp.freetoken = false; cp.moe_cache_size = 0;
            llama_context_ptr control(llama_init_from_model(init->model(), cp));
            if (!control || llama_state_seq_set_data_ext(control.get(), state.data(), state.size(), 0, 0) != state.size()) { return 4; }
            expected = eval(control.get(), tokens.data() + n, 1, n);
        }
        double error = 0, scale = 0, max_error = 0, path_error = 0, path_scale = 0;
        for (size_t i = 0; i < result.size(); ++i) {
            if (!std::isfinite(result[i])) { throw std::runtime_error("non-finite logit"); }
            path_error += std::pow(result[i] - reference[i], 2);
            path_scale += double(reference[i])*reference[i];
            const double diff = result[i] - expected[i];
            error += diff*diff; scale += double(expected[i])*expected[i];
            max_error = std::max(max_error, std::abs(diff));
        }
        const double relative = std::sqrt(error/std::max(scale, 1e-30));
        std::printf("cache_mib=%zu relative_logit_l2=%g max_logit_delta=%g top1_equal=%d delta_from_initial_path=%g\n", bytes/1024/1024,
            relative, max_error, std::max_element(result.begin(), result.end()) - result.begin() == std::max_element(expected.begin(), expected.end()) - expected.begin(), std::sqrt(path_error/std::max(path_scale, 1e-30)));
        if (relative > 0.01) { matched = false; }
    }
    if (!matched) { throw std::runtime_error("logit mismatch after resize"); }
    std::puts("PASS: cache shrink, release, grow and recurrent/KV restore");
    return 0;
}
