#pragma once

#include "llama.h"
#include "draft-acceptance-profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

inline std::array<float, 4> common_draft_acceptance_stats(const float * logits, int vocab, llama_token token) {
    DFLASH_PROFILE("dflash.features", vocab);
    if (!logits || vocab < 10 || token < 0 || token >= vocab) {
        throw std::runtime_error("invalid DFlash acceptance logits");
    }
    std::array<float, 10> top;
    std::partial_sort_copy(logits, logits + vocab, top.begin(), top.end(), std::greater<float>());
    double sum = 0.0, weighted = 0.0, sum10 = 0.0;
    for (int v = 0; v < vocab; ++v) {
        const double delta = double(logits[v]) - top[0];
        const double weight = std::exp(delta);
        sum += weight;
        if (weight > 0.0) {
            weighted += weight * delta;
        }
    }
    for (float value : top) {
        sum10 += std::exp(double(value) - top[0]);
    }
    const double logp = double(logits[token]) - top[0] - std::log(sum);
    return {float(logp), top[0] - top[1], float(std::log(sum) - weighted / sum), float(1.0 / sum10)};
}

// Offline DFlash data collection. All features are copied before target verification.
struct common_draft_acceptance_trace {
    FILE * file = nullptr;
    uint32_t hidden_size;
    uint32_t prompt_size = 0;
    uint32_t request_id = 0;
    uint32_t round_id = 0;
    uint32_t position = 0;
    std::vector<llama_token> tokens;
    std::vector<float> stats;
    std::vector<float> hidden;

    common_draft_acceptance_trace(const char * path, llama_context * ctx) :
        hidden_size(llama_model_n_embd_out(llama_get_model(ctx))) {
        if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            throw std::runtime_error("DFlash acceptance trace requires unpooled embeddings");
        }
        file = fopen(path, "wbx");
        if (!file) {
            throw std::runtime_error("cannot create DFlash acceptance trace (path must be new)");
        }
        const uint32_t header[] = {0x31434144, 1, hidden_size, 4};
        if (fwrite(header, sizeof(header), 1, file) != 1) {
            fclose(file);
            file = nullptr;
            throw std::runtime_error("cannot write DFlash acceptance trace header");
        }
    }

    ~common_draft_acceptance_trace() {
        if (file) {
            fclose(file);
        }
    }

    void begin(uint32_t n_prompt) {
        prompt_size = n_prompt;
        ++request_id;
        tokens.clear();
    }

    void capture(llama_context * ctx, int first_row, const std::vector<llama_token> & draft, int limit, uint32_t pos) {
        const size_t count = limit > 0 ? std::min(draft.size(), size_t(limit)) : draft.size();
        tokens.assign(draft.begin(), draft.begin() + count);
        stats.clear();
        hidden.clear();
        position = pos;
        ++round_id;
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        for (size_t j = 0; j < count; ++j) {
            const float * logits = llama_get_logits_ith(ctx, first_row + j);
            const float * embedding = llama_get_embeddings_ith(ctx, first_row + j);
            if (!logits || !embedding || vocab < 10) {
                throw std::runtime_error("missing DFlash acceptance features");
            }
            const auto values = common_draft_acceptance_stats(logits, vocab, tokens[j]);
            stats.insert(stats.end(), values.begin(), values.end());
            hidden.insert(hidden.end(), embedding, embedding + hidden_size);
        }
    }

    void accept(uint32_t accepted) {
        if (tokens.empty()) {
            return;
        }
        if (accepted > tokens.size()) {
            throw std::runtime_error("invalid DFlash acceptance trace label");
        }
        const uint32_t header[] = {position, uint32_t(tokens.size()), accepted, prompt_size, round_id, request_id};
        const bool ok = fwrite(header, sizeof(header), 1, file) == 1 &&
            fwrite(tokens.data(), sizeof(llama_token), tokens.size(), file) == tokens.size() &&
            fwrite(stats.data(), sizeof(float), stats.size(), file) == stats.size() &&
            fwrite(hidden.data(), sizeof(float), hidden.size(), file) == hidden.size() && fflush(file) == 0;
        tokens.clear();
        if (!ok) {
            throw std::runtime_error("cannot write DFlash acceptance trace record");
        }
    }
};
