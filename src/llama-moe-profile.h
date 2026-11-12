#pragma once

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

inline uint64_t llama_moe_profile_integer(const nlohmann::json & value, uint64_t maximum) {
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0)) {
        throw std::runtime_error("profile requires nonnegative integers");
    }
    const uint64_t result = value.get<uint64_t>();
    if (result > maximum) {
        throw std::runtime_error("profile integer is out of range");
    }
    return result;
}

inline std::vector<std::vector<int32_t>> llama_moe_profile_parse(
        const nlohmann::json & profile, const std::string & arch, int n_layer, int n_expert, int n_used, int n_embd,
        uint32_t prior_tokens = 0, std::vector<std::vector<uint32_t>> * initial_freq = nullptr) {
    if (n_layer <= 0 || n_expert <= 0 || n_used <= 0 || n_used > n_expert) {
        throw std::runtime_error("profile requires a routed MoE model");
    }
    if (profile.at("format") != "moe-expert-activation-profile" || llama_moe_profile_integer(profile.at("version"), 1) != 1) {
        throw std::runtime_error("unsupported expert profile format/version");
    }
    const auto & model = profile.at("model");
    auto match = [&](const char * key, int expected) {
        if (llama_moe_profile_integer(model.at(key), INT32_MAX) != (uint64_t) expected) {
            throw std::runtime_error(std::string("expert profile model mismatch: ") + key);
        }
    };
    if (model.at("architecture") != arch) {
        throw std::runtime_error("expert profile architecture mismatch");
    }
    match("main_layer_count", n_layer);
    match("expert_count", n_expert);
    match("experts_per_token", n_used);
    match("hidden_size", n_embd);
    const auto & policy = profile.at("ranking_policy");
    const std::string phase = policy.at("phase").get<std::string>();
    if ((phase != "decode" && phase != "prefill") || policy.at("metric") != "activation_count" || policy.at("tie_break") != "expert_id_ascending") {
        throw std::runtime_error("unsupported expert ranking policy");
    }
    const auto & collection = profile.at("collection");
    if (collection.at("expert_scope") != "routed_only" || collection.at("aggregation") != "token_weighted") {
        throw std::runtime_error("unsupported expert profile counting semantics");
    }
    const auto & layer_ids = model.at("routed_layer_ids");
    const auto & layers = profile.at("layers");
    if (!layers.is_array() || layers.size() != (size_t) n_layer || !layer_ids.is_array() || layer_ids.size() != (size_t) n_layer) {
        throw std::runtime_error("expert profile must cover every main layer");
    }
    std::vector<bool> seen(n_layer, false);
    for (const auto & id : layer_ids) {
        const int il = (int) llama_moe_profile_integer(id, n_layer - 1);
        if (seen[il]) {
            throw std::runtime_error("duplicate routed layer ID");
        }
        seen[il] = true;
    }
    std::vector<std::vector<int32_t>> ranks(n_layer);
    if (initial_freq) {
        initial_freq->assign(n_layer, {});
    }
    for (const auto & layer : layers) {
        const int il = (int) llama_moe_profile_integer(layer.at("layer_id"), n_layer - 1);
        if (!ranks[il].empty()) {
            throw std::runtime_error("duplicate profile layer");
        }
        std::vector<uint64_t> counts(n_expert);
        uint64_t evaluated_tokens = 0;
        for (const std::string key : {"prefill", "decode"}) {
            const auto & stats = layer.at("stats").at(key);
            const uint64_t tokens = llama_moe_profile_integer(stats.at("evaluated_tokens"), std::numeric_limits<uint64_t>::max() / n_expert);
            if (tokens == 0 || tokens != llama_moe_profile_integer(collection.at("tokens_by_phase").at(key), std::numeric_limits<uint64_t>::max() / n_expert)) {
                throw std::runtime_error("inconsistent or empty profile token counts");
            }
            const auto & values = stats.at("activation_counts");
            if (key == phase) {
                evaluated_tokens = tokens;
            }
            if (!values.is_array() || values.size() != (size_t) n_expert) {
                throw std::runtime_error("incorrect expert count array length");
            }
            uint64_t sum = 0;
            for (int e = 0; e < n_expert; ++e) {
                const auto count = llama_moe_profile_integer(values[e], tokens);
                sum += count;
                if (key == phase) {
                    counts[e] = count;
                }
            }
            if (sum != tokens * n_used) {
                throw std::runtime_error("profile activation sum does not equal tokens times top-k");
            }
        }
        if (initial_freq && prior_tokens > 0) {
            auto & freq = initial_freq->at(il);
            freq.resize(n_expert);
            for (int e = 0; e < n_expert; ++e) {
                freq[e] = (uint32_t) std::round((long double) prior_tokens * counts[e] / evaluated_tokens);
            }
        }
        auto & order = ranks[il];
        order.resize(n_expert);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
        });
        const auto & stored = layer.at("ranking").at("expert_ids_desc");
        if (!stored.is_array() || stored.size() != order.size()) {
            throw std::runtime_error("profile requires a complete expert ranking");
        }
        for (int i = 0; i < n_expert; ++i) {
            if (llama_moe_profile_integer(stored[i], n_expert - 1) != (uint64_t) order[i]) {
                throw std::runtime_error("profile ranking disagrees with activation counts");
            }
        }
    }
    return ranks;
}

inline std::vector<std::vector<int32_t>> llama_moe_profile_load(
        const char * path, const std::string & arch, int n_layer, int n_expert, int n_used, int n_embd,
        uint32_t prior_tokens = 0, std::vector<std::vector<uint32_t>> * initial_freq = nullptr) {
    try {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("cannot open file");
        }
        nlohmann::json profile;
        file >> profile;
        return llama_moe_profile_parse(profile, arch, n_layer, n_expert, n_used, n_embd, prior_tokens, initial_freq);
    } catch (const std::exception & err) {
        throw std::runtime_error(std::string("expert profile '") + path + "': " + err.what());
    }
}
