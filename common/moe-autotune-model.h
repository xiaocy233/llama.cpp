#pragma once

#include "ggml-moe-tune.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <map>
#include <string>
#include <vector>

namespace moe_autotune_model {

using tune_json = nlohmann::ordered_json;

struct mean_value {
    int count = 0;
    double value = 0;
    void add(double v) {
        if (!std::isfinite(v) || v < 0) return;
        ++count;
        value += (v - value) / std::min(count, 32);
    }
};

inline double interpolate(const std::array<mean_value, 9> & values, int b, int & estimated) {
    if (b < 1 || b > 8) return -1;
    if (values[b].count) return values[b].value;
    int lo = b - 1, hi = b + 1;
    while (lo > 0 && !values[lo].count) --lo;
    while (hi < 9 && !values[hi].count) ++hi;
    if (lo == 0 || hi == 9) return -1;
    ++estimated;
    return values[lo].value + (values[hi].value - values[lo].value) * (b - lo) / (hi - lo);
}

struct configuration {
    int width = 0;
    float threshold = 0;
    int cpu = 0;
    bool operator==(const configuration & other) const {
        return width == other.width && threshold == other.threshold && cpu == other.cpu;
    }
    tune_json json() const { return {{"npred", width}, {"threshold", threshold}, {"cpu_cap", cpu}}; }
};

struct service_model {
    std::array<mean_value, 9> gpu;
    std::map<std::pair<int, int>, mean_value> joint;
    mean_value transfer, issue, cpu_per_assignment, protocol;
};

inline std::string shape_key(const ggml_moe_tune_layer & l) {
    std::string result = std::to_string(l.width) + ":" + std::to_string(l.top_k);
    for (int i = 0; i < 3; ++i) result += ":" + std::to_string(l.bytes[i]) + ":" + std::to_string(l.types[i]);
    return result;
}

struct round_sample {
    std::vector<float> confidence;
    std::vector<double> draft_prefix_us;
    std::vector<ggml_moe_tune_layer> layers;
    int accepted = 0, kept = 0, round = 0, max_draft = 7;
    double draft_us = 0, verify_us = 0;

    double draft_cost(int kept_prefix) const {
        if (draft_prefix_us.empty()) return draft_us;
        // A rejected candidate still requires one draft forward pass.
        const int executed = std::min(kept_prefix + 1, int(draft_prefix_us.size()));
        return draft_prefix_us[executed - 1] + std::max(0.0, draft_us - draft_prefix_us.back());
    }
};

struct token_sample_schedule {
    uint64_t tokens = 0, next = 0;
    int interval = 20;
    bool due() const { return tokens >= next; }
    void sampled() { next = (tokens / interval + 1) * interval; }
    void advance(int accepted) { GGML_ASSERT(accepted >= 0 && accepted <= 7); tokens += accepted + 1; }
};

inline int prefix_length(const std::vector<float> & confidence, float threshold) {
    int k = 0;
    while (k < int(confidence.size()) && confidence[k] >= threshold) ++k;
    return k;
}

inline std::vector<int> cpu_experts(const std::array<int, 256> & count, const std::array<bool, 256> & resident, int cap) {
    std::vector<int> result;
    if (!cap) return result;
    for (int e = 0; e < 256; ++e) if (count[e] > 0 && count[e] < 9 && !resident[e]) result.push_back(e);
    std::sort(result.begin(), result.end(), [&](int a, int b) { return count[a] != count[b] ? count[a] < count[b] : a < b; });
    if (int(result.size()) > cap) result.resize(cap);
    return result;
}

struct layer_work {
    int miss = 0, prefetch = 0, assignments = 0;
};

struct prepared_layer_work {
    std::array<bool, 256> resident{};
    std::array<int, 256> count{}, votes{};
    std::vector<int> predicted, cpu_predicted, actual, victims;

    prepared_layer_work(const ggml_moe_tune_layer & l, int b, int width) {
        GGML_ASSERT(l.top_k > 0 && l.n_ids % l.top_k == 0 && b > 0 && b <= l.n_ids / l.top_k);
        for (int i = 0; i < l.slots; ++i) if (l.owners[i] >= 0) resident[l.owners[i]] = true;
        const int source_width = l.n_pred / (l.n_ids / l.top_k);
        for (int t = 0; t < b; ++t) {
            for (int j = 0; j < std::min(width, source_width); ++j) {
                const int e = l.predicted[t * source_width + j];
                if (votes[e]++ == 0) predicted.push_back(e);
            }
            for (int j = 0; j < l.top_k; ++j) ++count[l.ids[t * l.top_k + j]];
        }
        std::stable_sort(predicted.begin(), predicted.end(), [&](int a, int b) { return votes[a] > votes[b]; });
        cpu_predicted = cpu_experts(votes, resident, 8);
        for (int e = 0; e < l.experts; ++e) if (count[e]) actual.push_back(e);
        std::stable_sort(actual.begin(), actual.end(), [&](int a, int b) { return count[a] < count[b]; });
        if (!predicted.empty()) {
            for (int i = 0; i < l.slots; ++i) if (l.prefetchable[i]) victims.push_back(l.owners[i]);
            std::stable_sort(victims.begin(), victims.end(), [&](int a, int b) {
                if (a < 0 || b < 0) return a < 0 && b >= 0;
                return l.freq[a] < l.freq[b];
            });
        }
    }

    layer_work evaluate(int cpu_cap) const {
        auto cached = resident;
        std::array<bool, 256> protect{}, skip{};
        for (int i = 0; i < std::min(cpu_cap, int(cpu_predicted.size())); ++i) skip[cpu_predicted[i]] = true;
        for (int e : predicted) if (!skip[e]) protect[e] = true;
        layer_work work;
        size_t next = 0;
        for (int e : predicted) {
            if (skip[e] || cached[e]) continue;
            while (next < victims.size() && victims[next] >= 0 && protect[victims[next]]) ++next;
            if (next == victims.size()) break;
            const int old = victims[next++];
            if (old >= 0) cached[old] = false;
            cached[e] = true;
            ++work.prefetch;
        }
        int assigned = 0;
        for (int e : actual) {
            if (cached[e]) continue;
            if (count[e] < 9 && assigned < cpu_cap) { work.assignments += count[e]; ++assigned; }
            else ++work.miss;
        }
        return work;
    }
};

inline layer_work estimate_work(const ggml_moe_tune_layer & l, int b, configuration config) {
    return prepared_layer_work(l, b, config.width).evaluate(config.cpu);
}

struct latency_model {
    std::map<std::string, service_model> costs;
    std::array<mean_value, 9> base{};
    static tune_json save_value(const mean_value & m) { return {{"samples", m.count}, {"us", m.value}}; }
    static mean_value load_value(const tune_json & j) {
        mean_value m{j.at("samples").get<int>(), j.at("us").get<double>()};
        if (m.count < 0 || m.count > 100000000 || !std::isfinite(m.value) || m.value < 0) throw std::runtime_error("invalid cost profile value");
        return m;
    }

    tune_json json(const std::string & signature = "") const {
        tune_json result = {{"version", 1}, {"signature", signature}, {"base", tune_json::array()}, {"shapes", tune_json::object()}};
        for (int b = 1; b <= 8; ++b) result["base"].push_back(save_value(base[b]));
        for (const auto & entry : costs) {
            const auto & c = entry.second;
            tune_json p = {{"transfer", save_value(c.transfer)}, {"issue", save_value(c.issue)},
                {"cpu_per_assignment", save_value(c.cpu_per_assignment)}, {"protocol", save_value(c.protocol)},
                {"gpu", tune_json::array()}, {"joint", tune_json::array()}};
            for (int b = 1; b <= 8; ++b) p["gpu"].push_back(save_value(c.gpu[b]));
            for (const auto & t : c.joint) p["joint"].push_back({{"batch", t.first.first}, {"assignments", t.first.second}, {"time", save_value(t.second)}});
            result["shapes"][entry.first] = p;
        }
        return result;
    }

    void restore(const tune_json & j) {
        if (j.at("version") != 1) throw std::runtime_error("unsupported cost profile version");
        if (j.at("base").size() != 8 || j.at("shapes").size() > 128) throw std::runtime_error("invalid cost profile dimensions");
        std::map<std::string, service_model> restored;
        std::array<mean_value, 9> restored_base;
        for (int b = 1; b <= 8; ++b) restored_base[b] = load_value(j.at("base").at(b - 1));
        for (auto it = j.at("shapes").begin(); it != j.at("shapes").end(); ++it) {
            auto & c = restored[it.key()];
            const auto & p = it.value();
            c.transfer = load_value(p.at("transfer")); c.issue = load_value(p.at("issue"));
            c.cpu_per_assignment = load_value(p.at("cpu_per_assignment")); c.protocol = load_value(p.at("protocol"));
            if (p.at("gpu").size() != 8 || p.at("joint").size() > 2048) throw std::runtime_error("invalid service table");
            for (int b = 1; b <= 8; ++b) c.gpu[b] = load_value(p.at("gpu").at(b - 1));
            for (const auto & t : p.at("joint")) {
                const int b = t.at("batch"), a = t.at("assignments");
                if (b < 1 || b > 8 || a < 1 || a > 256) throw std::runtime_error("invalid service table index");
                c.joint[{b, a}] = load_value(t.at("time"));
            }
        }
        costs = std::move(restored); base = restored_base;
    }
    using predictions_by_cpu = std::array<std::pair<double, int>, 4>;

    predictions_by_cpu predict_round(const round_sample & sample, int b, int width) const {
        predictions_by_cpu result;
        int estimated = 0;
        const double other = interpolate(base, b, estimated);
        for (auto & p : result) p = {other < 0 ? -1 : sample.draft_cost(b - 1) + other, estimated};
        if (other < 0) return result;
        const double gap = other / std::max<size_t>(1, sample.layers.size());
        std::array<double, 4> overlap{gap, gap, gap, gap};
        const int caps[] = {0, 2, 4, 8};
        for (const auto & l : sample.layers) {
            const auto it = costs.find(shape_key(l));
            estimated = 0;
            const double gpu = it == costs.end() ? -1 : interpolate(it->second.gpu, b, estimated);
            if (width > l.n_pred / (l.n_ids / l.top_k) || gpu < 0 || !it->second.transfer.count) {
                for (auto & p : result) p.first = -1;
                return result;
            }
            const auto & c = it->second;
            const prepared_layer_work prepared(l, b, width);
            for (int i = 0; i < 4; ++i) {
                auto & p = result[i];
                if (p.first < 0) continue;
                p.second += estimated;
                const auto work = prepared.evaluate(caps[i]);
                double joint = gpu;
                if (work.assignments) {
                    const auto exact = c.joint.find({b, work.assignments});
                    if (exact != c.joint.end() && exact->second.count >= 3) joint = std::max(gpu, exact->second.value);
                    else {
                        if (!c.cpu_per_assignment.count || !c.protocol.count) { p.first = -1; continue; }
                        joint = std::max(gpu, c.cpu_per_assignment.value * work.assignments) + c.protocol.value;
                        ++p.second;
                    }
                }
                const double copy = std::max(c.transfer.value, c.issue.value);
                p.first += work.miss * copy + std::max(0.0, work.prefetch * copy - overlap[i]) + joint;
                overlap[i] = joint + gap;
            }
        }
        return result;
    }

    tune_json rank(const std::deque<round_sample> & samples, int max_width) const {
        if (samples.size() < 8) throw std::runtime_error("need at least eight sampled rounds with layer records");
        const int64_t started = ggml_time_us();
        tune_json ranking = tune_json::array();
        configuration best{};
        double best_tpot = std::numeric_limits<double>::infinity();
        int valid = 0;
        std::map<std::tuple<int, int, int>, predictions_by_cpu> predictions;
        for (int width : {0, 8, 16, 24}) for (float threshold : {0.0f, 0.4f, 0.6f, 0.8f}) for (int cpu : {0, 2, 4, 8}) {
            if (width > max_width) continue;
            configuration cfg{width, threshold, cpu};
            double time = 0, tokens = 0, weight = 1;
            int estimated = 0;
            bool supported = true;
            for (auto it = samples.rbegin(); it != samples.rend(); ++it) {
                const int b = prefix_length(it->confidence, threshold) + 1;
                // A candidate needs observed routes and acceptance for its entire prefix.
                if (b > it->kept + 1) { supported = false; break; }
                const auto key = std::make_tuple(it->round, b, width);
                auto cell = predictions.find(key);
                if (cell == predictions.end()) {
                    cell = predictions.emplace(key, predict_round(*it, b, width)).first;
                }
                const auto & prediction = cell->second[cpu == 8 ? 3 : cpu / 2];
                const double us = prediction.first;
                estimated += prediction.second;
                if (us < 0) { supported = false; break; }
                const int k = prefix_length(it->confidence, threshold);
                time += weight * us;
                tokens += weight * (1 + std::min(it->accepted, k));
                weight *= 0.95;
            }
            if (!supported) continue;
            const double tpot = time / tokens;
            ++valid;
            ranking.push_back({{"config", cfg.json()}, {"predicted_tpot_ms", tpot / 1000}, {"estimated_service_lookups", estimated}});
            if (tpot < best_tpot) { best_tpot = tpot; best = cfg; }
        }
        const int64_t elapsed = ggml_time_us() - started;
        const int full = std::count_if(samples.begin(), samples.end(), [](const round_sample & s) { return s.kept == s.max_draft; });
        return {{"event", "recommendation"}, {"samples", samples.size()}, {"full_samples", full}, {"supported_candidates", valid},
                {"selected", valid ? best.json() : tune_json(nullptr)}, {"search_ms", elapsed / 1000.0}, {"ranking", ranking}};
    }
};

}
