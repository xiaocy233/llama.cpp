#pragma once

#include "ggml-moe-tune.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <future>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <map>
#include <string>
#include <vector>

namespace moe_autotune_model {

using tune_json = nlohmann::ordered_json;

static constexpr size_t max_workload_rounds = 2048;
static constexpr size_t max_workload_bytes = 128 * 1024 * 1024;

inline int next_execution_phase(int batch, int previous_batch, int previous_phase) {
    return batch == previous_batch ? std::min(2, previous_phase + 1) : 0;
}

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

inline tune_json branch_candidates(const tune_json & ranking, bool complete = true) {
    std::map<float, tune_json> branches;
    for (const auto & row : ranking) {
        const float threshold = row.at("config").at("threshold");
        auto it = branches.find(threshold);
        if (it == branches.end() || row.at("predicted_tpot_ms").get<double>() < it->second.at("predicted_tpot_ms").get<double>()) branches[threshold] = row;
    }
    if (complete && !ranking.empty()) {
        const tune_json * best = &ranking.front();
        for (const auto & row : ranking) if (row.at("predicted_tpot_ms").get<double>() < best->at("predicted_tpot_ms").get<double>()) best = &row;
        for (float threshold : {0.0f, 0.4f, 0.6f, 0.8f}) if (!branches.count(threshold)) {
            auto row = *best;
            row["config"]["threshold"] = threshold;
            row["predicted_tpot_ms"] = nullptr;
            branches[threshold] = std::move(row);
        }
    }
    tune_json result = tune_json::array();
    for (const auto & entry : branches) result.push_back({{"config", entry.second.at("config")}, {"predicted_tpot_ms", entry.second.at("predicted_tpot_ms")}});
    return result;
}

struct candidate_frontier {
    static constexpr double tolerance = 0.03;
    struct candidate {
        configuration config;
        int trials = 0;
        bool refined = false, finalist = false;
        double screening_ms = std::numeric_limits<double>::infinity();
        double confirmation_ms = std::numeric_limits<double>::infinity();
        uint64_t tokens = 0;
        double tpot() const { return screening_ms; }
    };
    std::vector<candidate> candidates;
    std::map<float, double> branch_errors;
    configuration incumbent;
    uint64_t output_tokens = 0, confirmation_tokens = 0;
    int completed = 0, budget = 24;
    size_t current = 0;
    bool confirming = false, warming = false, warmup_enabled = false, confirm_required = true;

    bool ready() const {
        return confirming && std::all_of(candidates.begin(), candidates.end(), [&](const candidate & c) { return !c.finalist || c.trials == (confirm_required ? 2 : 1); });
    }
    void add(configuration config) {
        if (confirming || candidates.size() >= size_t(budget / 2)) return;
        if (std::none_of(candidates.begin(), candidates.end(), [&](const candidate & c) { return c.config == config; })) candidates.push_back({config});
    }
    void seed(const tune_json & shortlist, configuration running, int trial_budget = 24, bool warmup = false, bool confirm = true) {
        *this = {};
        budget = trial_budget;
        incumbent = running;
        warmup_enabled = warming = warmup;
        confirm_required = confirm;
        add(running);
        for (const auto & row : shortlist) {
            const auto & c = row.at("config");
            add({c.at("npred"), c.at("threshold"), c.at("cpu_cap")});
        }
    }
    bool observe(configuration cfg, double elapsed_us, uint64_t tokens) {
        if (candidates.empty() || tokens < 64 || !std::isfinite(elapsed_us) || elapsed_us <= 0 || !(cfg == candidates[current].config)) return false;
        if (warming) { warming = false; return false; }
        auto & expected_tokens = confirming && confirm_required ? confirmation_tokens : output_tokens;
        if (expected_tokens && expected_tokens != tokens) {
            for (auto & c : candidates) c = candidate{c.config};
            completed = 0;
            branch_errors.clear();
            output_tokens = confirmation_tokens = 0;
            confirming = false;
            warming = warmup_enabled;
            current = 0;
            return false;
        }
        if (ready()) return false;
        expected_tokens = tokens;
        ++completed;
        auto & c = candidates[current];
        ++c.trials;
        c.tokens += tokens;
        const double value = elapsed_us / tokens / 1000;
        if (confirming) c.confirmation_ms = value;
        else c.screening_ms = value;
        return true;
    }
    bool needs_refinement(configuration config) const {
        for (const auto & c : candidates) if (c.config == config) return !confirming && !c.refined;
        return false;
    }
    void refine(const tune_json & ranking, configuration executed, double measured) {
        if (!needs_refinement(executed) || ranking.empty()) return;
        const tune_json * reference = nullptr;
        for (const auto & row : ranking) if (row.at("config") == executed.json()) reference = &row;
        if (!reference) return;
        for (auto & c : candidates) if (c.config == executed) c.refined = true;
        const double error = std::abs(reference->at("predicted_tpot_ms").get<double>() - measured);
        branch_errors[executed.threshold] = error;
        std::vector<const tune_json *> order;
        for (const auto & row : ranking) if (row.at("config").at("threshold").get<float>() == executed.threshold) order.push_back(&row);
        std::stable_sort(order.begin(), order.end(), [](const tune_json * a, const tune_json * b) { return a->at("predicted_tpot_ms").get<double>() < b->at("predicted_tpot_ms").get<double>(); });
        if (order.empty()) return;
        const double cutoff = order.front()->at("predicted_tpot_ms").get<double>() + error;
        const double reference_cost = reference->at("predicted_tpot_ms").get<double>();
        double best_measured = std::numeric_limits<double>::infinity();
        for (const auto & candidate : candidates) best_measured = std::min(best_measured, candidate.tpot());
        for (const auto * row : order) {
            const double predicted = row->at("predicted_tpot_ms").get<double>();
            if (predicted > cutoff) break;
            // A predicted difference contains error from both configurations.
            if (measured + predicted - reference_cost - 2 * error > best_measured * (1 + tolerance)) continue;
            const auto & c = row->at("config");
            configuration cfg{c.at("npred"), c.at("threshold"), c.at("cpu_cap")};
            if (std::none_of(candidates.begin(), candidates.end(), [&](const candidate & entry) { return entry.config == cfg; })) { add(cfg); break; }
        }
    }
    size_t winner() const {
        size_t best = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].confirmation_ms < candidates[best].confirmation_ms) best = i;
        }
        // Retain the incumbent when the measured gain is within the near-optimal target.
        if (candidates.front().confirmation_ms <= candidates[best].confirmation_ms * (1 + tolerance)) return 0;
        return best;
    }
    void advance() {
        if (!confirming) {
            for (size_t i = 0; i < candidates.size(); ++i) if (!candidates[i].trials) {
                current = i; warming = warmup_enabled; return;
            }
            if (!confirm_required) {
                for (auto & c : candidates) { c.finalist = true; c.confirmation_ms = c.screening_ms; }
                confirming = true; current = winner(); warming = false; return;
            }
            double best = std::numeric_limits<double>::infinity();
            for (const auto & c : candidates) best = std::min(best, c.screening_ms);
            for (auto & c : candidates) c.finalist = c.config == incumbent || c.screening_ms <= best * (1 + tolerance);
            confirming = true;
        }
        // Confirm in reverse order; short and long request scores remain separate.
        for (size_t n = candidates.size(); n > 0; --n) if (candidates[n - 1].finalist && candidates[n - 1].trials < 2) {
            current = n - 1; warming = warmup_enabled; return;
        }
        current = winner(); warming = false;
    }
    configuration next() const { GGML_ASSERT(!candidates.empty()); return candidates[current].config; }
    tune_json json() const {
        tune_json result = tune_json::array();
        for (const auto & c : candidates) result.push_back({{"config", c.config.json()}, {"trials", c.trials}, {"finalist", c.finalist},
            {"screening_tpot_ms", std::isfinite(c.screening_ms) ? tune_json(c.screening_ms) : tune_json(nullptr)},
            {"measured_tpot_ms", std::isfinite(c.confirmation_ms) ? tune_json(c.confirmation_ms) : tune_json(nullptr)}, {"output_tokens", c.tokens}});
        return result;
    }
};

struct cost_warmup {
    std::array<std::array<int, 9>, 2> calls{};
    bool observe(int batch, bool cpu) {
        GGML_ASSERT(batch >= 1 && batch <= 8);
        return ++calls[cpu][batch] > 3;
    }
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

struct layer_sample {
    ggml_moe_tune_layer stats{};
    std::vector<int> owners;
    std::vector<uint64_t> freq;
    std::vector<uint8_t> prefetchable;

    layer_sample() = default;
    explicit layer_sample(const ggml_moe_tune_layer & l) : stats(l), owners(l.owners, l.owners + l.slots), freq(l.freq, l.freq + l.experts), prefetchable(l.prefetchable, l.prefetchable + l.slots) {
        stats.owners = nullptr;
        stats.freq = nullptr;
        stats.prefetchable = nullptr;
    }

    ggml_moe_tune_layer view() const {
        GGML_ASSERT(owners.size() == size_t(stats.slots) && freq.size() == size_t(stats.experts) && prefetchable.size() == owners.size());
        auto l = stats;
        l.owners = owners.data();
        l.freq = freq.data();
        l.prefetchable = prefetchable.data();
        return l;
    }
};

struct round_sample {
    std::vector<float> confidence;
    std::vector<double> draft_prefix_us;
    std::vector<layer_sample> layers;
    int accepted = 0, kept = 0, round = 0, max_draft = 7, position = -1, phase = -1;
    configuration executed;
    double draft_us = 0, verify_us = 0;

    size_t storage_bytes() const {
        size_t bytes = sizeof(*this) + confidence.capacity() * sizeof(float) + draft_prefix_us.capacity() * sizeof(double) + layers.capacity() * sizeof(layer_sample);
        for (const auto & layer : layers) {
            bytes += layer.owners.capacity() * sizeof(int) + layer.freq.capacity() * sizeof(uint64_t) + layer.prefetchable.capacity();
        }
        return bytes;
    }

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

struct layer_work {
    int miss = 0, prefetch = 0, assignments = 0;
    bool fits = true;
};

struct layer_workload {
    std::vector<int> count, votes, predicted, actual;

    layer_workload() = default;
    layer_workload(const ggml_moe_tune_layer & l, int b, int width) : count(l.experts), votes(l.experts) {
        const int source_width = l.n_pred / (l.n_ids / l.top_k);
        for (int t = 0; t < b; ++t) {
            for (int j = 0; j < std::min(width, source_width); ++j) {
                const int e = l.predicted[t * source_width + j];
                if (votes[e]++ == 0) predicted.push_back(e);
            }
            for (int j = 0; j < l.top_k; ++j) {
                const int e = l.ids[t * l.top_k + j];
                if (count[e]++ == 0) actual.push_back(e);
            }
        }
        std::stable_sort(predicted.begin(), predicted.end(), [&](int a, int b) { return votes[a] > votes[b]; });
    }
};

struct cache_state {
    std::vector<int> owners, slots;
    std::vector<uint64_t> freq;
    std::vector<uint8_t> prefetchable;

    cache_state() = default;
    explicit cache_state(const layer_sample & l) : owners(l.owners), slots(l.freq.size(), -1), freq(l.freq), prefetchable(l.prefetchable) {
        for (size_t v = 0; v < owners.size(); ++v) if (owners[v] >= 0) slots[owners[v]] = int(v);
    }

    layer_work advance(const ggml_moe_tune_layer & l, int b, configuration cfg, const layer_workload * prepared = nullptr) {
        if (owners.size() != size_t(l.slots) || freq.size() != size_t(l.experts) || prefetchable.size() != owners.size()) throw std::runtime_error("cache dimensions changed within a trace segment");
        layer_workload local;
        if (!prepared) { local = layer_workload(l, b, cfg.width); prepared = &local; }
        const auto & count = prepared->count;
        const auto & votes = prepared->votes;
        const auto & predicted = prepared->predicted;
        std::vector<bool> protect(l.experts), pinned(owners.size());
        auto select_cpu = [&](const std::vector<int> & ids, const std::vector<int> & counts) {
            std::vector<int> result;
            if (!cfg.cpu) return result;
            for (int e : ids) if (counts[e] < 9 && slots[e] < 0) result.push_back(e);
            std::sort(result.begin(), result.end(), [&](int a, int b) { return counts[a] != counts[b] ? counts[a] < counts[b] : a < b; });
            if (int(result.size()) > cfg.cpu) result.resize(cfg.cpu);
            return result;
        };
        const auto cpu_predicted = select_cpu(predicted, votes);
        for (int e : predicted) protect[e] = std::find(cpu_predicted.begin(), cpu_predicted.end(), e) == cpu_predicted.end();
        auto victims = [&](bool prefetch) {
            std::vector<int> result;
            for (int v = int(owners.size()) - 1; v >= 0; --v) {
                if (owners[v] < 0 && prefetchable[v]) result.push_back(v);
            }
            const auto free = result.size();
            for (int v = 0; v < int(owners.size()); ++v) {
                const int e = owners[v];
                if (e < 0 || pinned[v] || (prefetch && (!prefetchable[v] || protect[e]))) continue;
                result.push_back(v);
            }
            std::sort(result.begin() + free, result.end(), [&](int a, int b) {
                return freq[owners[a]] != freq[owners[b]] ? freq[owners[a]] < freq[owners[b]] : a < b;
            });
            return result;
        };
        auto fill = [&](int e, int v) {
            if (owners[v] >= 0) slots[owners[v]] = -1;
            owners[v] = e; slots[e] = v; prefetchable[v] = false;
        };
        layer_work work;
        const bool needs_prefetch = std::any_of(predicted.begin(), predicted.end(), [&](int e) { return protect[e] && slots[e] < 0; });
        const auto prefetch_slots = needs_prefetch ? victims(true) : std::vector<int>{};
        size_t next = 0;
        for (int e : predicted) {
            if (!protect[e] || slots[e] >= 0) continue;
            if (next == prefetch_slots.size()) break;
            fill(e, prefetch_slots[next++]); ++work.prefetch;
        }
        const auto cpu = select_cpu(prepared->actual, count);
        if (prepared->actual.size() - cpu.size() > owners.size()) {
            work.fits = false;
            return work;
        }
        std::vector<bool> on_cpu(l.experts);
        for (int e : cpu) { on_cpu[e] = true; work.assignments += count[e]; }
        // Pin all hits before demand fills, as in the runtime.
        for (int e : prepared->actual) if (!on_cpu[e] && slots[e] >= 0) pinned[slots[e]] = true;
        const bool needs_fill = std::any_of(prepared->actual.begin(), prepared->actual.end(), [&](int e) { return !on_cpu[e] && slots[e] < 0; });
        const auto demand_slots = needs_fill ? victims(false) : std::vector<int>{};
        next = 0;
        for (int i = 0; i < b * l.top_k; ++i) {
            const int e = l.ids[i];
            if (on_cpu[e]) continue;
            if (slots[e] < 0) {
                if (next == demand_slots.size()) throw std::runtime_error("candidate exceeds expert cache capacity");
                fill(e, demand_slots[next++]); ++work.miss;
            }
            ++freq[e]; pinned[slots[e]] = true;
        }
        // The runtime advances the epoch after the layer completes.
        std::fill(prefetchable.begin(), prefetchable.end(), true);
        return work;
    }
};

inline layer_work estimate_work(const ggml_moe_tune_layer & l, int b, configuration config) {
    return cache_state(layer_sample(l)).advance(l, b, config);
}

struct round_prediction {
    double draft = 0, base = 0, miss = 0, exposed = 0, joint = 0, cpu_overhead = 0;
    int estimated = 0;
    bool supported = true;
    double total() const { return draft + base + miss + exposed + joint + cpu_overhead; }
};

struct residual_table {
    std::array<mean_value, 9> base{}, cpu_residual{}, cpu_layers{};
};

struct latency_model {
    std::map<std::string, service_model> costs;
    std::array<residual_table, 3> phases{};
    bool phase_costs = false;
    bool warm_costs = false;
    std::array<mean_value, 9> base{}, cpu_residual{}, cpu_layers{};

    void observe_residual(const round_sample & sample) {
        double accounted = 0;
        int active = 0;
        for (const auto & layer : sample.layers) {
            const auto & l = layer.stats;
            if (l.service_us < 0) return;
            accounted += l.gate_us + l.service_us;
            active += l.assignments > 0;
        }
        const int b = sample.kept + 1;
        // Gate timing includes exposed prefetch waits.
        const double residual = std::max(0.0, sample.verify_us - accounted);
        if (!active) base[b].add(residual);
        else { cpu_residual[b].add(residual); cpu_layers[b].add(active); }
        if (sample.phase >= 0 && sample.phase < 3) {
            phase_costs = true;
            auto & table = phases[sample.phase];
            if (!active) table.base[b].add(residual);
            else { table.cpu_residual[b].add(residual); table.cpu_layers[b].add(active); }
        }
    }

    static tune_json save_value(const mean_value & m) { return {{"samples", m.count}, {"us", m.value}}; }
    static mean_value load_value(const tune_json & j) {
        mean_value m{j.at("samples").get<int>(), j.at("us").get<double>()};
        if (m.count < 0 || m.count > 100000000 || !std::isfinite(m.value) || m.value < 0) throw std::runtime_error("invalid cost profile value");
        return m;
    }

    tune_json json(const std::string & signature = "") const {
        tune_json result = {{"version", phase_costs ? (warm_costs ? 4 : 3) : 2}, {"signature", signature}, {"base", tune_json::array()}, {"cpu_residual", tune_json::array()}, {"cpu_layers", tune_json::array()}, {"shapes", tune_json::object()}};
        for (int b = 1; b <= 8; ++b) {
            result["base"].push_back(save_value(base[b]));
            result["cpu_residual"].push_back(save_value(cpu_residual[b]));
            result["cpu_layers"].push_back(save_value(cpu_layers[b]));
        }
        if (phase_costs) {
            result["phases"] = tune_json::array();
            for (const auto & table : phases) {
                tune_json item = {{"base", tune_json::array()}, {"cpu_residual", tune_json::array()}, {"cpu_layers", tune_json::array()}};
                for (int b = 1; b <= 8; ++b) {
                    item["base"].push_back(save_value(table.base[b]));
                    item["cpu_residual"].push_back(save_value(table.cpu_residual[b]));
                    item["cpu_layers"].push_back(save_value(table.cpu_layers[b]));
                }
                result["phases"].push_back(std::move(item));
            }
        }
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
        if (j.at("version") != 1 && j.at("version") != 2 && j.at("version") != 3 && j.at("version") != 4) throw std::runtime_error("unsupported cost profile version");
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
        std::array<mean_value, 9> restored_cpu{}, restored_layers{};
        if (j.at("version") >= 2) {
            if (j.at("cpu_residual").size() != 8 || j.at("cpu_layers").size() != 8) throw std::runtime_error("invalid residual table");
            for (int b = 1; b <= 8; ++b) {
                restored_cpu[b] = load_value(j.at("cpu_residual").at(b - 1));
                restored_layers[b] = load_value(j.at("cpu_layers").at(b - 1));
                if (restored_cpu[b].count != restored_layers[b].count || (restored_layers[b].count && restored_layers[b].value <= 0)) throw std::runtime_error("invalid CPU residual coverage");
            }
        }
        std::array<residual_table, 3> restored_phases{};
        if (j.at("version") >= 3) {
            if (j.at("phases").size() != 3) throw std::runtime_error("invalid phase table");
            for (int phase = 0; phase < 3; ++phase) {
                const auto & p = j.at("phases").at(phase);
                auto & table = restored_phases[phase];
                if (p.at("base").size() != 8 || p.at("cpu_residual").size() != 8 || p.at("cpu_layers").size() != 8) throw std::runtime_error("invalid phase dimensions");
                for (int b = 1; b <= 8; ++b) {
                    table.base[b] = load_value(p.at("base").at(b - 1));
                    table.cpu_residual[b] = load_value(p.at("cpu_residual").at(b - 1));
                    table.cpu_layers[b] = load_value(p.at("cpu_layers").at(b - 1));
                    if (table.cpu_residual[b].count != table.cpu_layers[b].count || (table.cpu_layers[b].count && (table.cpu_layers[b].value <= 0 || table.cpu_layers[b].value > 128))) throw std::runtime_error("invalid phase CPU coverage");
                }
            }
        }
        costs = std::move(restored); base = restored_base;
        cpu_residual = restored_cpu; cpu_layers = restored_layers;
        phases = restored_phases; phase_costs = j.at("version") >= 3; warm_costs = j.at("version") == 4;
    }
    round_prediction predict_round(const round_sample & sample, int b, configuration cfg, std::map<int, cache_state> & states, const std::vector<layer_workload> * workload = nullptr, int phase = -1) const {
        round_prediction p;
        if (phase < -1 || phase > 2) { p.supported = false; return p; }
        const bool staged = phase >= 0 && phase_costs;
        const auto & base_table = staged ? phases[phase].base : base;
        const auto & cpu_table = staged ? phases[phase].cpu_residual : cpu_residual;
        p.base = interpolate(base_table, b, p.estimated);
        if (p.base < 0) { p.supported = false; return p; }
        p.draft = sample.draft_cost(b - 1);
        // Shape-transition overhead is not a GPU prefetch overlap window.
        const double steady = staged ? interpolate(phases[2].base, b, p.estimated) : p.base;
        if (steady < 0) { p.supported = false; return p; }
        const double gap = steady / std::max<size_t>(1, sample.layers.size());
        double overlap = gap;
        int active = 0;
        size_t layer_index = 0;
        for (const auto & layer : sample.layers) {
            const auto l = layer.view();
            const auto it = costs.find(shape_key(l));
            const double gpu = it == costs.end() ? -1 : interpolate(it->second.gpu, b, p.estimated);
            if (cfg.width > l.n_pred / (l.n_ids / l.top_k) || gpu < 0 || !it->second.transfer.count) { p.supported = false; return p; }
            const auto & c = it->second;
            auto state = states.find(l.layer);
            if (state == states.end()) state = states.emplace(l.layer, cache_state(layer)).first;
            const auto work = state->second.advance(l, b, cfg, workload ? &(*workload)[layer_index] : nullptr);
            if (!work.fits) { p.supported = false; return p; }
            ++layer_index;
            double joint = gpu;
            if (work.assignments) {
                ++active;
                const auto exact = c.joint.find({b, work.assignments});
                if (exact != c.joint.end() && exact->second.count >= 3) joint = std::max(gpu, exact->second.value);
                else {
                    if (!c.cpu_per_assignment.count || !c.protocol.count) { p.supported = false; return p; }
                    joint = std::max(gpu, c.cpu_per_assignment.value * work.assignments) + c.protocol.value;
                    ++p.estimated;
                }
            }
            const double copy = std::max(c.transfer.value, c.issue.value);
            p.miss += work.miss * copy;
            p.exposed += std::max(0.0, work.prefetch * copy - overlap);
            p.joint += joint;
            overlap = joint + gap;
        }
        if (active) {
            const double residual = interpolate(cpu_table, b, p.estimated);
            if (residual < 0) { p.supported = false; return p; }
            // This residual covers the whole verification, not one CPU-active layer.
            p.cpu_overhead = residual - p.base;
        }
        return p;
    }

    tune_json rank(const std::deque<round_sample> & samples, int max_width, bool own_threshold = false) const {
        if (samples.size() < 8) throw std::runtime_error("need at least eight sampled rounds with layer records");
        if (samples.size() > max_workload_rounds) throw std::runtime_error("workload exceeds the search limit");
        size_t bytes = 0;
        for (const auto & sample : samples) {
            if (sample.storage_bytes() > max_workload_bytes - bytes) throw std::runtime_error("workload exceeds the memory limit");
            bytes += sample.storage_bytes();
        }
        const int64_t started = ggml_time_us();
        auto evaluate_width = [&](int width) {
            tune_json rows = tune_json::array();
            std::map<std::pair<size_t, int>, std::vector<layer_workload>> workloads;
            const auto thresholds = own_threshold ? std::vector<float>{samples.front().executed.threshold} : std::vector<float>{0.0f, 0.4f, 0.6f, 0.8f};
            std::vector<int> cpu_caps = {0, 2, 4, 8};
            if (own_threshold && std::find(cpu_caps.begin(), cpu_caps.end(), samples.front().executed.cpu) == cpu_caps.end()) cpu_caps.push_back(samples.front().executed.cpu);
            for (float threshold : thresholds) for (int cpu : cpu_caps) {
                configuration cfg{width, threshold, cpu};
                double time = 0, tokens = 0, adjustment = 0;
                int estimated = 0, blocks = 0, transitions = 0, previous_batch = 0, phase = 0;
                bool supported = true;
                std::map<int, cache_state> states;
                const round_sample * previous = nullptr;
                for (size_t i = 0; i < samples.size(); ++i) {
                    const auto & sample = samples[i];
                    if (own_threshold && sample.executed.threshold != threshold) { supported = false; break; }
                    const int k = prefix_length(sample.confidence, threshold), b = k + 1;
                    if (k > sample.kept) { supported = false; break; }
                    // Missing rounds or changed accepted prefixes leave the observed trajectory.
                    const bool adjacent = previous && sample.round == previous->round + 1;
                    const bool consecutive = adjacent && previous->accepted <= prefix_length(previous->confidence, threshold) &&
                        previous->position >= 0 && sample.position == previous->position + previous->accepted + 1;
                    if (!consecutive) { states.clear(); ++blocks; }
                    else ++transitions;
                    const auto key = std::make_pair(i, b);
                    auto plans = workloads.find(key);
                    if (plans == workloads.end()) {
                        std::vector<layer_workload> layers;
                        layers.reserve(sample.layers.size());
                        for (const auto & layer : sample.layers) layers.emplace_back(layer.view(), b, width);
                        plans = workloads.emplace(key, std::move(layers)).first;
                    }
                    phase = consecutive ? next_execution_phase(b, previous_batch, phase) : 0;
                    previous_batch = b;
                    const auto prediction = predict_round(sample, b, cfg, states, &plans->second, phase);
                    if (!prediction.supported) { supported = false; break; }
                    time += prediction.total();
                    tokens += 1 + std::min(sample.accepted, k);
                    adjustment += prediction.cpu_overhead;
                    estimated += prediction.estimated;
                    previous = &sample;
                }
                if (!supported) continue;
                rows.push_back({{"config", cfg.json()}, {"predicted_tpot_ms", time / tokens / 1000},
                    {"cpu_residual_adjustment_ms_per_token", adjustment / tokens / 1000},
                    {"estimated_service_lookups", estimated}, {"cache_blocks", blocks}, {"cache_transitions", transitions}});
            }
            return rows;
        };
        // Widths have independent replay states. Merge in configuration order for stable ties.
        std::vector<std::future<tune_json>> jobs;
        std::vector<int> widths = {0, 8, 16, 24};
        if (own_threshold && std::find(widths.begin(), widths.end(), samples.front().executed.width) == widths.end()) widths.push_back(samples.front().executed.width);
        for (int width : widths) if (width <= max_width) {
            jobs.push_back(std::async(std::launch::async, evaluate_width, width));
        }
        tune_json ranking = tune_json::array(), best = nullptr;
        double best_tpot = std::numeric_limits<double>::infinity();
        for (auto & job : jobs) for (auto & row : job.get()) {
            const double tpot = row.at("predicted_tpot_ms");
            if (tpot < best_tpot) { best_tpot = tpot; best = row.at("config"); }
            ranking.push_back(std::move(row));
        }
        const int full = std::count_if(samples.begin(), samples.end(), [](const round_sample & s) { return s.kept == s.max_draft; });
        return {{"event", "recommendation"}, {"samples", samples.size()}, {"full_samples", full}, {"supported_candidates", ranking.size()},
            {"selected", best}, {"branch_candidates", branch_candidates(ranking, !own_threshold)}, {"trajectory_scope", own_threshold ? "observed_threshold" : "cross_threshold_approximation"},
            {"search_ms", (ggml_time_us() - started) / 1000.0}, {"ranking", ranking}};
    }
};

} // namespace moe_autotune_model
