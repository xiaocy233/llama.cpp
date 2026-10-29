#include "moe-autotune.h"
#include "moe-autotune-model.h"
#include "../src/llama-ext.h"
#include "log.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace moe_autotune_model;

namespace {

template <typename T>
void append_numbers(std::string & output, const char * key, const T * values, size_t count) {
    output += ",\"";
    output += key;
    output += "\":[";
    char buffer[32];
    for (size_t i = 0; i < count; ++i) {
        if (i) output += ',';
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), values[i]);
        GGML_ASSERT(result.ec == std::errc{});
        output.append(buffer, result.ptr);
    }
    output += ']';
}

template <typename T>
void append_field(std::string & output, const char * key, const T & value) {
    output += ",\"";
    output += key;
    output += "\":";
    if constexpr (std::is_integral<T>::value) {
        char buffer[32];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        GGML_ASSERT(result.ec == std::errc{});
        output.append(buffer, result.ptr);
    } else {
        output += tune_json(value).dump();
    }
}

int env_int(const char * key, int fallback, int low, int high) {
    const char * s = std::getenv(key);
    if (!s) return fallback;
    char * end = nullptr;
    const long n = std::strtol(s, &end, 10);
    if (!*s || *end || n < low || n > high) throw std::runtime_error(std::string("invalid ") + key);
    return int(n);
}

}

struct common_moe_autotune::impl {
    llama_context * target;
    llama_context * draft;
    int max_draft;
    bool sequential;
    bool online, adaptive, active = false, pending = false, collecting = false, probing = false, profiled = false, loaded = false, detail = false;
    int request = 0, round = 0, max_width, window, cost_limit, cost_done = 0, prompt_tokens = 0;
    int sampled_rounds = 0, traced_rounds = 0, trial_budget = 24;
    bool trace_only = false, warming = false;
    int warmup_done = 0;
    bool cache_saved = false, candidate_warmup = false, candidate_confirm = true;
    int64_t cache_reset_us = 0;
    int64_t warmup_us = 0, decode_start = 0;
    cost_warmup warmup;
    candidate_frontier frontier;
    std::vector<llama_token> frontier_prompt;
    token_sample_schedule sampling;
    std::deque<round_sample> samples;
    size_t workload_bytes = 0;
    bool workload_overflow = false;
    int force_keep = -1, position = 0, full_seen = 0;
    int previous_batch = 0, phase = 0;
    int64_t request_start = 0, round_start = 0, verify_start = 0, calibration_us = 0, control_us = 0;
    int64_t search_us = 0;
    double measured_us = 0;
    uint64_t measured_tokens = 0;
    std::string profile_path, signature, output_path;
    std::ofstream output;
    std::string serialized;
    std::array<ggml_moe_tune_layer, 128> layer_buffer{};
    latency_model model;
    std::map<int, std::vector<uint64_t>> frequencies;
    std::vector<int> undrafted_steps;
    round_sample current;
    configuration configured, executing;

    impl(llama_context * target, llama_context * draft, int max_draft, bool sequential) : target(target), draft(draft), max_draft(max_draft), sequential(sequential) {
        if (max_draft < 1 || max_draft > 7) throw std::runtime_error("autotune supports between one and seven draft tokens");
        const std::string mode = std::getenv("LLAMA_MOE_AUTOTUNE");
        if (mode != "offline" && mode != "online") throw std::runtime_error("LLAMA_MOE_AUTOTUNE must be offline or online");
        online = mode == "online";
        adaptive = env_int("LLAMA_MOE_AUTOTUNE_ADAPT", online ? 1 : 0, 0, 1);
        sampling.interval = online ? env_int("LLAMA_MOE_AUTOTUNE_SAMPLE_TOKENS", 20, 1, 10000) : 1;
        if (online && std::getenv("LLAMA_MOE_AUTOTUNE_SAMPLE_INTERVAL")) throw std::runtime_error("use LLAMA_MOE_AUTOTUNE_SAMPLE_TOKENS; intervals now count output tokens, not decode steps");
        detail = env_int("LLAMA_MOE_AUTOTUNE_DETAIL", online ? 0 : 1, 0, 1) != 0;
        const char * path = std::getenv("LLAMA_MOE_AUTOTUNE_OUTPUT");
        if (!path || !*path) throw std::runtime_error("LLAMA_MOE_AUTOTUNE_OUTPUT is required");
        output_path = path;
        if (std::ifstream(output_path).good()) throw std::runtime_error("autotune output already exists");
        output.open(output_path);
        if (!output) throw std::runtime_error("cannot open autotune output");
        max_width = env_int("LLAMA_MOE_AUTOTUNE_MAX_PRED", 24, 8, 24);
        window = env_int("LLAMA_MOE_AUTOTUNE_WINDOW", 32, 16, 128);
        cost_limit = env_int("LLAMA_MOE_AUTOTUNE_COST_ROUNDS", 48, 0, 96);
        trial_budget = env_int("LLAMA_MOE_AUTOTUNE_TRIAL_BUDGET", 24, 10, 24);
        candidate_warmup = env_int("LLAMA_MOE_AUTOTUNE_WARMUP_REQUESTS", sequential ? 1 : 0, 0, 1);
        candidate_confirm = env_int("LLAMA_MOE_AUTOTUNE_CONFIRM_REQUESTS", sequential ? 0 : 1, 0, 1);
        if (const char * s = std::getenv("LLAMA_MOE_AUTOTUNE_PROFILE")) profile_path = s;
        if (online) {
            configured.width = env_int("LLAMA_MOE_AUTOTUNE_NPRED", 0, 0, max_width);
            configured.cpu = env_int("LLAMA_MOE_AUTOTUNE_CPU_CAP", 0, 0, 8);
            if (const char * value = std::getenv("LLAMA_MOE_AUTOTUNE_THRESHOLD")) {
                char * end = nullptr;
                configured.threshold = std::strtof(value, &end);
                if (!*value || *end || !std::isfinite(configured.threshold) || configured.threshold < 0 || configured.threshold > 1) throw std::runtime_error("invalid online collection threshold");
            }
        }
        emit({{"event", "start"}, {"version", 8}, {"mode", mode}, {"spec_type", sequential ? "draft-mtp" : "draft-dflash"}, {"max_draft", max_draft},
              {"collection_only", !adaptive}, {"configured", configured.json()}, {"window_rounds", window},
              {"workload_scope", "whole_request"}, {"max_workload_rounds", max_workload_rounds}, {"max_workload_bytes", max_workload_bytes},
              {"sample_tokens", sampling.interval}, {"sampling_unit", online ? "output_tokens" : "decode_rounds"},
              {"search_at", adaptive ? "request_end" : "disabled"}, {"frequency_scope", "traced_rounds"},
              {"candidate_trial_budget", trial_budget}, {"candidate_warmup_requests", int(candidate_warmup)}, {"candidate_confirmation_requests", int(candidate_confirm)}, {"candidate_tolerance", candidate_frontier::tolerance}, {"cost_round_budget", cost_limit}, {"built_prediction_width", max_width},
              {"prediction_semantics", "runtime prefix; predictor graph stays at built width"},
              {"cache_model", "candidate LFU replay within contiguous semantic segments; reanchor at sampling gaps or changed round starts"}});
    }

    void emit(const tune_json & j) {
        output << j.dump() << '\n';
        if (!output) throw std::runtime_error("cannot write autotune telemetry");
    }

    void emit_round(const tune_json & record) {
        serialized.clear();
        serialized += record.dump();
        if (trace_only || detail || (online && profiled)) {
            serialized.pop_back();
            serialized += ",\"layers\":[";
            for (size_t i = 0; i < current.layers.size(); ++i) {
                if (i) serialized += ',';
                const auto & layer = current.layers[i];
                const auto & l = layer.stats;
                serialized += "{\"layer\":";
                serialized += std::to_string(l.layer);
                if (!trace_only) {
                    append_field(serialized, "shape", shape_key(l));
                    append_field(serialized, "slots", l.slots);
                    append_field(serialized, "experts", l.experts);
                    append_field(serialized, "top_k", l.top_k);
                    append_field(serialized, "width", l.width);
                    append_field(serialized, "loads", l.loads);
                    append_field(serialized, "prefetches", l.prefetches);
                    append_field(serialized, "cpu_assignments", l.assignments);
                    append_field(serialized, "gate_us", l.gate_us);
                    append_field(serialized, "fill_us", l.fill_us);
                    append_field(serialized, "prefetch_us", l.prefetch_us);
                    append_field(serialized, "cpu_us", l.cpu_us);
                    append_field(serialized, "service_us", l.service_us >= 0 ? tune_json(l.service_us) : tune_json(nullptr));
                }
                // Write integer arrays without a JSON allocation for each element.
                if (!trace_only) {
                    append_numbers(serialized, "bytes", l.bytes, 3);
                    append_numbers(serialized, "types", l.types, 3);
                    append_numbers(serialized, "owners", layer.owners.data(), layer.owners.size());
                    append_numbers(serialized, "lfu", layer.freq.data(), layer.freq.size());
                    serialized += ",\"prefetchable\":[";
                    for (size_t v = 0; v < layer.prefetchable.size(); ++v) {
                        if (v) serialized += ',';
                        serialized += layer.prefetchable[v] ? "true" : "false";
                    }
                    serialized += ']';
                }
                append_numbers(serialized, "actual", l.ids, l.n_ids);
                append_numbers(serialized, "predicted", l.predicted, l.n_pred);
                serialized += '}';
            }
            serialized += "]}";
        }
        output << serialized << '\n';
        if (!output) throw std::runtime_error("cannot write autotune telemetry");
    }

    std::string identity(const std::vector<layer_sample> & layers) {
        char description[256];
        llama_model_desc(llama_get_model(target), description, sizeof(description));
        std::string id = description;
        llama_model_desc(llama_get_model(draft), description, sizeof(description));
        id += std::string("|") + description;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(ggml_backend_dev_get(i), &props);
            id += std::string("|") + props.description + "|" + (props.device_id ? props.device_id : "");
        }
        if (const char * t = std::getenv("GGML_MOE_HYBRID_THREADS")) id += std::string("|threads=") + t;
        if (const char * t = std::getenv("LLAMA_MOE_AUTOTUNE_PROFILE_KEY")) id += std::string("|") + t;
        id += "|context=" + std::to_string(llama_n_ctx(target)) + "|prompt_bucket=" + std::to_string(prompt_tokens / 256);
        id += std::string("|drafter=") + (sequential ? "mtp" : "dflash") + "|max_draft=" + std::to_string(max_draft);
        for (const auto & layer : layers) {
            const auto l = layer.view();
            id += "|" + std::to_string(l.layer) + ":" + std::to_string(l.slots) + ":" + shape_key(l);
        }
        return id;
    }

    void load_profile() {
        if (profile_path.empty()) return;
        std::ifstream file(profile_path);
        if (!file) { emit({{"event", "profile"}, {"status", "missing"}}); return; }
        file.seekg(0, std::ios::end);
        if (file.tellg() > 4 * 1024 * 1024) throw std::runtime_error("cost profile too large");
        file.seekg(0);
        tune_json j; file >> j;
        if (j.at("version") != 4 || j.at("signature") != signature) {
            emit({{"event", "profile"}, {"status", "incompatible"}});
            return;
        }
        model.restore(j); loaded = true;
        emit({{"event", "profile"}, {"status", "reused"}, {"values", model.json(signature)}});
    }

    void learn(const round_sample & sample) {
        model.warm_costs = true;
        const int b = sample.kept + 1;
        for (const auto & layer : sample.layers) {
            const auto l = layer.view();
            auto & c = model.costs[shape_key(l)];
            if (executing.width == 0 && l.loads >= 4) {
                c.transfer.add(l.gate_us / l.loads);
                c.issue.add(l.fill_us / l.loads);
            }
            if (l.service_us < 0) continue;
            if (!l.assignments) {
                if (executing.cpu == 0) c.gpu[b].add(l.service_us);
            } else {
                c.joint[{b, l.assignments}].add(l.service_us);
                c.cpu_per_assignment.add(l.cpu_us / l.assignments);
                int estimated = 0;
                const double gpu = interpolate(c.gpu, b, estimated);
                if (gpu >= 0) c.protocol.add(std::max(0.0, l.service_us - std::max(gpu, l.cpu_us)));
            }
        }
        model.observe_residual(sample);
    }


    void report(const char * event) {
        if (!request) return;
        const int64_t started = ggml_time_us();
        emit({{"event", event}, {"request", request}, {"rounds", round}, {"full_rounds", full_seen},
              {"sampled_rounds", sampled_rounds}, {"traced_rounds", traced_rounds}, {"sample_tokens", sampling.interval}, {"output_tokens", sampling.tokens},
              {"search_ms", search_us / 1000.0}, {"cache_reset_ms", cache_reset_us / 1000.0},
              {"workload_bytes", workload_bytes}, {"workload_limit_exceeded", workload_overflow},
              {"undrafted_steps", undrafted_steps},
              {"cost_probe_rounds", cost_done}, {"cost_probe_ms", calibration_us / 1000.0},
              {"cost_warmup_rounds", warmup_done}, {"cost_warmup_ms", warmup_us / 1000.0},
              {"controller_ms", control_us / 1000.0}, {"elapsed_ms", (ggml_time_us() - request_start) / 1000.0},
              {"sampled_round_tpot_ms", measured_tokens ? measured_us / measured_tokens / 1000.0 : 0},
              {"configured", configured.json()}, {"costs", model.json(signature)}});
        LOG_INF("MoE autotune: rounds=%d sampled=%d full=%d cost_probes=%d probe_time=%.3f ms controller=%.3f ms elapsed=%.3f ms\n",
                round, sampled_rounds, full_seen, cost_done, calibration_us / 1000.0, control_us / 1000.0, (ggml_time_us() - request_start) / 1000.0);
        for (const auto & entry : model.costs) {
            const auto & c = entry.second;
            LOG_INF("MoE autotune cost: shape=%s copy=%.3f us/expert n=%d issue=%.3f us/expert CPU=%.3f us/assignment n=%d protocol=%.3f us\n",
                    entry.first.c_str(), c.transfer.value, c.transfer.count, c.issue.value, c.cpu_per_assignment.value, c.cpu_per_assignment.count, c.protocol.value);
            for (int b = 1; b <= 8; ++b) if (c.gpu[b].count) LOG_INF("MoE autotune GPU: B=%d service=%.3f us n=%d\n", b, c.gpu[b].value, c.gpu[b].count);
        }
        for (int b = 1; b <= 8; ++b) if (model.base[b].count) {
            LOG_INF("MoE autotune residual: B=%d GPU=%.3f us n=%d CPU=%.3f us n=%d active_layers=%.2f\n", b, model.base[b].value, model.base[b].count, model.cpu_residual[b].value, model.cpu_residual[b].count, model.cpu_layers[b].value);
        }
        for (int phase = 0; phase < 3; ++phase) for (int b = 1; b <= 8; ++b) {
            const auto & table = model.phases[phase];
            if (table.base[b].count || table.cpu_residual[b].count) {
                LOG_INF("MoE autotune phase: B=%d shape_run=%d GPU=%.3f us n=%d CPU=%.3f us n=%d active_layers=%.2f\n",
                        b, phase + 1, table.base[b].value, table.base[b].count, table.cpu_residual[b].value, table.cpu_residual[b].count, table.cpu_layers[b].value);
            }
        }
        tune_json observed = tune_json::array();
        for (const auto & entry : frequencies) observed.push_back({{"layer", entry.first}, {"counts", entry.second}});
        emit({{"event", "expert_frequencies"}, {"request", request}, {"rounds", round}, {"sampled_rounds", sampled_rounds},
              {"sample_tokens", sampling.interval}, {"frequency_scope", "traced_rounds"}, {"layers", observed}});
        output.flush();
        if (!profile_path.empty() && !signature.empty()) {
            const std::string temporary = profile_path + "." + std::to_string(ggml_time_us()) + ".tmp";
            std::ofstream file(temporary);
            file << model.json(signature).dump(2) << '\n'; file.close();
            if (!file || std::rename(temporary.c_str(), profile_path.c_str())) throw std::runtime_error("cannot save autotune cost profile");
        }
        control_us += ggml_time_us() - started;
    }
};

common_moe_autotune::common_moe_autotune(llama_context * target, llama_context * draft, int max_draft, bool sequential) : state(new impl(target, draft, max_draft, sequential)) {}

common_moe_autotune::~common_moe_autotune() {
    try { if (state->active) state->report("final"); }
    catch (const std::exception & e) { LOG_ERR("MoE autotune final report: %s\n", e.what()); }
}

void common_moe_autotune::begin(const std::vector<llama_token> & prompt) {
    auto & s = *state;
    if (s.active) end(s.sampling.tokens, false);
    if (s.frontier_prompt != prompt) {
        s.frontier = {};
        s.frontier_prompt = prompt;
        s.cache_saved = false;
    }
    s.cache_reset_us = 0;
    if (s.adaptive) {
        const int64_t started = ggml_time_us();
        if (!s.cache_saved) {
            if (!llama_moe_tune_cache(s.target, true)) throw std::runtime_error("cannot capture autotune expert cache");
            s.cache_saved = true;
        } else if (!s.frontier.candidates.empty() && !s.frontier.ready() && (s.frontier.warming || !s.frontier.warmup_enabled)) {
            if (!llama_moe_tune_cache(s.target, false)) throw std::runtime_error("cannot restore autotune expert cache");
        }
        s.cache_reset_us = ggml_time_us() - started;
    }
    if (s.adaptive && !s.online && s.request) {
        s.online = true;
        s.sampling.interval = env_int("LLAMA_MOE_AUTOTUNE_SAMPLE_TOKENS", 20, 1, 10000);
    }
    s.active = true;
    ++s.request; s.round = 0; s.sampled_rounds = 0; s.traced_rounds = 0; s.full_seen = 0; s.cost_done = 0;
    s.previous_batch = 0; s.phase = 0;
    s.workload_bytes = 0; s.workload_overflow = false;
    s.sampling.tokens = s.sampling.next = 0; s.samples.clear(); s.search_us = 0;
    s.frequencies.clear(); s.undrafted_steps.clear(); s.pending = false;
    // Workload observations never cross request boundaries.
    s.signature.clear(); s.loaded = false; s.model = {};
    s.prompt_tokens = prompt.size(); s.request_start = ggml_time_us();
    s.warmup_done = 0; s.warmup_us = 0; s.decode_start = 0;
    s.calibration_us = s.control_us = 0; s.measured_us = 0; s.measured_tokens = 0;
    s.emit({{"event", "request"}, {"request", s.request}, {"prompt_tokens", s.prompt_tokens}, {"configured", s.configured.json()}});
}

bool common_moe_autotune::prepare() {
    auto & s = *state;
    if (s.pending) throw std::runtime_error("unfinished autotune round; use single-slot llama-server without checkpoint replay");
    s.pending = true;
    // Keep route history continuous; only service timing is sampled.
    s.collecting = true;
    s.round_start = ggml_time_us();
    if (!s.decode_start) s.decode_start = s.round_start;
    s.executing = s.online ? s.configured : configuration{};
    s.probing = s.warming = false; s.force_keep = -1;
    // Warm each execution shape before measuring shape transitions and reuse.
    std::vector<int> sizes;
    for (int b : {1, 2, 4, 8}) if (b <= s.max_draft + 1) sizes.push_back(b);
    if (sizes.back() != s.max_draft + 1) sizes.push_back(s.max_draft + 1);
    const int cycle = 6 * int(sizes.size());
    if (!s.online && !s.loaded && s.round >= 8 && s.cost_done < s.cost_limit) {
        s.warming = s.warmup_done < cycle;
        const int index = s.warming ? s.warmup_done : s.cost_done % cycle;
        const int cell = index / 3;
        s.executing = {0, 0, cell < int(sizes.size()) ? 0 : 4};
        s.force_keep = sizes[cell % sizes.size()] - 1;
        s.probing = !s.warming;
    }
    s.profiled = s.online ? s.sampling.due() : (s.probing || s.warming || s.round % 8 == 0);
    s.trace_only = s.online && !s.profiled;
    if (!llama_moe_tune_begin(s.target, s.executing.width, s.executing.cpu, s.profiled, s.online, s.trace_only)) throw std::runtime_error("autotune needs CUDA expert slots >= 4 * top_k, legacy CPU diversion, and LFU");
    if (!s.trace_only) s.sampling.sampled();
    s.current = {};
    s.current.max_draft = s.max_draft;
    s.control_us += ggml_time_us() - s.round_start;
    return true;
}

float common_moe_autotune::threshold() const {
    return state->configured.threshold;
}

void common_moe_autotune::record_draft_token(float confidence) {
    auto & s = *state;
    if (!s.sequential || !s.collecting || !s.pending || s.current.confidence.size() >= size_t(s.max_draft)) throw std::runtime_error("invalid sequential draft observation");
    if (!std::isfinite(confidence) || confidence < 0 || confidence > 1) throw std::runtime_error("invalid draft confidence");
    s.current.confidence.push_back(confidence);
    s.current.draft_prefix_us.push_back(ggml_time_us() - s.round_start);
}

size_t common_moe_autotune::finish_draft(int limit, int position) {
    return truncate(state->current.confidence, limit, position);
}

size_t common_moe_autotune::truncate(const std::vector<float> & confidence, int limit, int position) {
    auto & s = *state;
    if (!s.pending || !s.collecting || confidence.empty() || confidence.size() > size_t(s.max_draft)) throw std::runtime_error("invalid autotune draft");
    for (float p : confidence) if (!std::isfinite(p) || p < 0 || p > 1) throw std::runtime_error("invalid draft confidence");
    s.current.confidence = confidence;
    s.position = position;
    int keep = prefix_length(confidence, s.executing.threshold);
    if (s.force_keep >= 0) keep = std::min(keep, s.force_keep);
    if (limit > 0) keep = std::min(keep, limit);
    s.phase = next_execution_phase(keep + 1, s.previous_batch, s.phase);
    s.previous_batch = keep + 1;
    s.current.phase = s.phase;
    s.current.kept = keep;
    s.verify_start = ggml_time_us();
    s.current.draft_us = s.verify_start - s.round_start;
    return keep;
}

void common_moe_autotune::finish(int accepted) {
    auto & s = *state;
    if (!s.pending) {
        // The last output token can use a normal decode without a draft.
        s.previous_batch = 0; s.phase = 0;
        s.undrafted_steps.push_back(s.round++);
        s.sampling.advance(accepted);
        return;
    }
    const int64_t finished = ggml_time_us();
    const int64_t control_started = finished;
    if (accepted < 0 || accepted > s.current.kept) throw std::runtime_error("invalid autotune acceptance count");
    s.current.verify_us = finished - s.verify_start;
    s.current.accepted = accepted;
    s.current.round = s.round;
    s.current.position = s.position;
    s.current.executed = s.executing;
    auto & layers = s.layer_buffer;
    const int count = llama_moe_tune_end(s.target, layers.data(), layers.size(), s.trace_only);
    if (count <= 0) throw std::runtime_error("missing autotune layer telemetry");
    s.current.layers.clear();
    s.current.layers.reserve(count);
    for (int i = 0; i < count; ++i) {
        if (layers[i].n_ids != (s.current.kept + 1) * layers[i].top_k) throw std::runtime_error("autotune verification shape mismatch");
        s.current.layers.emplace_back(layers[i]);
    }
    if (s.signature.empty()) { s.signature = s.identity(s.current.layers); s.load_profile(); }
    const bool warm_cost = s.warmup.observe(s.current.kept + 1, s.executing.cpu > 0) && !s.warming;
    if (!s.trace_only && warm_cost && s.round > 1) s.learn(s.current);
    const bool full = s.current.kept == int(s.current.confidence.size()) && s.current.kept == s.max_draft;
    // Threshold-selected full drafts bias acceptance statistics. Use untruncated rounds.
    const bool reference = full && s.executing.threshold == 0;
    const bool observed_end = int(s.current.confidence.size()) == s.max_draft || s.current.kept < int(s.current.confidence.size());
    const bool search_sample = !s.probing && !s.warming && observed_end && s.current.kept == prefix_length(s.current.confidence, s.executing.threshold);
    if (reference) ++s.full_seen;
    int loads = 0, prefetches = 0, assignments = 0;
    double gate = 0, service = 0;
    for (const auto & layer : s.current.layers) {
        const auto l = layer.view();
        auto & counts = s.frequencies[l.layer];
        counts.resize(l.experts);
        for (int i = 0; i < l.n_ids; ++i) ++counts[l.ids[i]];
        loads += l.loads; prefetches += l.prefetches; assignments += l.assignments;
        gate += l.gate_us; service += std::max(0.0, l.service_us);
    }
    const double duration = s.current.draft_us + s.current.verify_us;
    if (!s.trace_only) { s.measured_us += duration; s.measured_tokens += accepted + 1; }
    if (s.probing) { ++s.cost_done; s.calibration_us += finished - s.round_start; }
    if (s.warming) { ++s.warmup_done; s.warmup_us += finished - s.round_start; }
    tune_json record = {{"event", s.trace_only ? "route" : "round"}, {"request", s.request}, {"round", s.round}, {"position", s.position},
            {"output_tokens_before", s.sampling.tokens}, {"next_sample_tokens", s.sampling.next},
            {"config", s.executing.json()}, {"confidence", s.current.confidence}, {"kept", s.current.kept},
            {"max_draft", s.max_draft}, {"draft_prefix_us", s.current.draft_prefix_us},
            {"accepted", accepted}, {"execution_phase", s.current.phase}, {"right_censored", !full && accepted == s.current.kept}, {"full", full}, {"workload_sample", reference}, {"search_sample", search_sample},
            {"cost_probe", s.probing}, {"cost_warmup", s.warming}, {"warm_cost", warm_cost}, {"profiled", s.profiled}, {"draft_ms", s.current.draft_us / 1000},
            {"verify_ms", s.current.verify_us / 1000}, {"gate_ms", gate / 1000},
            {"expert_service_ms", s.profiled ? tune_json(service / 1000) : tune_json(nullptr)},
            {"demand_copies", loads}, {"prefetch_copies", prefetches}, {"cpu_assignments", assignments}};
    s.emit_round(record);
    if (s.adaptive && !s.frontier.ready() && search_sample && !s.workload_overflow) {
        const size_t bytes = s.current.storage_bytes();
        if (s.samples.size() == max_workload_rounds || bytes > max_workload_bytes - s.workload_bytes) {
            s.workload_overflow = true;
            s.samples.clear();
            s.workload_bytes = 0;
        } else {
            s.workload_bytes += bytes;
            s.samples.push_back(std::move(s.current));
        }
    }
    s.pending = false; ++s.round; ++s.traced_rounds;
    if (!s.trace_only) ++s.sampled_rounds;
    s.sampling.advance(accepted);
    s.control_us += ggml_time_us() - control_started;
    if (!s.trace_only && s.sampled_rounds % s.window == 0) s.report("window");
}

void common_moe_autotune::end(uint64_t output_tokens, bool completed) {
    auto & s = *state;
    if (!s.active) return;
    if (completed && s.pending) throw std::runtime_error("unfinished autotune verification at request end");
    // EOS or a stop string can end output inside an accepted batch.
    if (output_tokens < s.sampling.tokens && !s.samples.empty() && s.samples.back().round == s.round - 1) {
        s.workload_bytes -= s.samples.back().storage_bytes();
        s.samples.pop_back();
    }
    s.sampling.tokens = output_tokens;
    if (s.adaptive && completed) {
        const int64_t started = ggml_time_us();
        tune_json result = {{"event", "recommendation"}, {"samples", s.samples.size()},
            {"supported_candidates", 0}, {"selected", nullptr}, {"search_ms", 0.0},
            {"status", s.workload_overflow ? "workload_limit_exceeded" : "insufficient_samples"}};
        const bool existing = !s.frontier.candidates.empty();
        bool searched = false;
        if (existing) {
            const bool warmup = s.frontier.warming;
            const bool was_ready = s.frontier.ready();
            const bool confirming = s.frontier.confirming;
            const double elapsed = started - s.decode_start;
            const bool measured = s.decode_start && !s.cost_done && !s.warmup_done && s.frontier.observe(s.configured, elapsed, output_tokens);
            result["trial_role"] = was_ready ? "selected" : warmup ? "warmup" : confirming ? "confirmation" : "measurement";
            bool branch_representative = true;
            if (s.configured == s.frontier.incumbent) {
                for (const auto & candidate : s.frontier.candidates) {
                    if (candidate.config.threshold == s.configured.threshold && !(candidate.config == s.configured)) branch_representative = false;
                }
            }
            if (measured && branch_representative && s.frontier.needs_refinement(s.configured) && s.samples.size() >= 8) {
                result = s.model.rank(s.samples, s.max_width, true);
                result["trial_role"] = "measurement";
                s.frontier.refine(result["ranking"], s.configured, elapsed / output_tokens / 1000);
                searched = true;
            }
            if (measured) s.frontier.advance();
        } else if (s.samples.size() >= 8) {
            result = s.model.rank(s.samples, s.max_width);
            result["predicted_selected"] = result["selected"];
            result["trial_role"] = "bootstrap";
            result["status"] = "missing_cost_or_prefix_coverage";
            if (!result["selected"].is_null()) s.frontier.seed(result["branch_candidates"], s.configured, s.trial_budget, s.candidate_warmup, s.candidate_confirm);
            searched = true;
        }
        result["request"] = s.request;
        result["previous_config"] = s.configured.json();
        result["applies_to"] = "next_request";
        if (!s.frontier.candidates.empty()) {
            s.configured = s.frontier.next();
            result["selected"] = s.configured.json();
            result["status"] = s.frontier.ready() ? "validated" : "candidate_trials";
            result["next_trial_role"] = s.frontier.ready() ? "selected" : s.frontier.warming ? "warmup" : s.frontier.confirming ? "confirmation" : "measurement";
            result["candidate_frontier"] = s.frontier.json();
            result["completed_trials"] = s.frontier.completed;
            result["trial_budget"] = s.frontier.budget;
            result["branch_errors"] = tune_json::array();
            for (const auto & entry : s.frontier.branch_errors) result["branch_errors"].push_back({{"threshold", entry.first}, {"absolute_prediction_error_ms", entry.second}});
        }
        result["searched"] = searched;
        result["next_config"] = s.configured.json();
        s.search_us = ggml_time_us() - started;
        result["search_ms"] = s.search_us / 1000.0;
        s.emit(result);
        LOG_INF("MoE autotune request end: request=%d search=%.3f ms next=%s\n", s.request, s.search_us / 1000.0, s.configured.json().dump().c_str());
    }
    if (s.adaptive && !completed && !s.frontier.candidates.empty() && !s.frontier.ready()) s.frontier.warming = s.frontier.warmup_enabled;
    s.report(completed ? "request_end" : "request_cancelled");
    s.output.flush();
    s.samples.clear();
    s.workload_bytes = 0;
    s.active = false;
}
