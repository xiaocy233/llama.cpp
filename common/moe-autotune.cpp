#include "moe-autotune.h"
#include "moe-autotune-model.h"
#include "../src/llama-ext.h"
#include "log.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

using namespace moe_autotune_model;

namespace {

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
    int sampled_rounds = 0;
    token_sample_schedule sampling;
    std::deque<round_sample> samples;
    int force_keep = -1, position = 0, full_seen = 0;
    int64_t request_start = 0, round_start = 0, verify_start = 0, calibration_us = 0, control_us = 0;
    int64_t search_us = 0;
    double measured_us = 0;
    uint64_t measured_tokens = 0;
    std::string profile_path, signature, output_path;
    std::ofstream output;
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
        adaptive = online && env_int("LLAMA_MOE_AUTOTUNE_ADAPT", 1, 0, 1);
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
        cost_limit = env_int("LLAMA_MOE_AUTOTUNE_COST_ROUNDS", 16, 0, 32);
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
        emit({{"event", "start"}, {"version", 3}, {"mode", mode}, {"spec_type", sequential ? "draft-mtp" : "draft-dflash"}, {"max_draft", max_draft},
              {"collection_only", !adaptive}, {"configured", configured.json()}, {"window_rounds", window},
              {"sample_tokens", sampling.interval}, {"sampling_unit", online ? "output_tokens" : "decode_rounds"},
              {"search_at", adaptive ? "request_end" : "disabled"}, {"frequency_scope", "sampled_rounds"},
              {"cost_round_budget", cost_limit}, {"built_prediction_width", max_width},
              {"prediction_semantics", "runtime prefix; predictor graph stays at built width"},
              {"cache_model", "one-round counterfactual from observed cache; no exact cross-threshold trajectory"}});
    }

    void emit(const tune_json & j) {
        output << j.dump() << '\n';
        if (!output) throw std::runtime_error("cannot write autotune telemetry");
    }

    std::string identity(const std::vector<ggml_moe_tune_layer> & layers) {
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
        for (const auto & l : layers) id += "|" + std::to_string(l.layer) + ":" + std::to_string(l.slots) + ":" + shape_key(l);
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
        if (j.at("version") != 1 || j.at("signature") != signature) {
            emit({{"event", "profile"}, {"status", "incompatible"}});
            return;
        }
        model.restore(j); loaded = true;
        emit({{"event", "profile"}, {"status", "reused"}, {"values", model.json(signature)}});
    }

    void learn(const round_sample & sample) {
        const int b = sample.kept + 1;
        double accounted = 0;
        bool complete = true;
        for (const auto & l : sample.layers) {
            auto & c = model.costs[shape_key(l)];
            if (executing.width == 0 && l.loads >= 4) {
                c.transfer.add(l.gate_us / l.loads);
                c.issue.add(l.fill_us / l.loads);
            }
            if (l.service_us < 0) { complete = false; continue; }
            accounted += l.gate_us + l.service_us;
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
        if (complete && executing.width == 0 && executing.cpu == 0) model.base[b].add(std::max(0.0, sample.verify_us - accounted));
    }


    void report(const char * event) {
        if (!request) return;
        const int64_t started = ggml_time_us();
        emit({{"event", event}, {"request", request}, {"rounds", round}, {"full_rounds", full_seen},
              {"sampled_rounds", sampled_rounds}, {"sample_tokens", sampling.interval}, {"output_tokens", sampling.tokens},
              {"search_ms", search_us / 1000.0},
              {"undrafted_steps", undrafted_steps},
              {"cost_probe_rounds", cost_done}, {"cost_probe_ms", calibration_us / 1000.0},
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
        tune_json observed = tune_json::array();
        for (const auto & entry : frequencies) observed.push_back({{"layer", entry.first}, {"counts", entry.second}});
        emit({{"event", "expert_frequencies"}, {"request", request}, {"rounds", round}, {"sampled_rounds", sampled_rounds},
              {"sample_tokens", sampling.interval}, {"frequency_scope", "sampled_rounds"}, {"layers", observed}});
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

void common_moe_autotune::begin(int prompt_tokens) {
    auto & s = *state;
    if (s.active) end(s.sampling.tokens, false);
    s.active = true;
    ++s.request; s.round = 0; s.sampled_rounds = 0; s.full_seen = 0; s.cost_done = 0;
    s.sampling.tokens = s.sampling.next = 0; s.samples.clear(); s.search_us = 0;
    s.frequencies.clear(); s.undrafted_steps.clear(); s.pending = false;
    // Workload observations never cross request boundaries.
    s.signature.clear(); s.loaded = false; s.model = {};
    s.prompt_tokens = prompt_tokens; s.request_start = ggml_time_us();
    s.calibration_us = s.control_us = 0; s.measured_us = 0; s.measured_tokens = 0;
    s.emit({{"event", "request"}, {"request", s.request}, {"prompt_tokens", prompt_tokens}, {"configured", s.configured.json()}});
}

bool common_moe_autotune::prepare() {
    auto & s = *state;
    if (s.pending) throw std::runtime_error("unfinished autotune round; use single-slot llama-server without checkpoint replay");
    s.pending = true;
    s.collecting = !s.online || s.sampling.due();
    if (!s.collecting) return false;
    s.round_start = ggml_time_us();
    s.executing = s.online ? s.configured : configuration{};
    s.probing = false; s.force_keep = -1;
    // Warm each service shape once before measuring it.
    std::vector<int> sizes;
    for (int b : {1, 2, 4, 8}) if (b <= s.max_draft + 1) sizes.push_back(b);
    if (sizes.back() != s.max_draft + 1) sizes.push_back(s.max_draft + 1);
    if (!s.online && !s.loaded && s.round >= 8 && s.cost_done < std::min(s.cost_limit, 4 * int(sizes.size()))) {
        const int cell = s.cost_done / 2;
        s.executing = {0, 0, cell < int(sizes.size()) ? 0 : 4};
        s.force_keep = sizes[cell % sizes.size()] - 1;
        s.probing = true;
    }
    s.profiled = s.online || (s.probing ? s.cost_done % 2 == 1 : s.round % 8 == 0);
    if (!llama_moe_tune_begin(s.target, s.executing.width, s.executing.cpu, s.profiled, s.online)) throw std::runtime_error("autotune needs pinned CUDA expert slots, legacy CPU diversion, and LFU");
    s.sampling.sampled();
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
    s.current.kept = keep;
    s.verify_start = ggml_time_us();
    s.current.draft_us = s.verify_start - s.round_start;
    return keep;
}

void common_moe_autotune::finish(int accepted) {
    auto & s = *state;
    if (!s.pending) {
        // The last output token can use a normal decode without a draft.
        s.undrafted_steps.push_back(s.round++);
        s.sampling.advance(accepted);
        return;
    }
    if (!s.collecting) {
        s.pending = false;
        ++s.round;
        s.sampling.advance(accepted);
        return;
    }
    const int64_t finished = ggml_time_us();
    const int64_t control_started = finished;
    if (accepted < 0 || accepted > s.current.kept) throw std::runtime_error("invalid autotune acceptance count");
    s.current.verify_us = finished - s.verify_start;
    s.current.accepted = accepted;
    s.current.round = s.round;
    s.current.layers.resize(128);
    const int count = llama_moe_tune_end(s.target, s.current.layers.data(), s.current.layers.size());
    if (count <= 0) throw std::runtime_error("missing autotune layer telemetry");
    s.current.layers.resize(count);
    for (const auto & l : s.current.layers) if (l.n_ids != (s.current.kept + 1) * l.top_k) throw std::runtime_error("autotune verification shape mismatch");
    if (s.signature.empty()) { s.signature = s.identity(s.current.layers); s.load_profile(); }
    if (s.round > 1 && (!s.probing || s.cost_done % 2 == 1)) s.learn(s.current);
    const bool full = s.current.kept == int(s.current.confidence.size()) && s.current.kept == s.max_draft;
    // Threshold-selected full drafts bias acceptance statistics. Use untruncated rounds.
    const bool reference = full && s.executing.threshold == 0;
    const bool observed_end = int(s.current.confidence.size()) == s.max_draft || s.current.kept < int(s.current.confidence.size());
    const bool search_sample = observed_end && s.current.kept == prefix_length(s.current.confidence, s.executing.threshold);
    if (reference) ++s.full_seen;
    int loads = 0, prefetches = 0, assignments = 0;
    double gate = 0, service = 0;
    for (const auto & l : s.current.layers) {
        auto & counts = s.frequencies[l.layer];
        counts.resize(l.experts);
        for (int i = 0; i < l.n_ids; ++i) ++counts[l.ids[i]];
        loads += l.loads; prefetches += l.prefetches; assignments += l.assignments;
        gate += l.gate_us; service += std::max(0.0, l.service_us);
    }
    const double duration = s.current.draft_us + s.current.verify_us;
    s.measured_us += duration; s.measured_tokens += accepted + 1;
    if (s.probing) { ++s.cost_done; s.calibration_us += finished - s.round_start; }
    tune_json record = {{"event", "round"}, {"request", s.request}, {"round", s.round}, {"position", s.position},
            {"output_tokens_before", s.sampling.tokens}, {"next_sample_tokens", s.sampling.next},
            {"config", s.executing.json()}, {"confidence", s.current.confidence}, {"kept", s.current.kept},
            {"max_draft", s.max_draft}, {"draft_prefix_us", s.current.draft_prefix_us},
            {"accepted", accepted}, {"right_censored", !full && accepted == s.current.kept}, {"full", full}, {"workload_sample", reference}, {"search_sample", search_sample},
            {"cost_probe", s.probing}, {"profiled", s.profiled}, {"draft_ms", s.current.draft_us / 1000},
            {"verify_ms", s.current.verify_us / 1000}, {"gate_ms", gate / 1000},
            {"expert_service_ms", s.profiled ? tune_json(service / 1000) : tune_json(nullptr)},
            {"demand_copies", loads}, {"prefetch_copies", prefetches}, {"cpu_assignments", assignments}};
    if (s.detail || (s.online && s.profiled)) {
        record["layers"] = tune_json::array();
        for (const auto & l : s.current.layers) {
            record["layers"].push_back({{"layer", l.layer}, {"shape", shape_key(l)}, {"slots", l.slots},
                {"experts", l.experts}, {"top_k", l.top_k}, {"width", l.width},
                {"bytes", std::vector<uint64_t>(l.bytes, l.bytes + 3)}, {"types", std::vector<int>(l.types, l.types + 3)},
                {"owners", std::vector<int>(l.owners, l.owners + l.slots)},
                {"lfu", std::vector<uint64_t>(l.freq, l.freq + l.experts)},
                {"prefetchable", std::vector<bool>(l.prefetchable, l.prefetchable + l.slots)},
                {"actual", std::vector<int>(l.ids, l.ids + l.n_ids)},
                {"predicted", std::vector<int>(l.predicted, l.predicted + l.n_pred)},
                {"loads", l.loads}, {"prefetches", l.prefetches}, {"cpu_assignments", l.assignments},
                {"gate_us", l.gate_us}, {"fill_us", l.fill_us}, {"prefetch_us", l.prefetch_us},
                {"cpu_us", l.cpu_us}, {"service_us", l.service_us >= 0 ? tune_json(l.service_us) : tune_json(nullptr)}});
        }
    }
    s.emit(record);
    if (s.adaptive && search_sample) {
        s.samples.push_back(s.current);
        while (s.samples.size() > size_t(s.window)) s.samples.pop_front();
    }
    s.pending = false; ++s.round; ++s.sampled_rounds;
    s.sampling.advance(accepted);
    s.control_us += ggml_time_us() - control_started;
    if (s.sampled_rounds % s.window == 0) s.report("window");
}

void common_moe_autotune::end(uint64_t output_tokens, bool completed) {
    auto & s = *state;
    if (!s.active) return;
    if (completed && s.pending) throw std::runtime_error("unfinished autotune verification at request end");
    // EOS or a stop string can end output inside an accepted batch.
    if (output_tokens < s.sampling.tokens && !s.samples.empty() && s.samples.back().round == s.round - 1) s.samples.pop_back();
    s.sampling.tokens = output_tokens;
    if (s.adaptive && completed) {
        const int64_t started = ggml_time_us();
        tune_json result = {{"event", "recommendation"}, {"samples", s.samples.size()},
            {"supported_candidates", 0}, {"selected", nullptr}, {"search_ms", 0.0},
            {"status", "insufficient_samples"}};
        if (s.samples.size() >= 8) {
            result = s.model.rank(s.samples, s.max_width);
            result["status"] = result["selected"].is_null() ? "missing_cost_or_prefix_coverage" : "selected";
        }
        result["request"] = s.request;
        result["previous_config"] = s.configured.json();
        result["applies_to"] = "next_request";
        if (!result["selected"].is_null()) {
            const auto & cfg = result["selected"];
            s.configured = {cfg.at("npred"), cfg.at("threshold"), cfg.at("cpu_cap")};
        }
        result["next_config"] = s.configured.json();
        s.emit(result);
        s.search_us = ggml_time_us() - started;
        LOG_INF("MoE autotune request end: request=%d search=%.3f ms next=%s\n", s.request, s.search_us / 1000.0, s.configured.json().dump().c_str());
    }
    s.report(completed ? "request_end" : "request_cancelled");
    s.output.flush();
    s.active = false;
}
