#include "moe-autotune-model.h"

#include <cstdio>
#include <fstream>
#include <iostream>

using namespace moe_autotune_model;

static int bounded(const tune_json & value, int low, int high) {
    if (!value.is_number_integer()) throw std::runtime_error("expected integer");
    const auto n = value.get<int64_t>();
    if (n < low || n > high) throw std::runtime_error("trace integer out of range");
    return int(n);
}

static double duration(const tune_json & value) {
    const double n = value.get<double>();
    if (!std::isfinite(n) || n < 0) throw std::runtime_error("invalid trace duration");
    return n;
}

static round_sample read_round(const tune_json & j, int max_draft, bool sequential) {
    round_sample sample;
    sample.phase = bounded(j.value("execution_phase", tune_json(-1)), -1, 2);
    sample.round = bounded(j.at("round"), 0, 1000000000);
    sample.max_draft = max_draft;
    if (j.contains("max_draft") && bounded(j.at("max_draft"), 1, 7) != max_draft) throw std::runtime_error("inconsistent draft limit");
    sample.kept = bounded(j.at("kept"), 0, max_draft);
    sample.position = bounded(j.at("position"), 0, 1000000000);
    const auto & cfg = j.at("config");
    sample.executed.width = bounded(cfg.at("npred"), 0, 24);
    sample.executed.cpu = bounded(cfg.at("cpu_cap"), 0, 8);
    sample.executed.threshold = duration(cfg.at("threshold"));
    if (sample.executed.threshold > 1) throw std::runtime_error("invalid threshold");
    sample.accepted = bounded(j.at("accepted"), 0, sample.kept);
    if (j.at("confidence").size() < size_t(sample.kept) || j.at("confidence").size() > size_t(max_draft) || j.at("confidence").empty()) throw std::runtime_error("invalid draft confidence vector");
    for (const auto & p : j.at("confidence")) {
        const double v = duration(p);
        if (v > 1) throw std::runtime_error("invalid confidence");
        sample.confidence.push_back(float(v));
    }
    sample.draft_us = duration(j.at("draft_ms")) * 1000;
    sample.verify_us = duration(j.at("verify_ms")) * 1000;
    if (sequential) {
        const auto & prefix = j.at("draft_prefix_us");
        if (!prefix.is_array() || prefix.size() != sample.confidence.size()) throw std::runtime_error("invalid sequential draft timing dimensions");
        double previous = 0;
        for (const auto & value : prefix) {
            const double us = duration(value);
            if (us < previous || us > sample.draft_us + 1e-6) throw std::runtime_error("invalid sequential draft timing order");
            sample.draft_prefix_us.push_back(us);
            previous = us;
        }
    }
    const int batch = sample.kept + 1;
    const auto & layers = j.at("layers");
    if (layers.empty() || layers.size() > 128) throw std::runtime_error("invalid layer count");
    for (const auto & v : layers) {
        layer_sample layer;
        auto & l = layer.stats;
        l.layer = bounded(v.at("layer"), 0, 1023);
        for (const auto & previous : sample.layers) if (previous.stats.layer == l.layer) throw std::runtime_error("duplicate layer in trace");
        l.experts = bounded(v.at("experts"), 1, std::numeric_limits<int>::max());
        l.slots = bounded(v.at("slots"), 1, l.experts);
        l.top_k = bounded(v.at("top_k"), 1, std::min(l.experts, GGML_MOE_TUNE_IDS / batch));
        l.width = bounded(v.at("width"), 1, 1000000);
        if (v.at("bytes").size() != 3 || v.at("types").size() != 3 || v.at("owners").size() != size_t(l.slots) ||
                v.at("prefetchable").size() != size_t(l.slots) || v.at("lfu").size() != size_t(l.experts) ||
                v.at("actual").size() != size_t(batch * l.top_k) || v.at("predicted").size() > GGML_MOE_TUNE_IDS || v.at("predicted").size() % batch) {
            throw std::runtime_error("invalid layer dimensions");
        }
        for (int m = 0; m < 3; ++m) {
            l.bytes[m] = bounded(v.at("bytes").at(m), 0, 1073741824);
            l.types[m] = bounded(v.at("types").at(m), 0, GGML_TYPE_COUNT - 1);
        }
        layer.owners.resize(l.slots);
        layer.freq.resize(l.experts);
        layer.prefetchable.resize(l.slots);
        std::vector<bool> owners(l.experts);
        for (int i = 0; i < l.slots; ++i) {
            layer.owners[i] = bounded(v.at("owners").at(i), -1, l.experts - 1);
            layer.prefetchable[i] = v.at("prefetchable").at(i).get<bool>();
            if (layer.owners[i] >= 0) {
                if (owners[layer.owners[i]]) throw std::runtime_error("duplicate cache owner");
                owners[layer.owners[i]] = true;
            }
        }
        for (int e = 0; e < l.experts; ++e) layer.freq[e] = bounded(v.at("lfu").at(e), 0, 2147483647);
        for (const auto & id : v.at("actual")) l.ids[l.n_ids++] = bounded(id, 0, l.experts - 1);
        for (const auto & id : v.at("predicted")) l.predicted[l.n_pred++] = bounded(id, 0, l.experts - 1);
        if (v.at("shape") != shape_key(l)) throw std::runtime_error("inconsistent expert shape");
        l.loads = bounded(v.at("loads"), 0, GGML_MOE_TUNE_IDS);
        l.prefetches = bounded(v.at("prefetches"), 0, GGML_MOE_TUNE_IDS);
        l.assignments = bounded(v.at("cpu_assignments"), 0, GGML_MOE_TUNE_IDS);
        l.gate_us = duration(v.at("gate_us"));
        l.service_us = v.at("service_us").is_null() ? -1 : duration(v.at("service_us"));
        sample.layers.push_back(std::move(layer));
    }
    return sample;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s TELEMETRY.jsonl RESULT.json [REQUEST_ID] [--observed-threshold]\n", argv[0]);
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    try {
        if (std::ifstream(argv[2]).good()) throw std::runtime_error("result file already exists");
        const int64_t started = ggml_time_us();
        int requested = 0;
        bool own_threshold = false;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--observed-threshold") own_threshold = true;
            else if (!requested) requested = bounded(tune_json::parse(argv[i]), 1, 1000000000);
            else throw std::runtime_error("duplicate request ID");
        }
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot open telemetry");
        latency_model model, observed;
        std::deque<round_sample> samples;
        std::map<int, layer_sample> observed_layers;
        int observed_round = -2;
        int request = 0, width = 24, max_draft = 7, previous_batch = 0, phase = 0;
        size_t workload_bytes = 0;
        int64_t last_output_after = 0;
        bool sequential = false;
        bool have_costs = false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("trace record too large");
            auto j = tune_json::parse(line);
            const std::string event = j.at("event");
            if (event == "start") {
                width = bounded(j.at("built_prediction_width"), 0, 24);
                max_draft = bounded(j.value("max_draft", tune_json(7)), 1, 7);
                const std::string type = j.value("spec_type", "draft-dflash");
                if (type != "draft-dflash" && type != "draft-mtp") throw std::runtime_error("unsupported draft type");
                sequential = type == "draft-mtp";
                continue;
            }
            if (event == "request") {
                const int next = bounded(j.at("request"), 1, 1000000000);
                if (!requested || requested == next) {
                    request = next; samples.clear(); observed_layers.clear(); observed_round = -2; model = {}; observed = {}; have_costs = false;
                    previous_batch = 0; phase = 0; workload_bytes = 0; last_output_after = 0;
                }
                continue;
            }
            if (!request || (j.contains("request") && j.at("request") != request)) continue;
            if (event == "route") {
                if (bounded(j.at("round"), 0, 1000000000) != observed_round + 1 || j.at("layers").size() != observed_layers.size()) throw std::runtime_error("route trace lacks a contiguous state anchor");
                for (auto & v : j["layers"]) {
                    const int id = bounded(v.at("layer"), 0, 1023);
                    const auto it = observed_layers.find(id);
                    if (it == observed_layers.end()) throw std::runtime_error("route layer lacks a state anchor");
                    const auto & prior = it->second;
                    const auto l = prior.view();
                    v["shape"] = shape_key(l); v["experts"] = l.experts; v["slots"] = l.slots;
                    v["top_k"] = l.top_k; v["width"] = l.width;
                    v["bytes"] = std::vector<uint64_t>(l.bytes, l.bytes + 3);
                    v["types"] = std::vector<int>(l.types, l.types + 3);
                    v["owners"] = prior.owners; v["lfu"] = prior.freq;
                    v["prefetchable"] = tune_json::array();
                    for (auto eligible : prior.prefetchable) v["prefetchable"].push_back(bool(eligible));
                    v["loads"] = 0; v["prefetches"] = 0; v["cpu_assignments"] = 0;
                    v["gate_us"] = 0; v["service_us"] = nullptr;
                }
            }
            if ((event == "round" || event == "route") && j.contains("layers")) {
                auto sample = read_round(j, max_draft, sequential);
                phase = sample.round == observed_round + 1 ? next_execution_phase(sample.kept + 1, previous_batch, phase) : 0;
                previous_batch = sample.kept + 1;
                if (sample.phase >= 0 && sample.phase != phase) throw std::runtime_error("inconsistent execution phase");
                sample.phase = phase;
                last_output_after = bounded(j.value("output_tokens_before", tune_json(0)), 0, 1000000000) + sample.accepted + 1;
                if (event == "round" && sample.round == observed_round + 1) {
                    for (const auto & layer : sample.layers) {
                        const auto prior = observed_layers.find(layer.stats.layer);
                        if (prior != observed_layers.end() && (prior->second.owners != layer.owners || prior->second.freq != layer.freq || prior->second.prefetchable != layer.prefetchable)) throw std::runtime_error("cache replay differs from the next observed snapshot");
                    }
                }
                observed_layers.clear();
                for (const auto & layer : sample.layers) {
                    cache_state state(layer);
                    const auto work = state.advance(layer.view(), sample.kept + 1, sample.executed);
                    if (!work.fits) throw std::runtime_error("observed GPU experts exceed cache capacity");
                    if (event == "round" && (work.miss != layer.stats.loads || work.prefetch != layer.stats.prefetches || work.assignments != layer.stats.assignments)) {
                        throw std::runtime_error("replayed expert work differs at round " + std::to_string(sample.round) + " layer " + std::to_string(layer.stats.layer));
                    }
                    auto next = layer;
                    next.owners = std::move(state.owners); next.freq = std::move(state.freq); next.prefetchable = std::move(state.prefetchable);
                    observed_layers.emplace(layer.stats.layer, std::move(next));
                }
                observed_round = sample.round;
                if (sample.round > 1 && j.value("profiled", false) && j.value("warm_cost", true)) observed.observe_residual(sample);
                if (j.value("search_sample", j.value("workload_sample", false))) {
                    if (samples.size() == max_workload_rounds || sample.storage_bytes() > max_workload_bytes - workload_bytes) throw std::runtime_error("workload exceeds the search memory limit");
                    workload_bytes += sample.storage_bytes();
                    samples.push_back(std::move(sample));
                }
            }
            if (event == "request_end" && j.at("output_tokens").get<int64_t>() < last_output_after && !samples.empty() && samples.back().round == observed_round) {
                workload_bytes -= samples.back().storage_bytes();
                samples.pop_back();
            }
            if (j.contains("costs")) {
                model.restore(j.at("costs"));
                if (j.at("costs").at("version") == 1) {
                    model.cpu_residual = observed.cpu_residual;
                    model.cpu_layers = observed.cpu_layers;
                }
                have_costs = true;
            }
            if (requested && event == "request_end") break;
        }
        if (!have_costs) throw std::runtime_error("no completed cost snapshot for the requested workload");
        auto result = model.rank(samples, width, own_threshold);
        result["request"] = request;
        result["source"] = argv[1];
        result["load_and_search_ms"] = (ggml_time_us() - started) / 1000.0;
        std::ofstream output(argv[2]); output << result.dump(2) << '\n'; output.close();
        if (!output) throw std::runtime_error("cannot write search result");
        std::cout << result.dump(2) << '\n';
        return result.at("supported_candidates") == 0 ? 2 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "MoE search: %s\n", e.what());
        return 1;
    }
}
