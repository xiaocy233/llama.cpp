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

static round_sample read_round(const tune_json & j) {
    round_sample sample;
    sample.round = bounded(j.at("round"), 0, 1000000000);
    sample.kept = bounded(j.at("kept"), 7, 7);
    sample.accepted = bounded(j.at("accepted"), 0, 7);
    if (j.at("confidence").size() != 7) throw std::runtime_error("expected seven draft confidences");
    for (const auto & p : j.at("confidence")) {
        const double v = duration(p);
        if (v > 1) throw std::runtime_error("invalid confidence");
        sample.confidence.push_back(float(v));
    }
    sample.draft_us = duration(j.at("draft_ms")) * 1000;
    sample.verify_us = duration(j.at("verify_ms")) * 1000;
    const auto & layers = j.at("layers");
    if (layers.empty() || layers.size() > 128) throw std::runtime_error("invalid layer count");
    for (const auto & v : layers) {
        ggml_moe_tune_layer l{};
        l.layer = bounded(v.at("layer"), 0, 1023);
        l.experts = bounded(v.at("experts"), 1, 256);
        l.slots = bounded(v.at("slots"), 1, l.experts);
        l.top_k = bounded(v.at("top_k"), 1, 32);
        l.width = bounded(v.at("width"), 1, 1000000);
        if (v.at("bytes").size() != 3 || v.at("types").size() != 3 || v.at("owners").size() != size_t(l.slots) ||
                v.at("prefetchable").size() != size_t(l.slots) || v.at("lfu").size() != size_t(l.experts) ||
                v.at("actual").size() != size_t(8 * l.top_k) || v.at("predicted").size() > 256 || v.at("predicted").size() % 8) {
            throw std::runtime_error("invalid layer dimensions");
        }
        for (int m = 0; m < 3; ++m) {
            l.bytes[m] = bounded(v.at("bytes").at(m), 0, 1073741824);
            l.types[m] = bounded(v.at("types").at(m), 0, GGML_TYPE_COUNT - 1);
        }
        std::array<bool, 256> owners{};
        for (int i = 0; i < l.slots; ++i) {
            l.owners[i] = bounded(v.at("owners").at(i), -1, l.experts - 1);
            l.prefetchable[i] = v.at("prefetchable").at(i).get<bool>();
            if (l.owners[i] >= 0) {
                if (owners[l.owners[i]]) throw std::runtime_error("duplicate cache owner");
                owners[l.owners[i]] = true;
            }
        }
        for (int e = 0; e < l.experts; ++e) l.freq[e] = bounded(v.at("lfu").at(e), 0, 2147483647);
        for (const auto & id : v.at("actual")) l.ids[l.n_ids++] = bounded(id, 0, l.experts - 1);
        for (const auto & id : v.at("predicted")) l.predicted[l.n_pred++] = bounded(id, 0, l.experts - 1);
        if (v.at("shape") != shape_key(l)) throw std::runtime_error("inconsistent expert shape");
        sample.layers.push_back(l);
    }
    return sample;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s TELEMETRY.jsonl RESULT.json [REQUEST_ID]\n", argv[0]);
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    try {
        if (std::ifstream(argv[2]).good()) throw std::runtime_error("result file already exists");
        const int64_t started = ggml_time_us();
        int requested = 0;
        if (argc == 4) requested = bounded(tune_json::parse(argv[3]), 1, 1000000000);
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot open telemetry");
        latency_model model;
        std::deque<round_sample> samples;
        int request = 0, width = 24, window = 32;
        bool have_costs = false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.size() > 16 * 1024 * 1024) throw std::runtime_error("trace record too large");
            const auto j = tune_json::parse(line);
            const std::string event = j.at("event");
            if (event == "start") {
                width = bounded(j.at("built_prediction_width"), 0, 24);
                window = bounded(j.at("window_rounds"), 16, 128);
                continue;
            }
            if (event == "request") {
                const int next = bounded(j.at("request"), 1, 1000000000);
                if (!requested || requested == next) {
                    request = next; samples.clear(); model = {}; have_costs = false;
                }
                continue;
            }
            if (!request || (j.contains("request") && j.at("request") != request)) continue;
            if (event == "round" && j.value("workload_sample", false) && j.contains("layers")) {
                samples.push_back(read_round(j));
                while (samples.size() > size_t(window)) samples.pop_front();
            }
            if (j.contains("costs")) { model.restore(j.at("costs")); have_costs = true; }
        }
        if (!have_costs) throw std::runtime_error("no completed cost snapshot for the requested workload");
        auto result = model.rank(samples, width);
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
