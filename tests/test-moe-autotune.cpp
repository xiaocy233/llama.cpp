#include "moe-autotune-model.h"
#include <cstdio>

using namespace moe_autotune_model;

static layer_sample layer() {
    layer_sample l;
    l.stats.layer = 0; l.stats.experts = 4; l.stats.slots = 2; l.stats.top_k = 1;
    l.stats.width = 1; l.stats.n_ids = 1; l.stats.ids[0] = 2;
    l.owners = {0, 1}; l.freq = {0, 1, 0, 0}; l.prefetchable = {1, 1};
    return l;
}

int main() {
    auto l = layer();
    cache_state cache(l);
    const auto first = cache.advance(l.view(), 1, {});
    GGML_ASSERT(first.miss == 1 && cache.owners[0] == 2 && cache.freq[2] == 1);
    const auto second = cache.advance(l.view(), 1, {});
    GGML_ASSERT(second.miss == 0 && cache.freq[2] == 2);
    auto cpu_cache = cache_state(l);
    const auto cpu_work = cpu_cache.advance(l.view(), 1, {0, 0, 4});
    GGML_ASSERT(cpu_work.miss == 0 && cpu_work.assignments == 1 && cpu_cache.owners == l.owners && cpu_cache.freq[2] == 0);
    auto predicted = l;
    predicted.stats.n_pred = 2; predicted.stats.predicted[0] = 0; predicted.stats.predicted[1] = 2;
    auto pf_cache = cache_state(predicted);
    const auto pf = pf_cache.advance(predicted.view(), 1, {8, 0, 0});
    GGML_ASSERT(pf.prefetch == 1 && pf.miss == 0 && pf_cache.owners[0] == 0 && pf_cache.owners[1] == 2);

    auto crowded = l;
    crowded.stats.n_ids = 3;
    crowded.stats.ids[0] = 0; crowded.stats.ids[1] = 1; crowded.stats.ids[2] = 2;
    GGML_ASSERT(!cache_state(crowded).advance(crowded.view(), 3, {}).fits);
    const auto diverted = cache_state(crowded).advance(crowded.view(), 3, {0, 0, 1});
    GGML_ASSERT(diverted.fits && diverted.assignments == 1 && diverted.miss == 0);
    crowded.stats.ids[2] = 0;
    GGML_ASSERT(cache_state(crowded).advance(crowded.view(), 3, {}).fits);

    latency_model model;
    for (int b = 1; b <= 8; ++b) model.base[b].add(100);
    auto & c = model.costs[shape_key(l.view())];
    for (int b = 1; b <= 8; ++b) c.gpu[b].add(10);
    c.transfer.add(100); c.issue.add(1); c.cpu_per_assignment.add(10); c.protocol.add(0);
    crowded.stats.ids[2] = 2;
    round_sample capacity_sample;
    capacity_sample.layers = {crowded};
    std::map<int, cache_state> capacity_states;
    GGML_ASSERT(!model.predict_round(capacity_sample, 3, {}, capacity_states).supported);
    round_sample sample;
    sample.layers = {l}; sample.kept = 0; sample.confidence = {0.1f}; sample.draft_us = 10;
    std::map<int, cache_state> states;
    GGML_ASSERT(!model.predict_round(sample, 1, {0, 0, 4}, states).supported);
    sample.executed = {0, 0, 4}; sample.verify_us = 310;
    sample.layers[0].stats.service_us = 10; sample.layers[0].stats.assignments = 1;
    model.observe_residual(sample);
    states.clear();
    const auto with_cpu = model.predict_round(sample, 1, {0, 0, 4}, states);
    GGML_ASSERT(with_cpu.supported && with_cpu.cpu_overhead == 200 && with_cpu.total() == 320);
    latency_model restored;
    restored.restore(model.json());
    GGML_ASSERT(restored.cpu_residual[1].value == 300);

    std::deque<round_sample> samples;
    sample.executed = {}; sample.accepted = 0; sample.kept = 1; sample.confidence = {0.1f};
    sample.layers[0].stats.n_ids = 2; sample.layers[0].stats.ids[1] = 2;
    for (int r = 0; r < 8; ++r) { sample.round = r; sample.position = r; samples.push_back(sample); }
    auto ranking = model.rank(samples, 0);
    GGML_ASSERT(ranking["selected"]["cpu_cap"] == 0);
    const auto & first_cfg = ranking["ranking"][0];
    GGML_ASSERT(first_cfg["cache_transitions"] == 7 && first_cfg["cache_blocks"] == 1);
    samples[4].round += 10;
    ranking = model.rank(samples, 0);
    GGML_ASSERT(ranking["ranking"][0]["cache_transitions"] == 5 && ranking["ranking"][0]["cache_blocks"] == 3);
    for (auto & s : samples) { s.round = s.position; s.accepted = 1; s.position *= 2; }
    ranking = model.rank(samples, 0);
    for (const auto & item : ranking["ranking"]) {
        if (item["config"]["threshold"] != 0) GGML_ASSERT(item["cache_transitions"] == 0);
    }
    for (int b = 1; b <= 8; ++b) { model.cpu_residual[b] = {1, 100}; model.cpu_layers[b] = {1, 1}; }
    for (auto & s : samples) { s.accepted = 0; s.position = s.round; }
    ranking = model.rank(samples, 0);
    GGML_ASSERT(ranking["selected"]["cpu_cap"] != 0);
    sample.draft_us = 35; sample.draft_prefix_us = {10, 20, 30};
    GGML_ASSERT(sample.draft_cost(0) == 15 && sample.draft_cost(1) == 25 && sample.draft_cost(3) == 35);
    sample.draft_prefix_us.clear();
    GGML_ASSERT(sample.draft_cost(0) == 35 && sample.draft_cost(7) == 35);

    latency_model prefetched_model;
    auto prefetched_sample = sample;
    prefetched_sample.executed = {8, 0, 4};
    prefetched_sample.kept = 0;
    prefetched_sample.verify_us = 310;
    prefetched_sample.layers[0].stats.gate_us = 50;
    prefetched_sample.layers[0].stats.service_us = 10;
    prefetched_sample.layers[0].stats.assignments = 1;
    prefetched_model.observe_residual(prefetched_sample);
    GGML_ASSERT(prefetched_model.cpu_residual[1].count == 1 && prefetched_model.cpu_residual[1].value == 250);
    prefetched_sample.layers[0].stats.assignments = 0;
    prefetched_model.observe_residual(prefetched_sample);
    GGML_ASSERT(prefetched_model.base[1].count == 1 && prefetched_model.base[1].value == 250);
    prefetched_sample.layers[0].stats.service_us = -1;
    prefetched_model.observe_residual(prefetched_sample);
    GGML_ASSERT(prefetched_model.base[1].count == 1 && prefetched_model.cpu_residual[1].count == 1);

    GGML_ASSERT(next_execution_phase(8, 0, 0) == 0);
    GGML_ASSERT(next_execution_phase(8, 8, 0) == 1);
    GGML_ASSERT(next_execution_phase(8, 8, 1) == 2);
    GGML_ASSERT(next_execution_phase(8, 8, 2) == 2);
    GGML_ASSERT(next_execution_phase(4, 8, 2) == 0);
    prefetched_sample.layers[0].stats.service_us = 10;
    prefetched_sample.phase = 1;
    prefetched_model.observe_residual(prefetched_sample);
    restored.restore(prefetched_model.json());
    GGML_ASSERT(restored.phase_costs && restored.phases[1].base[1].value == 250 && !restored.phases[2].base[1].count);
    auto invalid_profile = prefetched_model.json();
    invalid_profile["phases"][0]["cpu_layers"][0]["samples"] = 1;
    bool invalid = false;
    try { restored.restore(invalid_profile); } catch (const std::runtime_error &) { invalid = true; }
    GGML_ASSERT(invalid && restored.phases[1].base[1].value == 250);

    auto negative_residual = model;
    negative_residual.cpu_residual[1] = {1, 50};
    states.clear();
    const auto lower_residual = negative_residual.predict_round(sample, 1, {0, 0, 4}, states);
    GGML_ASSERT(lower_residual.supported && lower_residual.cpu_overhead == -50 && lower_residual.total() == 95);

    auto more_cpu_layers = sample;
    more_cpu_layers.layers.push_back(sample.layers.front());
    more_cpu_layers.layers.back().stats.layer = 1;
    states.clear();
    const auto whole_verify = negative_residual.predict_round(more_cpu_layers, 1, {0, 0, 4}, states);
    GGML_ASSERT(whole_verify.supported && whole_verify.base + whole_verify.cpu_overhead == 50 && whole_verify.total() == 105);

    auto staged = model;
    staged.phase_costs = true;
    for (int phase = 0; phase < 3; ++phase) for (int b = 1; b <= 8; ++b) {
        auto & table = staged.phases[phase];
        table.base[b] = {1, phase == 0 ? 100.0 : phase == 1 ? 300.0 : 50.0};
        table.cpu_residual[b] = table.base[b];
        table.cpu_layers[b] = {1, 1};
    }
    for (auto & observation : samples) {
        observation.draft_us = 0;
        observation.confidence = {1};
    }
    const auto staged_ranking = staged.rank(samples, 0);
    GGML_ASSERT(std::abs(staged_ranking["ranking"][0]["predicted_tpot_ms"].get<double>() - 0.110) < 1e-9);
    states.clear();
    GGML_ASSERT(!staged.predict_round(sample, 1, {}, states, nullptr, 3).supported);

    std::deque<round_sample> full_request;
    for (int r = 0; r < 96; ++r) {
        auto observation = samples.front();
        observation.round = observation.position = r;
        observation.draft_us = r < 64 ? 10 : 100;
        full_request.push_back(std::move(observation));
    }
    auto full_ranking = model.rank(full_request, 24);
    GGML_ASSERT(std::abs(full_ranking["ranking"][0]["predicted_tpot_ms"].get<double>() - 14.5 / 96) < 1e-9);
    auto repeated_ranking = model.rank(full_request, 24);
    full_ranking.erase("search_ms"); repeated_ranking.erase("search_ms");
    GGML_ASSERT(full_ranking == repeated_ranking);
    std::deque<round_sample> oversized(max_workload_rounds + 1);
    invalid = false;
    try { model.rank(oversized, 0); } catch (const std::runtime_error &) { invalid = true; }
    GGML_ASSERT(invalid);
    tune_json choices = tune_json::array();
    const std::array<configuration, 4> configs = {{{8, 0.6f, 4}, {8, 0, 4}, {8, 0.4f, 4}, {8, 0.8f, 4}}};
    for (size_t i : {3, 1, 0, 2}) choices.push_back({{"config", configs[i].json()}, {"predicted_tpot_ms", i == 3 ? 6.1 : 6.0 + i * 0.01}});
    auto branches = branch_candidates(choices);
    GGML_ASSERT(branches.size() == 4 && branches[0]["config"] == configs[1].json());
    auto incomplete = tune_json::array({choices.front()});
    GGML_ASSERT(branch_candidates(incomplete).size() == 4 && branch_candidates(incomplete, false).size() == 1);
    candidate_frontier frontier;
    frontier.seed(branches, {});
    GGML_ASSERT(frontier.candidates.size() == 5 && !frontier.ready() && frontier.next() == configuration{});
    frontier.observe({}, 1, 32);
    frontier.observe({}, -1, 128);
    GGML_ASSERT(frontier.completed == 0 && !frontier.warming);
    for (int n = 0; n < 5; ++n) {
        const auto cfg = frontier.next();
        GGML_ASSERT(!frontier.confirming);
        GGML_ASSERT(frontier.observe(cfg, (cfg == configs[2] ? 5500 : 6000) * 128, 128));
        frontier.advance();
    }
    GGML_ASSERT(frontier.confirming && !frontier.ready());
    while (!frontier.ready()) {
        const auto cfg = frontier.next();
        GGML_ASSERT(frontier.observe(cfg, (cfg == configs[2] ? 5500 : 6000) * 1023, 1023));
        frontier.advance();
    }
    GGML_ASSERT(frontier.next() == configs[2] && frontier.completed == 7);
    GGML_ASSERT(frontier.output_tokens == 128 && frontier.confirmation_tokens == 1023);
    GGML_ASSERT(!frontier.observe(configs[2], 500000, 256));
    GGML_ASSERT(!frontier.ready() && frontier.completed == 0);
    frontier.seed(branches, configs[0], 24, true);
    auto own = tune_json::array({{{"config", configs[0].json()}, {"predicted_tpot_ms", 6.0}},
                              {{"config", configuration{16, 0.6f, 4}.json()}, {"predicted_tpot_ms", 6.1}}});
    frontier.refine(own, configs[0], 6.2);
    GGML_ASSERT(frontier.branch_errors.count(0.6f) && frontier.candidates.size() == 5);
    frontier.refine(own, configs[0], 6.4);
    GGML_ASSERT(frontier.candidates.size() == 5);
    GGML_ASSERT(!frontier.observe(frontier.next(), 900000, 64));
    GGML_ASSERT(frontier.observe(frontier.next(), 900000, 128));
    frontier.advance();
    GGML_ASSERT(!frontier.observe(frontier.next(), 900000, 64));
    GGML_ASSERT(!frontier.observe(frontier.next(), 900000, 256));
    GGML_ASSERT(frontier.completed == 0 && !frontier.output_tokens && frontier.branch_errors.empty() && frontier.warming);
    const auto own_ranking = model.rank(full_request, 24, true);
    GGML_ASSERT(own_ranking["trajectory_scope"] == "observed_threshold");
    for (const auto & row : own_ranking["ranking"]) GGML_ASSERT(row["config"]["threshold"] == full_request.front().executed.threshold);
    for (auto & observation : full_request) observation.executed.threshold = 0.2f;
    const auto custom_threshold = model.rank(full_request, 24, true);
    GGML_ASSERT(!custom_threshold["ranking"].empty());
    for (const auto & row : custom_threshold["ranking"]) GGML_ASSERT(row["config"]["threshold"] == 0.2f);
    full_request[1].executed.threshold = 0.6f;
    GGML_ASSERT(model.rank(full_request, 24, true)["ranking"].empty());
    candidate_frontier guarded;
    guarded.seed(tune_json::array({{{"config", configs[0].json()}}}), {});
    for (int n = 0; n < 4; ++n) {
        const auto cfg = guarded.next();
        const double cost = cfg == configuration{} ? 6.0 : 5.9;
        GGML_ASSERT(guarded.observe(cfg, cost * 1000 * 128, 128));
        guarded.advance();
    }
    GGML_ASSERT(guarded.ready() && guarded.next() == configuration{});
    candidate_frontier changing_order;
    changing_order.seed(tune_json::array({{{"config", configs[0].json()}}, {{"config", configs[1].json()}}}), {});
    while (!changing_order.ready()) {
        const auto cfg = changing_order.next();
        const bool short_request = !changing_order.confirming;
        const int tokens = short_request ? 511 : 1023;
        const double cost = cfg == configuration{} ? 8.0 : cfg == configs[0] ? (short_request ? 5.0 : 6.0) : 5.1;
        GGML_ASSERT(changing_order.observe(cfg, cost * 1000 * tokens, tokens));
        changing_order.advance();
    }
    GGML_ASSERT(changing_order.next() == configs[1] && changing_order.completed == 6);
    GGML_ASSERT(changing_order.candidates[2].confirmation_ms == 5.1);
    candidate_frontier bounded_frontier;
    bounded_frontier.seed(branches, {}, 10);
    bounded_frontier.refine(own, configs[0], 6.2);
    GGML_ASSERT(bounded_frontier.candidates.size() == 5);
    candidate_frontier continuing;
    const configuration start{8, 0.4f, 0}, middle{8, 0.4f, 2}, finish{8, 0.4f, 4};
    continuing.seed(tune_json::array({{{"config", start.json()}}}), {});
    continuing.refine(tune_json::array({{{"config", start.json()}, {"predicted_tpot_ms", 7.0}},
                                       {{"config", middle.json()}, {"predicted_tpot_ms", 6.8}}}), start, 7.4);
    GGML_ASSERT(!continuing.needs_refinement(start) && continuing.needs_refinement(middle));
    continuing.refine(tune_json::array({{{"config", middle.json()}, {"predicted_tpot_ms", 6.8}},
                                       {{"config", finish.json()}, {"predicted_tpot_ms", 6.9}}}), middle, 7.2);
    GGML_ASSERT(continuing.candidates.size() == 4 && continuing.candidates.back().config == finish);
    GGML_ASSERT(!continuing.needs_refinement(middle) && continuing.needs_refinement(finish));
    candidate_frontier uncertain_branch;
    uncertain_branch.seed(tune_json::array({{{"config", start.json()}}}), {});
    GGML_ASSERT(uncertain_branch.observe({}, 7100 * 128, 128));
    uncertain_branch.advance();
    GGML_ASSERT(uncertain_branch.observe(start, 7350 * 128, 128));
    uncertain_branch.refine(tune_json::array({{{"config", start.json()}, {"predicted_tpot_ms", 6.88}},
                                             {{"config", middle.json()}, {"predicted_tpot_ms", 7.15}}}), start, 7.35);
    GGML_ASSERT(uncertain_branch.candidates.size() == 3 && uncertain_branch.candidates.back().config == middle);
    candidate_frontier full_length;
    full_length.seed(branches, {}, 24, false, false);
    for (int i = 0; i < 5; ++i) {
        const auto cfg = full_length.next();
        GGML_ASSERT(full_length.observe(cfg, (cfg == configs[2] ? 5500 : 6000) * 1023, 1023));
        full_length.advance();
    }
    GGML_ASSERT(full_length.ready() && full_length.completed == 5 && full_length.next() == configs[2]);
    for (const auto & c : full_length.candidates) GGML_ASSERT(c.trials == 1 && c.confirmation_ms == c.screening_ms);
    GGML_ASSERT(!full_length.observe(full_length.next(), 6000 * 511, 511));
    GGML_ASSERT(!full_length.ready() && full_length.completed == 0 && !full_length.output_tokens);
    cost_warmup warm;
    for (int path = 0; path < 2; ++path) for (int b : {1, 2, 4, 8}) {
        for (int call = 0; call < 3; ++call) GGML_ASSERT(!warm.observe(b, path));
    }
    for (int path = 0; path < 2; ++path) for (int b : {1, 2, 4, 8}) GGML_ASSERT(warm.observe(b, path));
    prefetched_model.warm_costs = true;
    GGML_ASSERT(prefetched_model.json()["version"] == 4);
    restored.restore(prefetched_model.json());
    GGML_ASSERT(restored.warm_costs && restored.phase_costs && restored.phases[1].base[1].value == 250);
    std::puts("MoE autotune cache, residual, coverage, frontier and warm calibration tests passed");
}
