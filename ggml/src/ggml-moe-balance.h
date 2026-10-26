#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

static constexpr int ggml_moe_calibration_threads = 8;
static constexpr std::array<int, 3> ggml_moe_calibration_counts = {{1, 5, 10}};
static constexpr const char * ggml_moe_profile_version = "ggml-moe-latency-v2";

struct ggml_moe_cpu_cost {
    double a_us = 0;
    double b_us = 0;

    bool valid() const {
        return a_us > 0 && b_us >= 0 && std::isfinite(a_us) && std::isfinite(b_us);
    }

    double time_us(int assignments) const {
        return assignments > 0 ? a_us * assignments + b_us : 0;
    }
};

inline bool ggml_moe_cpu_cost_fit(const std::array<double, 3> & times_us, ggml_moe_cpu_cost & cost) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < times_us.size(); ++i) {
        if (!(times_us[i] > 0 && std::isfinite(times_us[i]))) return false;
        const double x = ggml_moe_calibration_counts[i];
        sx += x;
        sy += times_us[i];
        sxx += x * x;
        sxy += x * times_us[i];
    }
    const double n = times_us.size();
    cost.a_us = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    cost.b_us = (sy - cost.a_us * sx) / n;
    if (!(cost.a_us > 0)) return false;
    if (cost.b_us < 0) {
        cost.b_us = 0;
        cost.a_us = sxy / sxx;
    }
    return cost.valid();
}

struct ggml_moe_balance_item {
    int expert;
    int assignments;
    uint64_t recency;
};

inline std::vector<int> ggml_moe_balance_select(
        std::vector<ggml_moe_balance_item> misses, int gpu_slots,
        const ggml_moe_cpu_cost & cpu_cost, size_t expert_bytes, double h2d_gbs) {
    std::stable_sort(misses.begin(), misses.end(), [](const auto & a, const auto & b) {
        return a.assignments != b.assignments ? a.assignments < b.assignments : a.recency < b.recency;
    });
    const int n = misses.size();
    const int min_count = n - std::max(0, std::min(n, gpu_slots));
    int best_count = min_count;
    double best_us = std::numeric_limits<double>::infinity();
    int assignments = 0;
    for (int count = 0; count <= n; ++count) {
        if (count > 0) assignments += misses[count - 1].assignments;
        if (count < min_count) continue;
        const double h2d_us = (n - count) * (double) expert_bytes / (h2d_gbs * 1000.0);
        const double elapsed_us = std::max(h2d_us, cpu_cost.time_us(assignments));
        if (elapsed_us <= best_us) {
            best_us = elapsed_us;
            best_count = count;
        }
    }
    std::vector<int> result;
    for (int i = 0; i < best_count; ++i) result.push_back(misses[i].expert);
    return result;
}
