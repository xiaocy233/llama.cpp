#pragma once

#include "draft-acceptance-profile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

// DAP1 stores the calibrated FP32 head without a training-runtime dependency.
struct common_draft_acceptance_model {
    uint32_t inputs, width, hidden, drafts;
    float threshold, scale, bias;
    std::vector<float> mean, stddev, w1, b1, w2, b2, normalized;

    explicit common_draft_acceptance_model(const char * path) {
        std::ifstream file(path, std::ios::binary);
        uint32_t header[6] = {};
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        if (!file || header[0] != 0x31504144 || header[1] != 1 || header[2] != 9 + header[4] || header[4] > 65536 || header[3] != 64 || header[5] != 7) {
            throw std::runtime_error("invalid DFlash acceptance model header");
        }
        inputs = header[2]; width = header[3]; hidden = header[4]; drafts = header[5];
        auto read = [&](std::vector<float> & data, size_t count) {
            data.resize(count);
            file.read(reinterpret_cast<char *>(data.data()), count * sizeof(float));
            if (!file || !std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("invalid DFlash acceptance model weights");
            }
        };
        std::vector<float> calibration;
        read(calibration, 3);
        threshold = calibration[0]; scale = calibration[1]; bias = calibration[2];
        read(mean, inputs); read(stddev, inputs);
        read(w1, width * inputs); read(b1, width); read(w2, width); read(b2, 1);
        if (file.peek() != std::ifstream::traits_type::eof() || threshold < 0 || threshold > 1 || scale <= 0 || !std::all_of(stddev.begin(), stddev.end(), [](float v) { return v > 0; })) {
            throw std::runtime_error("invalid DFlash acceptance model calibration");
        }
        normalized.resize(inputs);
    }

    float probability(const float * features) {
        DFLASH_PROFILE("dflash.head", inputs);
        for (uint32_t i = 0; i < inputs; ++i) {
            if (!std::isfinite(features[i])) {
                throw std::runtime_error("non-finite DFlash acceptance feature");
            }
            normalized[i] = std::max(-10.0f, std::min(10.0f, (features[i] - mean[i]) / stddev[i]));
        }
        float output = b2[0];
        for (uint32_t j = 0; j < width; ++j) {
            float value = b1[j];
            for (uint32_t i = 0; i < inputs; ++i) {
                value += w1[j * inputs + i] * normalized[i];
            }
            output += w2[j] * value / (1.0f + std::exp(-value));
        }
        return 1.0f / (1.0f + std::exp(-(output * scale + bias)));
    }
};

// Buffered audit rows: request, position, generated, verified, accepted, predictor_us.
struct common_draft_acceptance_audit {
    FILE * file;
    uint32_t request = 0, position = 0, generated = 0, kept = 0;
    int64_t predictor_us = 0;
    bool pending = false;

    explicit common_draft_acceptance_audit(const char * path) : file(fopen(path, "wx")) {
        if (!file) {
            throw std::runtime_error("cannot create DFlash acceptance audit (path must be new)");
        }
    }
    ~common_draft_acceptance_audit() {
        finish(-1);
        fclose(file);
    }
    void finish(int accepted) {
        if (pending) {
            fprintf(file, "[%u,%u,%u,%u,%d,%lld]\n", request, position, generated, kept, accepted, (long long) predictor_us);
            pending = false;
        }
    }
    void begin() {
        finish(-1);
        fflush(file);
        ++request;
    }
    void draft(uint32_t pos, uint32_t count, uint32_t retained, int64_t elapsed) {
        finish(-1);
        position = pos; generated = count; kept = retained; predictor_us = elapsed;
        pending = true;
        if (kept == 0) {
            finish(0);
        }
    }
};
