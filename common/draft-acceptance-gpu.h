#pragma once

#include "draft-acceptance-model.h"
#include "../ggml/src/ggml-draft-acceptance.h"
#include "../src/llama-ext.h"

#include <cstring>

struct common_draft_acceptance_gpu {
    ggml_draft_acceptance implementation;
    ggml_backend_t backend = nullptr;

    static ggml_draft_acceptance_init_fn get_init(ggml_backend_dev_t device) {
        if (!device) {
            return nullptr;
        }
        return reinterpret_cast<ggml_draft_acceptance_init_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_draft_acceptance_init"));
    }

    static bool supported(const llama_model * model) {
        return llama_model_n_devices(model) == 1 && get_init(llama_model_get_device(model, 0));
    }

    ~common_draft_acceptance_gpu() {
        if (implementation.context) {
            implementation.free(implementation.context);
        }
    }

    ggml_draft_acceptance_result evaluate(llama_context * ctx, const common_draft_acceptance_model & model, const ggml_draft_acceptance_batch & batch) {
        ggml_tensor * logits = nullptr;
        ggml_tensor * hidden = nullptr;
        ggml_backend_t current = nullptr;
        if (!llama_get_draft_acceptance_inputs(ctx, &logits, &hidden, &current)) {
            throw std::runtime_error("DFlash GPU predictor requires single-ubatch outputs on one backend");
        }
        if (!implementation.context) {
            auto init = get_init(ggml_backend_get_device(current));
            if (!init) {
                throw std::runtime_error("DFlash acceptance GPU backend is unavailable");
            }
            ggml_draft_acceptance_weights weights = {
                int(model.inputs), int(model.hidden), int(model.width), model.threshold, model.scale, model.bias,
                model.mean.data(), model.stddev.data(), model.w1.data(), model.b1.data(), model.w2.data(), model.b2.data(),
            };
            implementation = init(current, &weights);
            backend = current;
        }
        if (current != backend || !implementation.context) {
            throw std::runtime_error("DFlash acceptance GPU backend changed");
        }
        ggml_draft_acceptance_result result{};
        implementation.evaluate(implementation.context, logits, model.hidden ? hidden : nullptr, &batch, &result);
        if (result.kept < 0 || result.kept > GGML_DRAFT_ACCEPTANCE_COUNT) {
            throw std::runtime_error("invalid GPU acceptance cutoff");
        }
        for (int j = 0; j < GGML_DRAFT_ACCEPTANCE_COUNT; ++j) {
            if (!std::isfinite(result.probabilities[j]) || result.probabilities[j] < 0 || result.probabilities[j] > 1) {
                throw std::runtime_error("invalid GPU acceptance probability");
            }
            for (float value : result.stats[j]) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("invalid GPU acceptance feature");
                }
            }
        }
        return result;
    }
};
