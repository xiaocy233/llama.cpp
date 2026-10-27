#pragma once

#include "llama.h"

#include <memory>
#include <vector>

class common_moe_autotune {
public:
    explicit common_moe_autotune(llama_context * target, llama_context * draft);
    ~common_moe_autotune();
    void begin(int prompt_tokens);
    bool prepare();
    float threshold() const;
    size_t truncate(const std::vector<float> & confidence, int limit, int position);
    void finish(int accepted);
    void end(uint64_t output_tokens, bool completed);

private:
    struct impl;
    std::unique_ptr<impl> state;
};
