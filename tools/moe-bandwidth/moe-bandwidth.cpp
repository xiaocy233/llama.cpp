#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-moe-balance.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(name, value) == 0);
#else
    GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
}

int main(int argc, char ** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        fprintf(stderr, "Usage: %s OUTPUT -m MODEL [model/context options]\n", argv[0]);
        fprintf(stderr, "Fit CPU latency at 1/5/10 experts with 8 threads, cold CPU caches and concurrent H2D on one idle CUDA GPU.\n");
        return argc < 2 ? 1 : 0;
    }
    const std::string output = argv[1];
    common_params params;
    params.n_ctx = 512;
    params.n_gpu_layers = -1;
    params.n_cache_layers = 0;
    params.n_cache_slots = 64;
    params.cpuparams.n_threads = ggml_moe_calibration_threads;
    params.fit_params = false;
    common_init();
    std::vector<char *> args{argv[0]};
    for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
    args.push_back(nullptr);
    if (!common_params_parse(args.size() - 1, args.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (params.cpuparams.n_threads != ggml_moe_calibration_threads) {
        fprintf(stderr, "CPU latency calibration requires exactly 8 participating threads (-t 8).\n");
        return 1;
    }
    params.warmup = false;
    ggml_time_init();
    const std::string temporary = output + ".tmp." + std::to_string(ggml_time_us());
    {
        std::ofstream file(temporary);
        file << ggml_moe_profile_version << '\n';
        file.close();
        if (!file) { fprintf(stderr, "Cannot write %s\n", temporary.c_str()); return 1; }
    }
    set_env("GGML_MOE_HYBRID", "1");
    set_env("GGML_MOE_HYBRID_POLICY", "balanced");
    set_env("GGML_MOE_HYBRID_BW_MEASURE", "1");
    set_env("GGML_MOE_HYBRID_BW_PROFILE", temporary.c_str());
    set_env("GGML_MOE_HYBRID_THREADS", std::to_string(params.cpuparams.n_threads).c_str());
    llama_backend_init();
    llama_numa_init(params.numa);
    bool loaded = false;
    {
        auto init = common_init_from_params(params);
        loaded = init && init->model() && init->context();
    }
    llama_backend_free();
    std::ifstream file(temporary);
    std::string header, first;
    std::getline(file, header);
    std::getline(file, first);
    file.close();
    if (!loaded || first.empty()) {
        fprintf(stderr, "No latency profile generated; use a supported quantized MoE with CUDA expert slots.\n");
        std::remove(temporary.c_str());
        return 1;
    }
    if (std::rename(temporary.c_str(), output.c_str()) != 0) {
        fprintf(stderr, "Cannot publish %s; profile remains in %s\n", output.c_str(), temporary.c_str());
        return 1;
    }
    printf("CPU latency and H2D profile saved to %s\n", output.c_str());
    return 0;
}
