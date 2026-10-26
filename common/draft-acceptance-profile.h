#pragma once

#ifdef GGML_DFLASH_NVTX
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <cstdlib>

struct common_dflash_profile_range {
    common_dflash_profile_range(const char * name, uint64_t value = 0) {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message.ascii = name;
        event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        event.payload.ullValue = value;
        nvtxRangePushEx(&event);
    }
    ~common_dflash_profile_range() { nvtxRangePop(); }
};

inline void common_dflash_profile_window() {
    static const int skip = std::getenv("DFLASH_PROFILE_SKIP") ? std::atoi(std::getenv("DFLASH_PROFILE_SKIP")) : 8;
    static const int steps = std::getenv("DFLASH_PROFILE_STEPS") ? std::atoi(std::getenv("DFLASH_PROFILE_STEPS")) : 32;
    static int round = 0;
    static nvtxRangeId_t capture = 0;
    ++round;
    if (round == skip + 1) {
        capture = nvtxRangeStartA("dflash.generation");
    }
    if (round == skip + steps + 1) {
        nvtxRangeEnd(capture);
    }
}

#define DFLASH_PROFILE_JOIN_(a, b) a##b
#define DFLASH_PROFILE_JOIN(a, b) DFLASH_PROFILE_JOIN_(a, b)
#define DFLASH_PROFILE(...) common_dflash_profile_range DFLASH_PROFILE_JOIN(profile_range_, __LINE__)(__VA_ARGS__)
#else
inline void common_dflash_profile_window() {}
#define DFLASH_PROFILE(...) do {} while (0)
#endif
