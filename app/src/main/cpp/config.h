#pragma once

struct NnapiOptions {
    enum class Flag : uint32_t {
        None        = 0,
        UseFp16     = 1u << 0,  // NNAPI_FLAG_USE_FP16
        CpuDisabled = 1u << 1,  // NNAPI_FLAG_CPU_DISABLED
        CpuOnly     = 1u << 2,  // NNAPI_FLAG_CPU_ONLY
        UseNchw     = 1u << 3,  // NNAPI_FLAG_USE_NCHW
    };
    static constexpr uint32_t to_raw(Flag f) { return static_cast<uint32_t>(f); }

    // bitwise helpers
    friend constexpr Flag operator|(Flag a, Flag b) {
        return static_cast<Flag>( static_cast<uint32_t>(a) | static_cast<uint32_t>(b) );
    }
    friend constexpr Flag& operator|=(Flag& a, Flag b) { a = a | b; return a; }
    Flag flags = Flag::None;
};

struct XnnPackOptions {
    bool use_session_threads = false;
};

struct RunnerSettings {
    int  num_cpu_cores;

    bool use_xnnpack   = true;
    bool use_nnapi     = true;
    bool use_parallel_execution  = false; // github says parallel execution is deprecated but also says its needed for some cases
    bool use_layout_optimization_instead_of_extended = false; // website said if not nnapi use extended but this seems faster

    NnapiOptions   nnapi{};
    XnnPackOptions xnnpack{};
};
