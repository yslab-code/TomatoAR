#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "allocator.h"
#include "cpu.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace {

constexpr int kModelInputSize = 512;
constexpr int kNumAnchors = 5376;
constexpr int kOutRows = 8;
constexpr size_t kMaxPreNmsCandidates = 1024;
constexpr size_t kMaxTrusses = 100;
constexpr size_t kMaxFruits = 300;
constexpr float kNormScale = 1.0f / 255.0f;
constexpr float kPadNormValue = 114.0f / 255.0f;
constexpr float kGpuProbeMaxClsDiff = 0.08f;

struct CachedOutput {
    bool valid = false;
    int img_w = 0;
    int img_h = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    float infer_ms = 0.0f;
    std::vector<float> data;
};

struct CpuAffinityCandidate {
    ncnn::CpuSet mask;
    int threads = 1;
    std::string tag;
    bool includes_little = false;
};

struct CpuProfile {
    bool valid = false;
    bool fp16_arith = false;
    ncnn::CpuSet mask;
    int threads = 1;
    std::string tag;
    float bench_ms = 0.0f;
};

// Allocators MUST be declared before g_yolo_net so reverse destruction destroys Net first.
std::mutex g_mutex;
ncnn::PoolAllocator g_blob_pool_allocator;
ncnn::PoolAllocator g_workspace_pool_allocator;
ncnn::Net g_yolo_net;
ncnn::Mat g_input_pad;
bool g_model_loaded = false;
bool g_gpu_active = false;
bool g_gpu_instance_created = false;
CachedOutput g_cached;
CpuProfile g_cached_cpu_profile;
ncnn::CpuSet g_active_cpu_mask;
int g_active_cpu_threads = 4;
int g_cpu_affinity_epoch = 0;
int g_pad_last_left = -1;
int g_pad_last_top = -1;
int g_pad_last_w = -1;
int g_pad_last_h = -1;

struct DetBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int cls;
    int parent_truss = -1;
};

struct TrussSummary {
    DetBox box;
    int ripe = 0;
    int turning = 0;
    int immature = 0;
    int grade_code = 0; // 0: 待检, 1: A, 2: B, 3: C, 4: D, 5: E
    float maturity = -1.0f;
};

void reset_pad_border_cache() {
    g_pad_last_left = -1;
    g_pad_last_top = -1;
    g_pad_last_w = -1;
    g_pad_last_h = -1;
}

void clear_runtime_state(bool clear_cached_data = false) {
    g_yolo_net.clear();
    g_input_pad.release();
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    g_model_loaded = false;
    g_gpu_active = false;
    g_cached.valid = false;
    reset_pad_border_cache();
    if (clear_cached_data) {
        std::vector<float>().swap(g_cached.data);
    }
}

void apply_cpu_affinity_now(const ncnn::CpuSet& mask, int threads) {
    ncnn::set_omp_num_threads(threads);
    if (mask.num_enabled() > 0) {
        ncnn::set_cpu_thread_affinity(mask);
    } else {
        ncnn::set_cpu_powersave(0);
    }
}

void ensure_thread_cpu_affinity() {
    thread_local int tl_applied_epoch = -1;
    if (!g_gpu_active && g_cpu_affinity_epoch > 0 && tl_applied_epoch != g_cpu_affinity_epoch) {
        apply_cpu_affinity_now(g_active_cpu_mask, g_active_cpu_threads);
        tl_applied_epoch = g_cpu_affinity_epoch;
    }
}

bool cpu_supports_hw_fp16_arith() {
#if defined(__aarch64__)
    return ncnn::cpu_support_arm_asimdhp() != 0;
#else
    return false;
#endif
}

bool cpu_sets_equal(const ncnn::CpuSet& a, const ncnn::CpuSet& b, int total_cores) {
    const int limit = std::clamp(total_cores, 8, 64);
    for (int i = 0; i < limit; ++i) {
        if (a.is_enabled(i) != b.is_enabled(i)) {
            return false;
        }
    }
    return true;
}

long read_sysfs_cpu_max_freq(int cpu_id) {
    char path[128];
    const char* patterns[2] = {
        "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
        "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq"
    };
    for (const char* pat : patterns) {
        std::snprintf(path, sizeof(path), pat, cpu_id);
        FILE* fp = std::fopen(path, "r");
        if (!fp) continue;
        long khz = 0;
        const int matched = std::fscanf(fp, "%ld", &khz);
        std::fclose(fp);
        if (matched == 1 && khz > 0) {
            return khz;
        }
    }
    return -1;
}

void add_cpu_candidate(
        std::vector<CpuAffinityCandidate>& out,
        int total_cores,
        const ncnn::CpuSet& mask,
        int threads,
        const std::string& tag,
        bool includes_little) {
    if (threads <= 0) return;
    if (mask.num_enabled() > 0) {
        threads = std::min(threads, mask.num_enabled());
    }
    for (const auto& existing : out) {
        if (existing.threads == threads && cpu_sets_equal(existing.mask, mask, total_cores)) {
            return;
        }
    }
    CpuAffinityCandidate cand;
    cand.mask = mask;
    cand.threads = threads;
    cand.tag = tag;
    cand.includes_little = includes_little;
    out.push_back(std::move(cand));
}

void build_cpu_affinity_candidates(std::vector<CpuAffinityCandidate>& cands) {
    cands.clear();
    const int total_cores = std::max(1, ncnn::get_cpu_count());
    const int big_cores = ncnn::get_big_cpu_count();

    if (big_cores > 0) {
        const ncnn::CpuSet& big_mask = ncnn::get_cpu_thread_affinity_mask(2);
        const int big_t = std::clamp(big_cores, 1, 4);
        add_cpu_candidate(cands, total_cores, big_mask, big_t, std::to_string(big_t) + "大核", false);
        if (big_cores >= 4) {
            add_cpu_candidate(cands, total_cores, big_mask, 2, "2大核", false);
        }
    }

    const int probe_cpus = std::min(total_cores, 32);
    std::vector<long> freqs(static_cast<size_t>(probe_cpus), -1);
    long min_freq = -1;
    long max_freq = -1;
    int valid_cnt = 0;
    for (int i = 0; i < probe_cpus; ++i) {
        const long f = read_sysfs_cpu_max_freq(i);
        freqs[static_cast<size_t>(i)] = f;
        if (f > 0) {
            if (min_freq < 0 || f < min_freq) min_freq = f;
            if (max_freq < 0 || f > max_freq) max_freq = f;
            ++valid_cnt;
        }
    }

    bool added_multi_non_little = (big_cores >= 2);
    if (valid_cnt >= 2 && min_freq > 0 && max_freq > min_freq) {
        ncnn::CpuSet perf_mask;
        ncnn::CpuSet mid_mask;
        for (int i = 0; i < probe_cpus; ++i) {
            const long f = freqs[static_cast<size_t>(i)];
            if (f > min_freq) {
                perf_mask.enable(i);
                if (f < max_freq) {
                    mid_mask.enable(i);
                }
            }
        }
        const int perf_cnt = perf_mask.num_enabled();
        if (perf_cnt >= 2) {
            const int perf_t = std::clamp(perf_cnt, 2, 4);
            add_cpu_candidate(cands, total_cores, perf_mask, perf_t, std::to_string(perf_t) + "性能核", false);
            if (perf_cnt >= 4) {
                add_cpu_candidate(cands, total_cores, perf_mask, 2, "2性能核", false);
            }
            added_multi_non_little = true;
        }
        const int mid_cnt = mid_mask.num_enabled();
        if (mid_cnt >= 2) {
            const int mid_t = std::clamp(mid_cnt, 2, 4);
            add_cpu_candidate(cands, total_cores, mid_mask, mid_t, std::to_string(mid_t) + "中核", false);
            added_multi_non_little = true;
        }
    }

    const ncnn::CpuSet& all_mask = ncnn::get_cpu_thread_affinity_mask(0);
    if (cands.empty()) {
        const int all_t = std::clamp(total_cores, 1, 4);
        add_cpu_candidate(cands, total_cores, all_mask, all_t, std::to_string(all_t) + "线程", false);
    } else if (big_cores == 1 && total_cores >= 4 && !added_multi_non_little) {
        add_cpu_candidate(cands, total_cores, all_mask, 4, "4线程", true);
        add_cpu_candidate(cands, total_cores, all_mask, 2, "2线程", true);
    }
}

float box_iou(const DetBox& a, const DetBox& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? (inter / denom) : 0.0f;
}

std::vector<DetBox> apply_nms(std::vector<DetBox>& boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const DetBox& a, const DetBox& b) {
        return a.conf > b.conf;
    });
    std::vector<DetBox> kept;
    kept.reserve(boxes.size());
    for (const auto& cand : boxes) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (box_iou(cand, k) > iou_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(cand);
        }
    }
    return kept;
}

int compute_grade_code(int ripe, int turning, int immature) {
    const int total = ripe + turning + immature;
    if (total <= 0) {
        return 0; // 待检
    }
    // Exact integer comparison matching Table 1:
    // I = (ripe + 0.5 * turning) / total; 5 * (2*ripe + turning) vs {8, 6, 4, 2} * total
    const int score5 = 5 * (2 * ripe + turning);
    if (score5 > 8 * total) return 1; // A: 0.80 < I <= 1.00
    if (score5 > 6 * total) return 2; // B: 0.60 < I <= 0.80
    if (score5 > 4 * total) return 3; // C: 0.40 < I <= 0.60
    if (score5 > 2 * total) return 4; // D: 0.20 < I <= 0.40
    return 5;                         // E: 0.00 <= I <= 0.20
}

std::string format_backend_label(bool use_gpu, bool enable_fp16, int cpu_threads, int big_cores) {
    const std::string prec = enable_fp16 ? " (FP16存储)" : " (FP32)";
    if (use_gpu) {
        return "Vulkan GPU" + prec;
    }
    if (big_cores > 0) {
        return "CPU " + std::to_string(cpu_threads) + "大核" + prec;
    }
    return "CPU " + std::to_string(cpu_threads) + "线程" + prec;
}

std::string format_cpu_tune_label(const std::string& tag, bool fp16_arith) {
    return "CPU " + tag + (fp16_arith ? " (FP16加速)" : " (FP32)");
}

void build_gradient_probe(ncnn::Mat& probe_in) {
    probe_in.create(kModelInputSize, kModelInputSize, 3);
    for (int c = 0; c < 3; ++c) {
        float* ptr = probe_in.channel(c);
        for (int y = 0; y < kModelInputSize; ++y) {
            float* row = ptr + y * kModelInputSize;
            for (int x = 0; x < kModelInputSize; ++x) {
                row[x] = static_cast<float>((y * 17 + x * 31 + c * 53) & 255) * kNormScale;
            }
        }
    }
}

bool validate_probe_output(const ncnn::Mat& out, bool enable_fp16, std::string& err_msg) {
    if (out.dims != 2 || out.h != kOutRows || out.w != kNumAnchors) {
        err_msg = "模型输出形状异常: dims=" + std::to_string(out.dims) +
                  " h=" + std::to_string(out.h) + " w=" + std::to_string(out.w) +
                  " (期望 2D 8x5376)";
        return false;
    }
    for (int r = 0; r < out.h; ++r) {
        const float* row_ptr = out.row(r);
        for (int c = 0; c < out.w; ++c) {
            if (!std::isfinite(row_ptr[c])) {
                err_msg = std::string(enable_fp16 ? "FP16" : "FP32") + " 试推理输出含 NaN/Inf";
                return false;
            }
        }
    }
    return true;
}

float max_cls_diff_between(const ncnn::Mat& a, const ncnn::Mat& b) {
    float max_diff = 0.0f;
    for (int r = 4; r < kOutRows; ++r) {
        const float* a_row = a.row(r);
        const float* b_row = b.row(r);
        for (int i = 0; i < kNumAnchors; ++i) {
            const float diff = std::abs(a_row[i] - b_row[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

bool run_cpu_fp32_gradient_ref(
        AAssetManager* mgr,
        int cpu_threads,
        const ncnn::Mat& grad_in,
        ncnn::Mat& ref_out,
        std::string& err_msg) {
    ncnn::Net cpu_ref;
    ncnn::Option ref_opt;
    ref_opt.use_vulkan_compute = false;
    ref_opt.num_threads = cpu_threads;
    ref_opt.lightmode = true;
    ref_opt.use_fp16_packed = false;
    ref_opt.use_fp16_storage = false;
    ref_opt.use_fp16_arithmetic = false;
    ref_opt.use_bf16_packed = false;
    ref_opt.use_bf16_storage = false;
    cpu_ref.opt = ref_opt;

    const int p_ret = cpu_ref.load_param(mgr, "best.ncnn.param");
    if (p_ret != 0) {
        err_msg = "Vulkan 参考 CPU 模型 load_param 失败 (ret=" + std::to_string(p_ret) + ")";
        return false;
    }
    const int m_ret = cpu_ref.load_model(mgr, "best.ncnn.bin");
    if (m_ret != 0) {
        err_msg = "Vulkan 参考 CPU 模型 load_model 失败 (ret=" + std::to_string(m_ret) + ")";
        return false;
    }

    ncnn::Extractor ref_ex = cpu_ref.create_extractor();
    if (ref_ex.input("in0", grad_in) != 0) {
        err_msg = "Vulkan 参考 CPU 模型输入节点 in0 绑定失败";
        return false;
    }
    const int ex_ret = ref_ex.extract("out0", ref_out);
    if (ex_ret != 0) {
        err_msg = "Vulkan 参考 CPU 模型 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
        return false;
    }
    if (!validate_probe_output(ref_out, false, err_msg)) {
        err_msg = "Vulkan 参考 CPU 输出异常: " + err_msg;
        return false;
    }
    return true;
}

bool load_cpu_net_and_probe(
        AAssetManager* mgr,
        bool fp16_arith,
        const ncnn::CpuSet& mask,
        int threads,
        const ncnn::Mat& grad_in,
        ncnn::Mat& grad_out,
        std::string& err_msg) {
    clear_runtime_state();
    apply_cpu_affinity_now(mask, threads);

    ncnn::Option opt;
    opt.use_vulkan_compute = false;
    opt.num_threads = threads;
    opt.lightmode = true;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_fp16_packed = fp16_arith;
    opt.use_fp16_storage = fp16_arith;
    opt.use_fp16_arithmetic = fp16_arith;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_packing_layout = true;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.openmp_blocktime = 20;
    g_yolo_net.opt = opt;

    const int p_ret = g_yolo_net.load_param(mgr, "best.ncnn.param");
    if (p_ret != 0) {
        err_msg = "load_param 失败 (ret=" + std::to_string(p_ret) + ")";
        clear_runtime_state();
        return false;
    }
    const int m_ret = g_yolo_net.load_model(mgr, "best.ncnn.bin");
    if (m_ret != 0) {
        err_msg = "load_model 失败 (ret=" + std::to_string(m_ret) + ")";
        clear_runtime_state();
        return false;
    }

    {
        ncnn::Mat zero_in(kModelInputSize, kModelInputSize, 3);
        zero_in.fill(0.0f);
        ncnn::Extractor ex = g_yolo_net.create_extractor();
        ex.set_num_threads(threads);
        if (ex.input("in0", zero_in) != 0) {
            err_msg = "试推理输入节点 in0 绑定失败";
            clear_runtime_state();
            return false;
        }
        ncnn::Mat zero_out;
        const int ex_ret = ex.extract("out0", zero_out);
        if (ex_ret != 0) {
            err_msg = "试推理 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
            clear_runtime_state();
            return false;
        }
        if (!validate_probe_output(zero_out, fp16_arith, err_msg)) {
            clear_runtime_state();
            return false;
        }
    }

    {
        ncnn::Extractor ex = g_yolo_net.create_extractor();
        ex.set_num_threads(threads);
        if (ex.input("in0", grad_in) != 0) {
            err_msg = "梯度试推理输入节点 in0 绑定失败";
            clear_runtime_state();
            return false;
        }
        const int ex_ret = ex.extract("out0", grad_out);
        if (ex_ret != 0) {
            err_msg = "梯度试推理 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
            clear_runtime_state();
            return false;
        }
        if (!validate_probe_output(grad_out, fp16_arith, err_msg)) {
            clear_runtime_state();
            return false;
        }
    }

    return true;
}

bool benchmark_loaded_cpu_net(
        const ncnn::Mat& grad_in,
        const std::vector<CpuAffinityCandidate>& cands,
        CpuAffinityCandidate& best_cand,
        float& best_ms) {
    bool has_best = false;
    for (const auto& c : cands) {
        apply_cpu_affinity_now(c.mask, c.threads);
        {
            ncnn::Extractor warm = g_yolo_net.create_extractor();
            warm.set_num_threads(c.threads);
            ncnn::Mat warm_out;
            if (warm.input("in0", grad_in) != 0 || warm.extract("out0", warm_out) != 0) {
                continue;
            }
        }
        float cand_min_ms = 1e9f;
        bool ok = true;
        for (int rep = 0; rep < 2; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            ncnn::Extractor ex = g_yolo_net.create_extractor();
            ex.set_num_threads(c.threads);
            ncnn::Mat out;
            if (ex.input("in0", grad_in) != 0 || ex.extract("out0", out) != 0) {
                ok = false;
                break;
            }
            const float ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
            if (ms < cand_min_ms) {
                cand_min_ms = ms;
            }
        }
        if (!ok) continue;

        bool better = !has_best;
        if (has_best) {
            if (!best_cand.includes_little && c.includes_little) {
                better = (cand_min_ms < best_ms * 0.85f);
            } else if (best_cand.includes_little && !c.includes_little) {
                better = (cand_min_ms * 0.85f <= best_ms);
            } else {
                better = (cand_min_ms < best_ms);
            }
        }
        if (better) {
            best_cand = c;
            best_ms = cand_min_ms;
            has_best = true;
        }
    }
    return has_best;
}

void commit_cpu_state(const CpuAffinityCandidate& cand, bool fp16_arith, float bench_ms, std::string& backend_label) {
    g_yolo_net.opt.num_threads = cand.threads;
    g_active_cpu_mask = cand.mask;
    g_active_cpu_threads = cand.threads;
    ++g_cpu_affinity_epoch;
    apply_cpu_affinity_now(g_active_cpu_mask, g_active_cpu_threads);
    g_cached_cpu_profile.valid = true;
    g_cached_cpu_profile.fp16_arith = fp16_arith;
    g_cached_cpu_profile.mask = cand.mask;
    g_cached_cpu_profile.threads = cand.threads;
    g_cached_cpu_profile.tag = cand.tag;
    g_cached_cpu_profile.bench_ms = bench_ms;
    g_model_loaded = true;
    g_gpu_active = false;
    backend_label = format_cpu_tune_label(cand.tag, fp16_arith);
}

bool load_and_tune_cpu(AAssetManager* mgr, std::string& err_msg, std::string& backend_label) {
    ncnn::Mat grad_in;
    build_gradient_probe(grad_in);

    if (g_cached_cpu_profile.valid) {
        ncnn::Mat cached_out;
        std::string cached_err;
        if (load_cpu_net_and_probe(
                mgr,
                g_cached_cpu_profile.fp16_arith,
                g_cached_cpu_profile.mask,
                g_cached_cpu_profile.threads,
                grad_in,
                cached_out,
                cached_err)) {
            g_active_cpu_mask = g_cached_cpu_profile.mask;
            g_active_cpu_threads = g_cached_cpu_profile.threads;
            ++g_cpu_affinity_epoch;
            apply_cpu_affinity_now(g_active_cpu_mask, g_active_cpu_threads);
            g_model_loaded = true;
            g_gpu_active = false;
            backend_label = format_cpu_tune_label(g_cached_cpu_profile.tag, g_cached_cpu_profile.fp16_arith);
            return true;
        }
        g_cached_cpu_profile = CpuProfile();
    }

    std::vector<CpuAffinityCandidate> cands;
    build_cpu_affinity_candidates(cands);
    const CpuAffinityCandidate& init_cand = cands.front();

    ncnn::Mat fp32_grad_out;
    if (!load_cpu_net_and_probe(mgr, false, init_cand.mask, init_cand.threads, grad_in, fp32_grad_out, err_msg)) {
        clear_runtime_state();
        return false;
    }

    CpuAffinityCandidate best_fp32;
    float best_fp32_ms = 0.0f;
    if (!benchmark_loaded_cpu_net(grad_in, cands, best_fp32, best_fp32_ms)) {
        err_msg = "CPU FP32 基准测试失败";
        clear_runtime_state();
        return false;
    }

    bool tried_fp16_load = false;
    if (cpu_supports_hw_fp16_arith()) {
        tried_fp16_load = true;
        ncnn::Mat fp16_grad_out;
        std::string fp16_err;
        if (load_cpu_net_and_probe(mgr, true, init_cand.mask, init_cand.threads, grad_in, fp16_grad_out, fp16_err)) {
            const float diff = max_cls_diff_between(fp16_grad_out, fp32_grad_out);
            if (std::isfinite(diff) && diff <= kGpuProbeMaxClsDiff) {
                CpuAffinityCandidate best_fp16;
                float best_fp16_ms = 0.0f;
                if (benchmark_loaded_cpu_net(grad_in, cands, best_fp16, best_fp16_ms) &&
                    best_fp16_ms < best_fp32_ms) {
                    commit_cpu_state(best_fp16, true, best_fp16_ms, backend_label);
                    return true;
                }
            }
        }
    }

    if (tried_fp16_load) {
        if (!load_cpu_net_and_probe(mgr, false, best_fp32.mask, best_fp32.threads, grad_in, fp32_grad_out, err_msg)) {
            clear_runtime_state();
            return false;
        }
    }
    commit_cpu_state(best_fp32, false, best_fp32_ms, backend_label);
    return true;
}

bool load_and_probe(
        AAssetManager* mgr,
        bool use_gpu,
        bool enable_fp16,
        std::string& err_msg,
        std::string& backend_label,
        ncnn::Mat* cached_cpu_ref = nullptr) {
    clear_runtime_state();

    if (use_gpu) {
#if NCNN_VULKAN
        if (!g_gpu_instance_created) {
            if (ncnn::create_gpu_instance() != 0) {
                err_msg = "Vulkan GPU 实例初始化失败";
                return false;
            }
            g_gpu_instance_created = true;
        }
        if (ncnn::get_gpu_count() <= 0) {
            err_msg = "设备未检测到可用 Vulkan GPU";
            return false;
        }
#else
        err_msg = "当前 NCNN 库未编译 Vulkan 支持";
        return false;
#endif
    }

    const int big_cores = ncnn::get_big_cpu_count();
    const int total_cores = ncnn::get_cpu_count();
    const bool use_big_only = (big_cores > 0);
    const int cpu_threads = std::clamp(use_big_only ? big_cores : total_cores, 1, 4);
    ncnn::set_cpu_powersave(use_big_only ? 2 : 0);

    ncnn::Option opt;
    opt.use_vulkan_compute = use_gpu;
    opt.num_threads = cpu_threads;
    opt.lightmode = true;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_fp16_packed = enable_fp16;
    opt.use_fp16_storage = enable_fp16;
    // Keep FP32 accumulation for ScConv/SRU stability; FP16 packed+storage gives bandwidth gain safely.
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_packing_layout = true;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.openmp_blocktime = 20;
    g_yolo_net.opt = opt;

    const int p_ret = g_yolo_net.load_param(mgr, "best.ncnn.param");
    if (p_ret != 0) {
        err_msg = "load_param 失败 (ret=" + std::to_string(p_ret) + ")";
        clear_runtime_state();
        return false;
    }
    const int m_ret = g_yolo_net.load_model(mgr, "best.ncnn.bin");
    if (m_ret != 0) {
        err_msg = "load_model 失败 (ret=" + std::to_string(m_ret) + ")";
        clear_runtime_state();
        return false;
    }

    // 1. Zero-input probe
    {
        ncnn::Mat zero_in(kModelInputSize, kModelInputSize, 3);
        zero_in.fill(0.0f);
        ncnn::Extractor ex = g_yolo_net.create_extractor();
        if (ex.input("in0", zero_in) != 0) {
            err_msg = "试推理输入节点 in0 绑定失败";
            clear_runtime_state();
            return false;
        }
        ncnn::Mat zero_out;
        const int ex_ret = ex.extract("out0", zero_out);
        if (ex_ret != 0) {
            err_msg = "试推理 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
            clear_runtime_state();
            return false;
        }
        if (!validate_probe_output(zero_out, enable_fp16, err_msg)) {
            clear_runtime_state();
            return false;
        }
    }

    // 2. Non-uniform gradient/stripe probe (+ CPU FP32 cross-check when Vulkan is active)
    ncnn::Mat grad_in;
    build_gradient_probe(grad_in);
    ncnn::Mat grad_out;
    {
        ncnn::Extractor ex = g_yolo_net.create_extractor();
        if (ex.input("in0", grad_in) != 0) {
            err_msg = "梯度试推理输入节点 in0 绑定失败";
            clear_runtime_state();
            return false;
        }
        const int ex_ret = ex.extract("out0", grad_out);
        if (ex_ret != 0) {
            err_msg = "梯度试推理 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
            clear_runtime_state();
            return false;
        }
        if (!validate_probe_output(grad_out, enable_fp16, err_msg)) {
            clear_runtime_state();
            return false;
        }
    }

    if (use_gpu) {
        ncnn::Mat local_ref;
        const ncnn::Mat* ref_ptr = nullptr;
        if (cached_cpu_ref && !cached_cpu_ref->empty()) {
            ref_ptr = cached_cpu_ref;
        } else {
            if (!run_cpu_fp32_gradient_ref(mgr, cpu_threads, grad_in, local_ref, err_msg)) {
                clear_runtime_state();
                return false;
            }
            if (cached_cpu_ref) {
                *cached_cpu_ref = local_ref;
            }
            ref_ptr = &local_ref;
        }

        const float max_cls_diff = max_cls_diff_between(grad_out, *ref_ptr);
        if (!std::isfinite(max_cls_diff) || max_cls_diff > kGpuProbeMaxClsDiff) {
            err_msg = "Vulkan 数值校验偏差过大 (diff=" + std::to_string(max_cls_diff) + ")";
            clear_runtime_state();
            return false;
        }
    }

    g_model_loaded = true;
    g_gpu_active = use_gpu;
    backend_label = format_backend_label(use_gpu, enable_fp16, cpu_threads, big_cores);
    return true;
}

jfloatArray make_error_array(JNIEnv* env, float code, float ret = 0.0f, float dims = 0.0f, float h = 0.0f, float w = 0.0f) {
    const float vals[6] = {-1.0f, code, ret, dims, h, w};
    jfloatArray arr = env->NewFloatArray(6);
    if (arr != nullptr) {
        env->SetFloatArrayRegion(arr, 0, 6, vals);
    }
    return arr;
}

void fill_letterbox_borders(float* plane, int active_left, int active_top, int active_w, int active_h) {
    const int active_right = active_left + active_w;
    const int active_bottom = active_top + active_h;

    for (int y = 0; y < active_top; ++y) {
        std::fill_n(plane + y * kModelInputSize, kModelInputSize, kPadNormValue);
    }
    for (int y = active_top; y < active_bottom; ++y) {
        float* row = plane + y * kModelInputSize;
        if (active_left > 0) {
            std::fill_n(row, active_left, kPadNormValue);
        }
        if (active_right < kModelInputSize) {
            std::fill_n(row + active_right, kModelInputSize - active_right, kPadNormValue);
        }
    }
    for (int y = active_bottom; y < kModelInputSize; ++y) {
        std::fill_n(plane + y * kModelInputSize, kModelInputSize, kPadNormValue);
    }
}

// Fuses rotate/mirror + letterbox border padding + 1/255 normalization into a reusable 512x512x3 Mat.
bool fill_letterbox_input(
        const ncnn::Mat& pre_rot,
        int rot_deg,
        bool mirror_horizontal,
        int pad_left,
        int pad_top,
        ncnn::Mat& out_pad) {
    if (pre_rot.empty() || pre_rot.c != 3) {
        return false;
    }
    const int src_w = pre_rot.w;
    const int src_h = pre_rot.h;
    const bool swap_wh = (rot_deg == 90 || rot_deg == 270);
    const int dst_w = swap_wh ? src_h : src_w;
    const int dst_h = swap_wh ? src_w : src_h;

    if (pad_left < 0 || pad_top < 0 ||
        pad_left + dst_w > kModelInputSize ||
        pad_top + dst_h > kModelInputSize) {
        return false;
    }

    if (out_pad.empty() || out_pad.dims != 3 ||
        out_pad.w != kModelInputSize || out_pad.h != kModelInputSize || out_pad.c != 3) {
        out_pad.create(kModelInputSize, kModelInputSize, 3);
        if (out_pad.empty()) {
            return false;
        }
        reset_pad_border_cache();
    }

    const bool border_dirty = (pad_left != g_pad_last_left ||
                               pad_top != g_pad_last_top ||
                               dst_w != g_pad_last_w ||
                               dst_h != g_pad_last_h);
    const int mode = ((rot_deg / 90) & 3) * 2 + (mirror_horizontal ? 1 : 0);
    constexpr int kTile = 16;

    for (int c = 0; c < 3; ++c) {
        const float* sptr = pre_rot.channel(c);
        float* dptr = out_pad.channel(c);
        if (border_dirty) {
            fill_letterbox_borders(dptr, pad_left, pad_top, dst_w, dst_h);
        }

        switch (mode) {
            case 0: { // rot 0
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (pad_top + sy) * kModelInputSize + pad_left;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[sx] = srow[sx] * kNormScale;
                    }
                }
                break;
            }
            case 1: { // rot 0 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (pad_top + sy) * kModelInputSize + pad_left;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[dst_w - 1 - sx] = srow[sx] * kNormScale;
                    }
                }
                break;
            }
            case 2: { // rot 90 (cache-blocked)
                for (int sy0 = 0; sy0 < src_h; sy0 += kTile) {
                    const int sy1 = std::min(sy0 + kTile, src_h);
                    for (int sx0 = 0; sx0 < src_w; sx0 += kTile) {
                        const int sx1 = std::min(sx0 + kTile, src_w);
                        for (int sy = sy0; sy < sy1; ++sy) {
                            const float* srow = sptr + sy * src_w;
                            const int dcol = pad_left + (src_h - 1 - sy);
                            for (int sx = sx0; sx < sx1; ++sx) {
                                dptr[(pad_top + sx) * kModelInputSize + dcol] = srow[sx] * kNormScale;
                            }
                        }
                    }
                }
                break;
            }
            case 3: { // rot 90 + mirror (cache-blocked)
                for (int sy0 = 0; sy0 < src_h; sy0 += kTile) {
                    const int sy1 = std::min(sy0 + kTile, src_h);
                    for (int sx0 = 0; sx0 < src_w; sx0 += kTile) {
                        const int sx1 = std::min(sx0 + kTile, src_w);
                        for (int sy = sy0; sy < sy1; ++sy) {
                            const float* srow = sptr + sy * src_w;
                            const int dcol = pad_left + sy;
                            for (int sx = sx0; sx < sx1; ++sx) {
                                dptr[(pad_top + sx) * kModelInputSize + dcol] = srow[sx] * kNormScale;
                            }
                        }
                    }
                }
                break;
            }
            case 4: { // rot 180
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (pad_top + (src_h - 1 - sy)) * kModelInputSize + pad_left;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[src_w - 1 - sx] = srow[sx] * kNormScale;
                    }
                }
                break;
            }
            case 5: { // rot 180 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (pad_top + (src_h - 1 - sy)) * kModelInputSize + pad_left;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[sx] = srow[sx] * kNormScale;
                    }
                }
                break;
            }
            case 6: { // rot 270 (cache-blocked)
                for (int sy0 = 0; sy0 < src_h; sy0 += kTile) {
                    const int sy1 = std::min(sy0 + kTile, src_h);
                    for (int sx0 = 0; sx0 < src_w; sx0 += kTile) {
                        const int sx1 = std::min(sx0 + kTile, src_w);
                        for (int sy = sy0; sy < sy1; ++sy) {
                            const float* srow = sptr + sy * src_w;
                            const int dcol = pad_left + sy;
                            for (int sx = sx0; sx < sx1; ++sx) {
                                dptr[(pad_top + (src_w - 1 - sx)) * kModelInputSize + dcol] = srow[sx] * kNormScale;
                            }
                        }
                    }
                }
                break;
            }
            case 7: { // rot 270 + mirror (cache-blocked)
                for (int sy0 = 0; sy0 < src_h; sy0 += kTile) {
                    const int sy1 = std::min(sy0 + kTile, src_h);
                    for (int sx0 = 0; sx0 < src_w; sx0 += kTile) {
                        const int sx1 = std::min(sx0 + kTile, src_w);
                        for (int sy = sy0; sy < sy1; ++sy) {
                            const float* srow = sptr + sy * src_w;
                            const int dcol = pad_left + (src_h - 1 - sy);
                            for (int sx = sx0; sx < sx1; ++sx) {
                                dptr[(pad_top + (src_w - 1 - sx)) * kModelInputSize + dcol] = srow[sx] * kNormScale;
                            }
                        }
                    }
                }
                break;
            }
            default:
                return false;
        }
    }
    if (border_dirty) {
        g_pad_last_left = pad_left;
        g_pad_last_top = pad_top;
        g_pad_last_w = dst_w;
        g_pad_last_h = dst_h;
    }
    return true;
}

bool run_net(
        const ncnn::Mat& pre_rot,
        int rot_deg,
        bool mirror_horizontal,
        int upright_w,
        int upright_h,
        int pad_left,
        int pad_top,
        float scale_x,
        float scale_y,
        bool save_cache,
        const std::chrono::steady_clock::time_point& t_start,
        ncnn::Mat& out_mat,
        float& out_net_ms,
        float& out_ret,
        float& out_dims,
        float& out_h,
        float& out_w) {
    if (!fill_letterbox_input(pre_rot, rot_deg, mirror_horizontal, pad_left, pad_top, g_input_pad)) {
        out_ret = -2.0f;
        return false;
    }

    ensure_thread_cpu_affinity();
    ncnn::Extractor ex = g_yolo_net.create_extractor();
    if (!g_gpu_active) {
        ex.set_num_threads(g_active_cpu_threads);
    }
    if (ex.input("in0", g_input_pad) != 0) {
        out_ret = -1.0f;
        out_dims = 0.0f;
        out_h = 0.0f;
        out_w = 0.0f;
        return false;
    }
    const int ex_ret = ex.extract("out0", out_mat);
    if (ex_ret != 0 || out_mat.dims != 2 || out_mat.h != kOutRows || out_mat.w != kNumAnchors) {
        out_ret = static_cast<float>(ex_ret);
        out_dims = static_cast<float>(out_mat.dims);
        out_h = static_cast<float>(out_mat.h);
        out_w = static_cast<float>(out_mat.w);
        return false;
    }

    const auto t_net_end = std::chrono::steady_clock::now();
    out_net_ms = std::chrono::duration<float, std::milli>(t_net_end - t_start).count();

    if (save_cache) {
        g_cached.data.resize(static_cast<size_t>(kOutRows) * static_cast<size_t>(kNumAnchors));
        for (int r = 0; r < kOutRows; ++r) {
            std::memcpy(
                    g_cached.data.data() + static_cast<size_t>(r) * static_cast<size_t>(kNumAnchors),
                    out_mat.row(r),
                    static_cast<size_t>(kNumAnchors) * sizeof(float));
        }
        g_cached.img_w = upright_w;
        g_cached.img_h = upright_h;
        g_cached.scale_x = scale_x;
        g_cached.scale_y = scale_y;
        g_cached.pad_left = pad_left;
        g_cached.pad_top = pad_top;
        g_cached.infer_ms = out_net_ms;
        g_cached.valid = true;
    } else {
        g_cached.valid = false;
    }
    return true;
}

jfloatArray postprocess_and_pack(
        JNIEnv* env,
        const float* rows[kOutRows],
        int img_w,
        int img_h,
        float scale_x,
        float scale_y,
        int pad_left,
        int pad_top,
        float base_infer_ms,
        float conf_thresh,
        float iou_thresh,
        bool agnostic_fruit_nms,
        bool assign_nearest_center) {
    const auto t_post_start = std::chrono::steady_clock::now();

    const float* row_cx = rows[0];
    const float* row_cy = rows[1];
    const float* row_bw = rows[2];
    const float* row_bh = rows[3];
    const float* row_c0 = rows[4];
    const float* row_c1 = rows[5];
    const float* row_c2 = rows[6];
    const float* row_c3 = rows[7];

    std::vector<DetBox> truss_cands;
    std::vector<DetBox> fruit_cands;
    truss_cands.reserve(64);
    fruit_cands.reserve(256);

    const float fw = static_cast<float>(img_w);
    const float fh = static_cast<float>(img_h);
    const float f_pad_l = static_cast<float>(pad_left);
    const float f_pad_t = static_cast<float>(pad_top);
    const float inv_sx = scale_x > 0.0f ? (1.0f / scale_x) : 1.0f;
    const float inv_sy = scale_y > 0.0f ? (1.0f / scale_y) : 1.0f;

    for (int i = 0; i < kNumAnchors; ++i) {
        const float s0 = row_c0[i];
        const float s1 = row_c1[i];
        const float s2 = row_c2[i];
        const float s3 = row_c3[i];

        const bool want_truss = (s0 >= conf_thresh) && std::isfinite(s0);

        int best_fruit_cls = -1;
        float best_fruit_score = -1.0f;
        bool want_any_fruit = false;

        if (agnostic_fruit_nms) {
            if (s1 >= conf_thresh && std::isfinite(s1) && s1 > best_fruit_score) {
                best_fruit_score = s1;
                best_fruit_cls = 1;
            }
            if (s2 >= conf_thresh && std::isfinite(s2) && s2 > best_fruit_score) {
                best_fruit_score = s2;
                best_fruit_cls = 2;
            }
            if (s3 >= conf_thresh && std::isfinite(s3) && s3 > best_fruit_score) {
                best_fruit_score = s3;
                best_fruit_cls = 3;
            }
            want_any_fruit = (best_fruit_cls > 0);
        } else {
            want_any_fruit =
                    (s1 >= conf_thresh && std::isfinite(s1)) ||
                    (s2 >= conf_thresh && std::isfinite(s2)) ||
                    (s3 >= conf_thresh && std::isfinite(s3));
        }

        if (!want_truss && !want_any_fruit) {
            continue;
        }

        const float cx = row_cx[i];
        const float cy = row_cy[i];
        const float bw = row_bw[i];
        const float bh = row_bh[i];
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(bw) || !std::isfinite(bh)) {
            continue;
        }

        const float x1 = std::clamp(((cx - bw * 0.5f) - f_pad_l) * inv_sx, 0.0f, fw);
        const float y1 = std::clamp(((cy - bh * 0.5f) - f_pad_t) * inv_sy, 0.0f, fh);
        const float x2 = std::clamp(((cx + bw * 0.5f) - f_pad_l) * inv_sx, 0.0f, fw);
        const float y2 = std::clamp(((cy + bh * 0.5f) - f_pad_t) * inv_sy, 0.0f, fh);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        if (want_truss) {
            truss_cands.push_back(DetBox{x1, y1, x2, y2, s0, 0, -1});
        }
        if (want_any_fruit) {
            if (agnostic_fruit_nms) {
                fruit_cands.push_back(DetBox{x1, y1, x2, y2, best_fruit_score, best_fruit_cls, -1});
            } else {
                if (s1 >= conf_thresh && std::isfinite(s1)) {
                    fruit_cands.push_back(DetBox{x1, y1, x2, y2, s1, 1, -1});
                }
                if (s2 >= conf_thresh && std::isfinite(s2)) {
                    fruit_cands.push_back(DetBox{x1, y1, x2, y2, s2, 2, -1});
                }
                if (s3 >= conf_thresh && std::isfinite(s3)) {
                    fruit_cands.push_back(DetBox{x1, y1, x2, y2, s3, 3, -1});
                }
            }
        }
    }

    auto desc_cmp = [](const DetBox& a, const DetBox& b) { return a.conf > b.conf; };
    if (truss_cands.size() > kMaxPreNmsCandidates) {
        std::partial_sort(truss_cands.begin(), truss_cands.begin() + kMaxPreNmsCandidates, truss_cands.end(), desc_cmp);
        truss_cands.resize(kMaxPreNmsCandidates);
    }
    if (fruit_cands.size() > kMaxPreNmsCandidates) {
        std::partial_sort(fruit_cands.begin(), fruit_cands.begin() + kMaxPreNmsCandidates, fruit_cands.end(), desc_cmp);
        fruit_cands.resize(kMaxPreNmsCandidates);
    }

    std::vector<DetBox> trusses = apply_nms(truss_cands, iou_thresh);
    if (trusses.size() > kMaxTrusses) {
        trusses.resize(kMaxTrusses);
    }
    // Strict weak ordering: quantized column bucket, then exact y1, then exact x1.
    std::sort(trusses.begin(), trusses.end(), [](const DetBox& a, const DetBox& b) {
        const int bx_a = static_cast<int>(std::floor(a.x1));
        const int bx_b = static_cast<int>(std::floor(b.x1));
        if (bx_a != bx_b) return bx_a < bx_b;
        if (a.y1 != b.y1) return a.y1 < b.y1;
        return a.x1 < b.x1;
    });

    std::vector<DetBox> fruits;
    if (agnostic_fruit_nms) {
        fruits = apply_nms(fruit_cands, iou_thresh);
    } else {
        for (int cls_id = 1; cls_id <= 3; ++cls_id) {
            std::vector<DetBox> cls_boxes;
            for (const auto& f : fruit_cands) {
                if (f.cls == cls_id) cls_boxes.push_back(f);
            }
            auto kept = apply_nms(cls_boxes, iou_thresh);
            fruits.insert(fruits.end(), kept.begin(), kept.end());
        }
        std::sort(fruits.begin(), fruits.end(), desc_cmp);
    }
    if (fruits.size() > kMaxFruits) {
        fruits.resize(kMaxFruits);
    }

    std::vector<TrussSummary> summaries(trusses.size());
    for (size_t i = 0; i < trusses.size(); ++i) {
        summaries[i].box = trusses[i];
    }

    for (auto& f : fruits) {
        const float fx = 0.5f * (f.x1 + f.x2);
        const float fy = 0.5f * (f.y1 + f.y2);

        int chosen_idx = -1;
        float best_dist2 = 0.0f;

        for (size_t tidx = 0; tidx < summaries.size(); ++tidx) {
            const auto& tb = summaries[tidx].box;
            if (fx >= tb.x1 && fx <= tb.x2 && fy >= tb.y1 && fy <= tb.y2) {
                if (!assign_nearest_center) {
                    chosen_idx = static_cast<int>(tidx);
                    break;
                }
                const float tcx = 0.5f * (tb.x1 + tb.x2);
                const float tcy = 0.5f * (tb.y1 + tb.y2);
                const float dx = fx - tcx;
                const float dy = fy - tcy;
                const float dist2 = dx * dx + dy * dy;
                if (chosen_idx < 0 || dist2 < best_dist2) {
                    best_dist2 = dist2;
                    chosen_idx = static_cast<int>(tidx);
                }
            }
        }

        f.parent_truss = chosen_idx;
        if (chosen_idx >= 0) {
            if (f.cls == 1) summaries[chosen_idx].ripe += 1;
            else if (f.cls == 2) summaries[chosen_idx].turning += 1;
            else if (f.cls == 3) summaries[chosen_idx].immature += 1;
        }
    }

    for (auto& s : summaries) {
        const int total = s.ripe + s.turning + s.immature;
        s.grade_code = compute_grade_code(s.ripe, s.turning, s.immature);
        s.maturity = (total > 0)
                ? (static_cast<float>(s.ripe) + 0.5f * static_cast<float>(s.turning)) / static_cast<float>(total)
                : -1.0f;
    }

    const auto t_post_end = std::chrono::steady_clock::now();
    const float post_ms = std::chrono::duration<float, std::milli>(t_post_end - t_post_start).count();
    const float total_infer_ms = std::max(0.0f, base_infer_ms + post_ms);

    std::vector<float> packed;
    packed.reserve(4 + summaries.size() * 10 + fruits.size() * 7);
    packed.push_back(1.0f);
    packed.push_back(total_infer_ms);
    packed.push_back(static_cast<float>(summaries.size()));
    packed.push_back(static_cast<float>(fruits.size()));

    for (const auto& s : summaries) {
        packed.push_back(s.box.x1);
        packed.push_back(s.box.y1);
        packed.push_back(s.box.x2);
        packed.push_back(s.box.y2);
        packed.push_back(s.box.conf);
        packed.push_back(static_cast<float>(s.ripe));
        packed.push_back(static_cast<float>(s.turning));
        packed.push_back(static_cast<float>(s.immature));
        packed.push_back(static_cast<float>(s.grade_code));
        packed.push_back(s.maturity);
    }

    for (const auto& f : fruits) {
        packed.push_back(f.x1);
        packed.push_back(f.y1);
        packed.push_back(f.x2);
        packed.push_back(f.y2);
        packed.push_back(f.conf);
        packed.push_back(static_cast<float>(f.cls));
        packed.push_back(static_cast<float>(f.parent_truss));
    }

    jfloatArray out_arr = env->NewFloatArray(static_cast<jsize>(packed.size()));
    if (out_arr != nullptr) {
        env->SetFloatArrayRegion(out_arr, 0, static_cast<jsize>(packed.size()), packed.data());
    }
    return out_arr;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_runtime_state(true);
    g_cached_cpu_profile = CpuProfile();
    g_active_cpu_mask = ncnn::CpuSet();
    g_cpu_affinity_epoch = 0;
#if NCNN_VULKAN
    if (g_gpu_instance_created) {
        ncnn::destroy_gpu_instance();
        g_gpu_instance_created = false;
    }
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_hzau_tomatoar_YoloDetector_initModel(
        JNIEnv* env, jobject thiz, jobject asset_manager, jboolean use_gpu) {
    std::lock_guard<std::mutex> lock(g_mutex);
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        return env->NewStringUTF("ERROR|未加载|AAssetManager 无效");
    }

    std::string backend;
    if (use_gpu == JNI_TRUE) {
        ncnn::Mat gpu_cpu_ref;
        std::string gpu_fp16_err;
        if (load_and_probe(mgr, true, true, gpu_fp16_err, backend, &gpu_cpu_ref)) {
            const std::string resp = "OK|" + backend + "|";
            return env->NewStringUTF(resp.c_str());
        }
        std::string gpu_fp32_err;
        if (load_and_probe(mgr, true, false, gpu_fp32_err, backend, &gpu_cpu_ref)) {
            const std::string resp = "OK|" + backend + "|";
            return env->NewStringUTF(resp.c_str());
        }

        const std::string gpu_err = gpu_fp16_err + "; FP32(" + gpu_fp32_err + ")";
        std::string cpu_err;
        std::string cpu_backend;
        if (load_and_tune_cpu(mgr, cpu_err, cpu_backend)) {
            const std::string resp = "FALLBACK|" + cpu_backend + "|" + gpu_err;
            return env->NewStringUTF(resp.c_str());
        }
        const std::string resp = "ERROR|未加载|GPU(" + gpu_err + "); CPU(" + cpu_err + ")";
        return env->NewStringUTF(resp.c_str());
    }

    std::string cpu_err;
    if (load_and_tune_cpu(mgr, cpu_err, backend)) {
        const std::string resp = "OK|" + backend + "|";
        return env->NewStringUTF(resp.c_str());
    }

    const std::string resp = "ERROR|未加载|" + cpu_err;
    return env->NewStringUTF(resp.c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_hzau_tomatoar_YoloDetector_detect(
        JNIEnv* env,
        jobject thiz,
        jobject bitmap,
        jfloat conf_thresh,
        jfloat iou_thresh,
        jboolean agnostic_fruit_nms,
        jboolean assign_nearest_center) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model_loaded) {
        return make_error_array(env, 1.0f);
    }

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return make_error_array(env, 2.0f);
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
        return make_error_array(env, 3.0f, static_cast<float>(info.format), 0.0f,
                                static_cast<float>(info.height), static_cast<float>(info.width));
    }

    const auto t_start = std::chrono::steady_clock::now();

    const int img_w = static_cast<int>(info.width);
    const int img_h = static_cast<int>(info.height);
    const float scale = std::min(static_cast<float>(kModelInputSize) / static_cast<float>(img_w),
                                 static_cast<float>(kModelInputSize) / static_cast<float>(img_h));
    const int new_w = std::clamp(static_cast<int>(std::lround(img_w * scale)), 1, kModelInputSize);
    const int new_h = std::clamp(static_cast<int>(std::lround(img_h * scale)), 1, kModelInputSize);
    const int pad_left = (kModelInputSize - new_w) / 2;
    const int pad_top = (kModelInputSize - new_h) / 2;
    const float scale_x = static_cast<float>(new_w) / static_cast<float>(img_w);
    const float scale_y = static_cast<float>(new_h) / static_cast<float>(img_h);

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(
            env, bitmap, ncnn::Mat::PIXEL_RGB, new_w, new_h);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return make_error_array(env, 4.0f);
    }
    if (in.empty()) {
        return make_error_array(env, 4.0f);
    }

    ncnn::Mat out_mat;
    float net_ms = 0.0f;
    float err_ret = 0.0f, err_dims = 0.0f, err_h = 0.0f, err_w = 0.0f;
    if (!run_net(in, 0, false, img_w, img_h, pad_left, pad_top,
                 scale_x, scale_y, true, t_start, out_mat, net_ms,
                 err_ret, err_dims, err_h, err_w)) {
        return make_error_array(env, 5.0f, err_ret, err_dims, err_h, err_w);
    }

    const float* rows[kOutRows];
    for (int r = 0; r < kOutRows; ++r) {
        rows[r] = g_cached.data.data() + static_cast<size_t>(r) * static_cast<size_t>(kNumAnchors);
    }
    return postprocess_and_pack(
            env, rows, img_w, img_h, scale_x, scale_y, pad_left, pad_top, net_ms,
            conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_hzau_tomatoar_YoloDetector_detectRgba(
        JNIEnv* env,
        jobject thiz,
        jobject rgba_buffer,
        jint buf_pos,
        jint buf_limit,
        jint buf_w,
        jint buf_h,
        jint row_stride,
        jint crop_left,
        jint crop_top,
        jint crop_w,
        jint crop_h,
        jint rotation_degrees,
        jboolean mirror_horizontal,
        jfloat conf_thresh,
        jfloat iou_thresh,
        jboolean agnostic_fruit_nms,
        jboolean assign_nearest_center) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model_loaded) {
        return make_error_array(env, 1.0f);
    }

    const int rot = ((rotation_degrees % 360) + 360) % 360;
    if (rot != 0 && rot != 90 && rot != 180 && rot != 270) {
        return make_error_array(env, 6.0f, static_cast<float>(rotation_degrees));
    }

    const int64_t bw64 = static_cast<int64_t>(buf_w);
    const int64_t bh64 = static_cast<int64_t>(buf_h);
    const int64_t cl64 = static_cast<int64_t>(crop_left);
    const int64_t ct64 = static_cast<int64_t>(crop_top);
    const int64_t cw64 = static_cast<int64_t>(crop_w);
    const int64_t ch64 = static_cast<int64_t>(crop_h);
    const int64_t stride64 = static_cast<int64_t>(row_stride);

    if (!rgba_buffer || bw64 <= 0 || bh64 <= 0 ||
        cw64 <= 0 || ch64 <= 0 ||
        cl64 < 0 || ct64 < 0 ||
        cl64 + cw64 > bw64 || ct64 + ch64 > bh64 ||
        stride64 < bw64 * 4LL) {
        return make_error_array(env, 6.0f, static_cast<float>(row_stride), 0.0f,
                                static_cast<float>(buf_h), static_cast<float>(buf_w));
    }

    const void* raw_ptr = env->GetDirectBufferAddress(rgba_buffer);
    const jlong cap = env->GetDirectBufferCapacity(rgba_buffer);
    if (!raw_ptr || cap <= 0) {
        return make_error_array(env, 6.0f, -1.0f);
    }

    const int64_t pos64 = static_cast<int64_t>(buf_pos);
    const int64_t limit64 = static_cast<int64_t>(buf_limit);
    if (pos64 < 0 || limit64 < pos64 || limit64 > static_cast<int64_t>(cap)) {
        return make_error_array(env, 6.0f, -2.0f, static_cast<float>(pos64),
                                static_cast<float>(limit64), static_cast<float>(cap));
    }

    const int64_t last_row = ct64 + ch64 - 1;
    const int64_t last_row_bytes = (cl64 + cw64) * 4LL;
    const int64_t required_bytes = last_row * stride64 + last_row_bytes;
    if (pos64 + required_bytes > limit64) {
        return make_error_array(env, 6.0f, static_cast<float>(required_bytes), 0.0f,
                                static_cast<float>(limit64 - pos64), static_cast<float>(row_stride));
    }

    const auto t_start = std::chrono::steady_clock::now();

    const bool swap_wh = (rot == 90 || rot == 270);
    const int upright_w = swap_wh ? crop_h : crop_w;
    const int upright_h = swap_wh ? crop_w : crop_h;

    const float scale = std::min(static_cast<float>(kModelInputSize) / static_cast<float>(upright_w),
                                 static_cast<float>(kModelInputSize) / static_cast<float>(upright_h));
    const int new_w = std::clamp(static_cast<int>(std::lround(upright_w * scale)), 1, kModelInputSize);
    const int new_h = std::clamp(static_cast<int>(std::lround(upright_h * scale)), 1, kModelInputSize);
    const int pad_left = (kModelInputSize - new_w) / 2;
    const int pad_top = (kModelInputSize - new_h) / 2;
    const float scale_x = static_cast<float>(new_w) / static_cast<float>(upright_w);
    const float scale_y = static_cast<float>(new_h) / static_cast<float>(upright_h);

    const int pre_rot_w = swap_wh ? new_h : new_w;
    const int pre_rot_h = swap_wh ? new_w : new_h;

    const auto* pixels = static_cast<const unsigned char*>(raw_ptr) + pos64;
    ncnn::Mat pre_rot = ncnn::Mat::from_pixels_roi_resize(
            pixels,
            ncnn::Mat::PIXEL_RGBA2RGB,
            buf_w,
            buf_h,
            row_stride,
            crop_left,
            crop_top,
            crop_w,
            crop_h,
            pre_rot_w,
            pre_rot_h,
            g_yolo_net.opt.blob_allocator);
    if (pre_rot.empty()) {
        return make_error_array(env, 4.0f);
    }

    ncnn::Mat out_mat;
    float net_ms = 0.0f;
    float err_ret = 0.0f, err_dims = 0.0f, err_h = 0.0f, err_w = 0.0f;
    if (!run_net(pre_rot, rot, mirror_horizontal == JNI_TRUE, upright_w, upright_h,
                 pad_left, pad_top, scale_x, scale_y, false, t_start,
                 out_mat, net_ms, err_ret, err_dims, err_h, err_w)) {
        return make_error_array(env, 5.0f, err_ret, err_dims, err_h, err_w);
    }

    const float* rows[kOutRows];
    for (int r = 0; r < kOutRows; ++r) {
        rows[r] = out_mat.row(r);
    }
    return postprocess_and_pack(
            env, rows, upright_w, upright_h, scale_x, scale_y, pad_left, pad_top, net_ms,
            conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_hzau_tomatoar_YoloDetector_repostprocess(
        JNIEnv* env,
        jobject thiz,
        jfloat conf_thresh,
        jfloat iou_thresh,
        jboolean agnostic_fruit_nms,
        jboolean assign_nearest_center) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model_loaded) {
        return make_error_array(env, 1.0f);
    }
    if (!g_cached.valid || g_cached.data.size() != static_cast<size_t>(kOutRows) * static_cast<size_t>(kNumAnchors)) {
        return make_error_array(env, 7.0f);
    }

    const float* rows[kOutRows];
    for (int r = 0; r < kOutRows; ++r) {
        rows[r] = g_cached.data.data() + static_cast<size_t>(r) * static_cast<size_t>(kNumAnchors);
    }
    return postprocess_and_pack(
            env, rows, g_cached.img_w, g_cached.img_h,
            g_cached.scale_x, g_cached.scale_y, g_cached.pad_left, g_cached.pad_top,
            g_cached.infer_ms,
            conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_hzau_tomatoar_YoloDetector_clearCache(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cached.valid = false;
    std::vector<float>().swap(g_cached.data);
    g_input_pad.release();
    reset_pad_border_cache();
}
