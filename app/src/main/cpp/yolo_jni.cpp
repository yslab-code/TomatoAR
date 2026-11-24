#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
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

std::mutex g_mutex;
ncnn::Net g_yolo_net;
ncnn::PoolAllocator g_blob_pool_allocator;
ncnn::PoolAllocator g_workspace_pool_allocator;
bool g_model_loaded = false;
bool g_gpu_active = false;
bool g_gpu_instance_created = false;
CachedOutput g_cached;

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
    const std::string prec = enable_fp16 ? " (FP16)" : " (FP32)";
    if (use_gpu) {
        return "Vulkan GPU" + prec;
    }
    if (big_cores >= 3) {
        return "CPU " + std::to_string(cpu_threads) + "大核" + prec;
    }
    return "CPU " + std::to_string(cpu_threads) + "线程" + prec;
}

bool load_and_probe(
        AAssetManager* mgr,
        bool use_gpu,
        bool enable_fp16,
        std::string& err_msg,
        std::string& backend_label) {
    g_yolo_net.clear();
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    g_model_loaded = false;
    g_gpu_active = false;
    g_cached.valid = false;

    if (use_gpu) {
#if NCNN_VULKAN
        if (!g_gpu_instance_created) {
            ncnn::create_gpu_instance();
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
    const int little_cores = ncnn::get_little_cpu_count();
    const int total_cores = ncnn::get_cpu_count();
    const int non_little_cores = (little_cores > 0 && total_cores > little_cores)
            ? (total_cores - little_cores)
            : total_cores;
    const int cpu_threads = std::clamp(non_little_cores >= 3 ? non_little_cores : total_cores, 1, 4);
    if (!use_gpu) {
        ncnn::set_cpu_powersave(big_cores >= 3 ? 2 : 0);
    }

    ncnn::Option opt;
    opt.use_vulkan_compute = use_gpu;
    opt.num_threads = cpu_threads;
    opt.lightmode = true;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_fp16_packed = enable_fp16;
    opt.use_fp16_storage = enable_fp16;
    opt.use_fp16_arithmetic = enable_fp16;
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
        g_yolo_net.clear();
        return false;
    }
    const int m_ret = g_yolo_net.load_model(mgr, "best.ncnn.bin");
    if (m_ret != 0) {
        err_msg = "load_model 失败 (ret=" + std::to_string(m_ret) + ")";
        g_yolo_net.clear();
        return false;
    }

    // Dual-probe inference (0.0 and 0.5) to validate Vulkan/FP16 pipeline, shape, and finite math
    const float probe_vals[2] = {0.0f, 0.5f};
    for (float pv : probe_vals) {
        ncnn::Mat probe_in(kModelInputSize, kModelInputSize, 3);
        probe_in.fill(pv);
        ncnn::Extractor ex = g_yolo_net.create_extractor();
        if (ex.input("in0", probe_in) != 0) {
            err_msg = "试推理输入节点 in0 绑定失败";
            g_yolo_net.clear();
            return false;
        }
        ncnn::Mat probe_out;
        const int ex_ret = ex.extract("out0", probe_out);
        if (ex_ret != 0) {
            err_msg = "试推理 extract(out0) 失败 (ret=" + std::to_string(ex_ret) + ")";
            g_yolo_net.clear();
            return false;
        }
        if (probe_out.dims != 2 || probe_out.h != kOutRows || probe_out.w != kNumAnchors) {
            err_msg = "模型输出形状异常: dims=" + std::to_string(probe_out.dims) +
                      " h=" + std::to_string(probe_out.h) + " w=" + std::to_string(probe_out.w) +
                      " (期望 2D 8x5376)";
            g_yolo_net.clear();
            return false;
        }
        for (int r = 0; r < probe_out.h; ++r) {
            const float* row_ptr = probe_out.row(r);
            for (int c = 0; c < probe_out.w; ++c) {
                if (!std::isfinite(row_ptr[c])) {
                    err_msg = std::string(enable_fp16 ? "FP16" : "FP32") + " 试推理输出含 NaN/Inf";
                    g_yolo_net.clear();
                    return false;
                }
            }
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

ncnn::Mat rotate_and_mirror_chw(const ncnn::Mat& src, int rot_deg, bool mirror_horizontal) {
    if (src.empty() || src.c != 3) {
        return ncnn::Mat();
    }
    if (rot_deg == 0 && !mirror_horizontal) {
        return src;
    }
    const int src_w = src.w;
    const int src_h = src.h;
    const int dst_w = (rot_deg == 90 || rot_deg == 270) ? src_h : src_w;
    const int dst_h = (rot_deg == 90 || rot_deg == 270) ? src_w : src_h;

    ncnn::Mat dst(dst_w, dst_h, 3);
    if (dst.empty()) {
        return ncnn::Mat();
    }

    const int mode = ((rot_deg / 90) & 3) * 2 + (mirror_horizontal ? 1 : 0);
    for (int c = 0; c < 3; ++c) {
        const float* sptr = src.channel(c);
        float* dptr = dst.channel(c);
        switch (mode) {
            case 1: { // rot 0 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + sy * dst_w;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[dst_w - 1 - sx] = srow[sx];
                    }
                }
                break;
            }
            case 2: { // rot 90
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    const int dx = src_h - 1 - sy;
                    for (int sx = 0; sx < src_w; ++sx) {
                        dptr[sx * dst_w + dx] = srow[sx];
                    }
                }
                break;
            }
            case 3: { // rot 90 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    const int dx = sy;
                    for (int sx = 0; sx < src_w; ++sx) {
                        dptr[sx * dst_w + dx] = srow[sx];
                    }
                }
                break;
            }
            case 4: { // rot 180
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (src_h - 1 - sy) * dst_w;
                    for (int sx = 0; sx < src_w; ++sx) {
                        drow[src_w - 1 - sx] = srow[sx];
                    }
                }
                break;
            }
            case 5: { // rot 180 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    float* drow = dptr + (src_h - 1 - sy) * dst_w;
                    std::memcpy(drow, srow, static_cast<size_t>(src_w) * sizeof(float));
                }
                break;
            }
            case 6: { // rot 270
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    const int dx = sy;
                    for (int sx = 0; sx < src_w; ++sx) {
                        dptr[(src_w - 1 - sx) * dst_w + dx] = srow[sx];
                    }
                }
                break;
            }
            case 7: { // rot 270 + mirror
                for (int sy = 0; sy < src_h; ++sy) {
                    const float* srow = sptr + sy * src_w;
                    const int dx = src_h - 1 - sy;
                    for (int sx = 0; sx < src_w; ++sx) {
                        dptr[(src_w - 1 - sx) * dst_w + dx] = srow[sx];
                    }
                }
                break;
            }
            default:
                break;
        }
    }
    return dst;
}

bool run_net_and_cache(
        const ncnn::Mat& resized_rgb,
        int upright_w,
        int upright_h,
        int pad_left,
        int pad_right,
        int pad_top,
        int pad_bottom,
        float scale_x,
        float scale_y,
        const std::chrono::steady_clock::time_point& t_start,
        float& out_ret,
        float& out_dims,
        float& out_h,
        float& out_w) {
    if (resized_rgb.empty()) {
        out_ret = -2.0f;
        return false;
    }

    ncnn::Mat in_pad;
    ncnn::copy_make_border(resized_rgb, in_pad, pad_top, pad_bottom, pad_left, pad_right,
                           ncnn::BORDER_CONSTANT, 114.0f, g_yolo_net.opt);
    if (in_pad.empty() || in_pad.w != kModelInputSize || in_pad.h != kModelInputSize) {
        out_ret = -3.0f;
        return false;
    }
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in_pad.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = g_yolo_net.create_extractor();
    if (ex.input("in0", in_pad) != 0) {
        out_ret = -1.0f;
        out_dims = 0.0f;
        out_h = 0.0f;
        out_w = 0.0f;
        return false;
    }
    ncnn::Mat out;
    const int ex_ret = ex.extract("out0", out);
    if (ex_ret != 0 || out.dims != 2 || out.h != kOutRows || out.w != kNumAnchors) {
        out_ret = static_cast<float>(ex_ret);
        out_dims = static_cast<float>(out.dims);
        out_h = static_cast<float>(out.h);
        out_w = static_cast<float>(out.w);
        return false;
    }

    g_cached.data.resize(static_cast<size_t>(kOutRows) * static_cast<size_t>(kNumAnchors));
    for (int r = 0; r < kOutRows; ++r) {
        std::memcpy(
                g_cached.data.data() + static_cast<size_t>(r) * static_cast<size_t>(kNumAnchors),
                out.row(r),
                static_cast<size_t>(kNumAnchors) * sizeof(float));
    }
    g_cached.img_w = upright_w;
    g_cached.img_h = upright_h;
    g_cached.scale_x = scale_x;
    g_cached.scale_y = scale_y;
    g_cached.pad_left = pad_left;
    g_cached.pad_top = pad_top;
    const auto t_net_end = std::chrono::steady_clock::now();
    g_cached.infer_ms = std::chrono::duration<float, std::milli>(t_net_end - t_start).count();
    g_cached.valid = true;
    return true;
}

jfloatArray postprocess_and_pack(
        JNIEnv* env,
        const CachedOutput& cached,
        float conf_thresh,
        float iou_thresh,
        bool agnostic_fruit_nms,
        bool assign_nearest_center,
        const std::chrono::steady_clock::time_point& t_start) {
    const float* row_cx = cached.data.data() + 0 * kNumAnchors;
    const float* row_cy = cached.data.data() + 1 * kNumAnchors;
    const float* row_bw = cached.data.data() + 2 * kNumAnchors;
    const float* row_bh = cached.data.data() + 3 * kNumAnchors;
    const float* row_c0 = cached.data.data() + 4 * kNumAnchors;
    const float* row_c1 = cached.data.data() + 5 * kNumAnchors;
    const float* row_c2 = cached.data.data() + 6 * kNumAnchors;
    const float* row_c3 = cached.data.data() + 7 * kNumAnchors;

    std::vector<DetBox> truss_cands;
    std::vector<DetBox> fruit_cands;
    truss_cands.reserve(64);
    fruit_cands.reserve(256);

    const float fw = static_cast<float>(cached.img_w);
    const float fh = static_cast<float>(cached.img_h);
    const float f_pad_l = static_cast<float>(cached.pad_left);
    const float f_pad_t = static_cast<float>(cached.pad_top);
    const float inv_sx = cached.scale_x > 0.0f ? (1.0f / cached.scale_x) : 1.0f;
    const float inv_sy = cached.scale_y > 0.0f ? (1.0f / cached.scale_y) : 1.0f;

    for (int i = 0; i < kNumAnchors; ++i) {
        const float s0 = row_c0[i];
        const float s1 = row_c1[i];
        const float s2 = row_c2[i];
        const float s3 = row_c3[i];

        if (!std::isfinite(s0) || !std::isfinite(s1) || !std::isfinite(s2) || !std::isfinite(s3)) {
            continue;
        }

        int best_cls = 0;
        float best_score = s0;
        if (s1 > best_score) { best_score = s1; best_cls = 1; }
        if (s2 > best_score) { best_score = s2; best_cls = 2; }
        if (s3 > best_score) { best_score = s3; best_cls = 3; }

        if (best_score < conf_thresh) {
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

        DetBox box{x1, y1, x2, y2, best_score, best_cls, -1};
        if (best_cls == 0) {
            truss_cands.push_back(box);
        } else {
            fruit_cands.push_back(box);
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
    std::sort(trusses.begin(), trusses.end(), [](const DetBox& a, const DetBox& b) {
        if (std::abs(a.x1 - b.x1) > 1.0f) return a.x1 < b.x1;
        return a.y1 < b.y1;
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

    const auto t_end = std::chrono::steady_clock::now();
    const float infer_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::vector<float> packed;
    packed.reserve(4 + summaries.size() * 10 + fruits.size() * 7);
    packed.push_back(1.0f);
    packed.push_back(infer_ms);
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
    g_yolo_net.clear();
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    g_model_loaded = false;
    g_cached.valid = false;
    g_cached.data.clear();
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
        std::string gpu_fp16_err;
        if (load_and_probe(mgr, true, true, gpu_fp16_err, backend)) {
            const std::string resp = "OK|" + backend + "|";
            return env->NewStringUTF(resp.c_str());
        }
        std::string gpu_fp32_err;
        if (load_and_probe(mgr, true, false, gpu_fp32_err, backend)) {
            const std::string resp = "OK|" + backend + "|";
            return env->NewStringUTF(resp.c_str());
        }

        const std::string gpu_err = gpu_fp16_err + "; FP32(" + gpu_fp32_err + ")";
        std::string cpu_fp16_err;
        std::string cpu_backend;
        if (load_and_probe(mgr, false, true, cpu_fp16_err, cpu_backend)) {
            const std::string resp = "FALLBACK|" + cpu_backend + "|" + gpu_err;
            return env->NewStringUTF(resp.c_str());
        }
        std::string cpu_fp32_err;
        if (load_and_probe(mgr, false, false, cpu_fp32_err, cpu_backend)) {
            const std::string resp = "FALLBACK|" + cpu_backend + "|" + gpu_err;
            return env->NewStringUTF(resp.c_str());
        }
        const std::string resp = "ERROR|未加载|GPU(" + gpu_err + "); CPU(" + cpu_fp16_err + "; " + cpu_fp32_err + ")";
        return env->NewStringUTF(resp.c_str());
    }

    std::string cpu_fp16_err;
    if (load_and_probe(mgr, false, true, cpu_fp16_err, backend)) {
        const std::string resp = "OK|" + backend + "|";
        return env->NewStringUTF(resp.c_str());
    }
    std::string cpu_fp32_err;
    if (load_and_probe(mgr, false, false, cpu_fp32_err, backend)) {
        const std::string resp = "OK|" + backend + "|";
        return env->NewStringUTF(resp.c_str());
    }

    const std::string resp = "ERROR|未加载|" + cpu_fp16_err + "; " + cpu_fp32_err;
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
    const int new_w = std::max(1, static_cast<int>(std::lround(img_w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(img_h * scale)));
    const int pad_left = (kModelInputSize - new_w) / 2;
    const int pad_right = kModelInputSize - new_w - pad_left;
    const int pad_top = (kModelInputSize - new_h) / 2;
    const int pad_bottom = kModelInputSize - new_h - pad_top;
    const float scale_x = static_cast<float>(new_w) / static_cast<float>(img_w);
    const float scale_y = static_cast<float>(new_h) / static_cast<float>(img_h);

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(
            env, bitmap, ncnn::Mat::PIXEL_RGB, new_w, new_h);
    if (in.empty()) {
        return make_error_array(env, 4.0f);
    }

    float err_ret = 0.0f, err_dims = 0.0f, err_h = 0.0f, err_w = 0.0f;
    if (!run_net_and_cache(in, img_w, img_h, pad_left, pad_right, pad_top, pad_bottom,
                           scale_x, scale_y, t_start, err_ret, err_dims, err_h, err_w)) {
        return make_error_array(env, 5.0f, err_ret, err_dims, err_h, err_w);
    }

    return postprocess_and_pack(
            env, g_cached, conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE, t_start);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_hzau_tomatoar_YoloDetector_detectRgba(
        JNIEnv* env,
        jobject thiz,
        jobject rgba_buffer,
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

    if (!rgba_buffer || buf_w <= 0 || buf_h <= 0 ||
        crop_w <= 0 || crop_h <= 0 ||
        crop_left < 0 || crop_top < 0 ||
        crop_left + crop_w > buf_w || crop_top + crop_h > buf_h ||
        row_stride < buf_w * 4) {
        return make_error_array(env, 6.0f, static_cast<float>(row_stride), 0.0f,
                                static_cast<float>(buf_h), static_cast<float>(buf_w));
    }

    const void* raw_ptr = env->GetDirectBufferAddress(rgba_buffer);
    const jlong cap = env->GetDirectBufferCapacity(rgba_buffer);
    if (!raw_ptr || cap <= 0) {
        return make_error_array(env, 6.0f, -1.0f);
    }

    const int64_t last_row = static_cast<int64_t>(crop_top) + static_cast<int64_t>(crop_h) - 1;
    const int64_t last_row_bytes = static_cast<int64_t>(crop_left + crop_w) * 4LL;
    const int64_t required_bytes = last_row * static_cast<int64_t>(row_stride) + last_row_bytes;
    if (required_bytes > static_cast<int64_t>(cap)) {
        return make_error_array(env, 6.0f, static_cast<float>(required_bytes), 0.0f,
                                static_cast<float>(cap), static_cast<float>(row_stride));
    }

    const auto t_start = std::chrono::steady_clock::now();

    const bool swap_wh = (rot == 90 || rot == 270);
    const int upright_w = swap_wh ? crop_h : crop_w;
    const int upright_h = swap_wh ? crop_w : crop_h;

    const float scale = std::min(static_cast<float>(kModelInputSize) / static_cast<float>(upright_w),
                                 static_cast<float>(kModelInputSize) / static_cast<float>(upright_h));
    const int new_w = std::max(1, static_cast<int>(std::lround(upright_w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(upright_h * scale)));
    const int pad_left = (kModelInputSize - new_w) / 2;
    const int pad_right = kModelInputSize - new_w - pad_left;
    const int pad_top = (kModelInputSize - new_h) / 2;
    const int pad_bottom = kModelInputSize - new_h - pad_top;
    const float scale_x = static_cast<float>(new_w) / static_cast<float>(upright_w);
    const float scale_y = static_cast<float>(new_h) / static_cast<float>(upright_h);

    const int pre_rot_w = swap_wh ? new_h : new_w;
    const int pre_rot_h = swap_wh ? new_w : new_h;

    const auto* pixels = static_cast<const unsigned char*>(raw_ptr);
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
            pre_rot_h);
    if (pre_rot.empty()) {
        return make_error_array(env, 4.0f);
    }

    ncnn::Mat upright_rgb = rotate_and_mirror_chw(pre_rot, rot, mirror_horizontal == JNI_TRUE);
    if (upright_rgb.empty()) {
        return make_error_array(env, 4.0f);
    }

    float err_ret = 0.0f, err_dims = 0.0f, err_h = 0.0f, err_w = 0.0f;
    if (!run_net_and_cache(upright_rgb, upright_w, upright_h, pad_left, pad_right, pad_top, pad_bottom,
                           scale_x, scale_y, t_start, err_ret, err_dims, err_h, err_w)) {
        return make_error_array(env, 5.0f, err_ret, err_dims, err_h, err_w);
    }

    return postprocess_and_pack(
            env, g_cached, conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE, t_start);
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

    const auto t_start = std::chrono::steady_clock::now();
    return postprocess_and_pack(
            env, g_cached, conf_thresh, iou_thresh,
            agnostic_fruit_nms == JNI_TRUE, assign_nearest_center == JNI_TRUE, t_start);
}
