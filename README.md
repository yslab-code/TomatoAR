# TomatoAR 采收相机（DGTruss-YOLO Android 部署工程）

基于 **DGTruss-YOLO**（`C3k2_ScConv + MFM-MAFPN + Efficient_Detect`）与 **NCNN** 的实时樱桃番茄果穗成熟度评估 Android 应用，支持手机端与 AR 眼镜投屏使用。

---

## 一、 模型与分级口径

- **内置模型**：`app/src/main/assets/best.ncnn.param`（48 KB，已将 MFM 4D `Reduction` 改写为 Vulkan 安全的 3D `Reshape + Slice + BinaryOp` 加法链）与 `best.ncnn.bin`（4.46 MiB，未压缩直读）
- **输入 / 输出节点**：`in0` (`[1, 3, 512, 512]` RGB Letterbox 归一化) → `out0` (`[1, 8, 5376]`)
- **类别定义**：`0: tomato`（果穗）、`1: ripe`（成熟果 `#FF3B30`）、`2: pink`（转色果 `#FF9500`）、`3: green`（未成熟果 `#34C759`）
- **果穗成熟度指数与五级判定（表 1）**：
  $$I = \frac{N_{\text{ripe}} + 0.5 \cdot N_{\text{turning}}}{N_{\text{total}}}$$
  - **A 级**（$0.80 < I \le 1.00$）：立即采收或销售
  - **B 级**（$0.60 < I \le 0.80$）：可采收（短途物流）
  - **C 级**（$0.40 < I \le 0.60$）：可采收（中程备货）
  - **D 级**（$0.20 < I \le 0.40$）：延后采收或远途物流
  - **E 级**（$0.00 \le I \le 0.20$）：不可采收，继续培育
  - **待检**（$N_{\text{total}} = 0$）：串内暂未检出单果

---

## 二、 构建 APK（云端 GitHub Actions / 本地 Gradle）

### 方式 A：GitHub Actions 云端自动编译（本机零 SDK 安装）
1. 推送代码到仓库 `main`/`master` 分支或手动触发 `Build TomatoAR Debug APK` 工作流。
2. 工作流自动安装 **NDK r30**、下载并缓存校验 `ncnn-20261020-android-vulkan.zip`（SHA-256）、执行单元测试 (`testDebugUnitTest`)、编译 APK 并校验所有 `.so` 的 **16 KB ELF 页对齐**。
3. 在完成的运行记录底部 **Artifacts** 下载 **`TomatoAR-debug-apk`** 解压安装即可。

### 方式 B：本地 Android Studio / Gradle 编译
1. 安装 Android SDK (API 34)、CMake `3.22.1` 及 NDK (r26–r30)。
2. 下载 `ncnn-20261020-android-vulkan.zip` 并解压至 `app/src/main/cpp/ncnn-android-vulkan/`。
3. 在项目根目录执行：
   ```bash
   ./gradlew testDebugUnitTest assembleDebug
   ```

---

## 三、 主要交互功能

- **实时取景**：左上方状态胶囊可点击查看实际后端、输入尺寸、耗时与实测 FPS；右上方调节图标可配置显示图层、置信度/IoU 阈值、NMS 策略、重叠果穗归属规则（含桌面对照模式）、CPU/Vulkan 切换与预览分辨率。
- **定格与保存分离**：点击底部白色圆形快门即可即时截取预览并异步进行高精度复检（同时在归一化坐标下保留当前选中果穗）；点击 **「保存标注图」** 可导出底部分栏式（不遮挡底部果实）、含时间戳与色标图例的带标注图片到手机相册 `Pictures/TomatoAR/`。
- **内置样本图 & 相册测图**：支持通过 `ImageDecoder`（API 28+）/ EXIF 自动纠偏从相册挑图，也可在参数面板点击 **「加载内置大棚番茄样本图」** 直接用验证集真实图像体验检测与分级效果。

---

## 四、 端侧工程与算法优化

- **Vulkan 3D 算子改写与数值交叉探针**：将 MFM 多分支注意力的 4D `Reduction` 求和层等效改写为 3D `Reshape + Slice + BinaryOp`，绕过 NCNN Vulkan `D*H*W >= 65536` 的计算派发上限；启用 **FP16 打包存储 + FP32 累加** (`use_fp16_packed/storage=true, use_fp16_arithmetic=false`) 保护 `ScConv/SRU` 精度，并在 `initModel` 时通过零输入 + 非均匀梯度双探针及 Vulkan/CPU 数值一致性比对防止静默驱动异常。
- **融合预处理与无拷贝实时解码**：实时取景路径通过 `detectRgba` 直接按 `ByteBuffer.position/limit` 读取 CameraX 缓冲区，在 `fill_letterbox_input` 中融合 16×16 分块缓存友好旋转镜像、Letterbox 填充与 `1/255` 归一化至复用的 `512×512×3` 输入缓冲区；实时流跳过 `g_cached` 冗余拷贝，独立处理 `truss` 与 `fruit` 类别置信度避免同锚点漏检。
- **动态调度与线程池隔离**：`big_cores >= 2` 时严格匹配大核线程数与 `powersave=2`（消除 2+6 异构核拖尾卡顿）；将 CameraX 分析线程、`inferenceExecutor` 与 `ioExecutor` 三路隔离，并严格校验 `activeSessionId` 与生命周期安全退出。
- **全量 16 KB 内存页兼容与渲染优化**：升级 CameraX 至 `1.4.2`（修复 `libimage_processing_util_jni.so` 4 KB 对齐问题）配合 `-Wl,-z,max-page-size=16384`、隐藏符号裁剪与模型资产 `noCompress`；`ArOverlayView` 实现上下安全区避让、堆叠徽章两阶段精确/中心距命中测试与零 `onDraw` 堆分配。
