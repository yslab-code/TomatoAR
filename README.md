# TomatoAR 采收相机（DGTruss-YOLO Android 部署工程）

基于 **DGTruss-YOLO**（`C3k2_ScConv + MFM-MAFPN + Efficient_Detect`）与 **NCNN** 的实时樱桃番茄果穗成熟度评估 Android 应用，支持手机端与 AR 眼镜投屏使用。

---

## 一、 模型与分级口径

- **内置模型**：`app/src/main/assets/best.ncnn.param`（47 KB）与 `best.ncnn.bin`（4.46 MiB，FP16 存储）
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

## 二、 使用 GitHub Actions 云端自动编译 APK（本机零 SDK 安装）

1. 在 GitHub 上新建一个 **Private（私有）仓库**（例如 `TomatoAR`）。
2. 在本目录（`F:\labagent\geminilab\TomatoAR`）执行以下命令提交并推送代码：
   ```bash
   git init -b main
   git add .
   git update-index --chmod=+x gradlew
   git commit -m "Initial TomatoAR Android NCNN project"
   git remote add origin <你的仓库地址>
   git push -u origin main
   ```
3. 打开 GitHub 仓库的 **Actions** 标签页，等待 `Build TomatoAR Debug APK` 运行完成（约 3–5 分钟）。工作流会自动安装 **NDK r30**、下载并校验 `ncnn-20261020-android-vulkan.zip`（SHA-256）、并编译生成 APK。
4. 在完成的运行记录底部 **Artifacts** 处下载 **`TomatoAR-debug-apk`**，解压出 `app-debug.apk` 传至安卓手机直接安装即可。

### GitHub Actions 配额说明
- **公开仓库**：标准 Linux 运行器不限构建时长。
- **私有仓库（推荐，保护模型权重与代码）**：使用账号套餐包含的月度免费额度（GitHub Free 账户每月含 **2,000 分钟** Linux 构建时长与 **500 MB** Artifact 存储空间，默认消费上限为 `$0`，用尽额度后仅会暂停当月构建而不会自动扣费）。
- 工作流已设置 `retention-days: 7`，编译产物在 7 天到期后会自动清理释放存储空间，在保留期内仍会计入上述 500 MB 存储配额。

---

## 三、 主要交互功能

- **实时取景**：左上方状态胶囊可点击查看实际后端、采集尺寸、耗时与实测 FPS；右上方调节图标可配置显示图层、置信度/IoU 阈值、NMS 策略、重叠果穗归属规则（含桌面对照模式）、CPU/Vulkan 切换与分辨率。
- **定格与保存分离**：点击底部白色圆形快门即可锁定当前检测帧与标注结果，可点选果穗查看详单或开关图层后再点击 **「保存标注图」** 导出纯净带标注图片到手机相册 `Pictures/TomatoAR/`。
- **内置样本图 & 相册测图**：支持从相册挑图，也可在参数面板点击 **「加载内置大棚番茄样本图」** 直接用验证集真实图像体验检测与分级效果。

---

## 四、 端侧工程与算法优化

- **Zero-Bitmap NCNN 直通预处理**：实时取景路径通过 `detectRgba` 直接读取 CameraX Direct `ByteBuffer`，经 `from_pixels_roi_resize` 缩放至 $\le 512\times 512$ 后在 CHW 域完成旋转与镜像，避免逐帧全尺寸 Bitmap 分配与 `rowStride` 尾行越界风险。
- **ScConv FP32 数值安全与 NaN 探针**：显式禁用 FP16/BF16 存储与运算以保护 `C3k2_ScConv/SRU` 中的 `1e-10` 防除零项与 `2048.0` 方差缩放项，并在模型加载试推理阶段校验输出全有限（`std::isfinite`）。
- **缓存零延迟重后处理**：定格与相册模式下缓存最新 `8×5376` 输出张量，调节置信度、IoU、NMS 策略与归属规则时直接复用缓存执行 `repostprocess`（微秒级响应），并支持防抖合并与 `SharedPreferences` 持久化。
- **移动端大核调度与内存池**：自动识别 big.LITTLE 大核绑定线程与 `powersave=2` 策略，启用 `lightmode` 与 `PoolAllocator` 复用推理缓冲区，开启 C++ `-O3` 编译优化与 Android 15+ 16 KB 内存页链接对齐。
- **相机节能与渲染增强**：对齐 `Preview` 与 `ImageAnalysis` 宽高比策略；定格/相册模式自动释放相机；支持实时选中果穗 IoU 追踪、多果穗标签碰撞避让以及导出图底部汇总自适应排版。
