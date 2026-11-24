package com.hzau.tomatoar

import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.util.Locale
import kotlin.math.roundToInt

data class InitResult(
    val success: Boolean,
    val fallback: Boolean,
    val activeBackend: String,
    val reason: String
)

data class DetectorConfig(
    val confThresh: Float = 0.35f,
    val iouThresh: Float = 0.70f,
    val agnosticFruitNms: Boolean = true,
    val assignNearestCenter: Boolean = true,
    val showFruitBoxes: Boolean = true,
    val showFruitLabels: Boolean = false,
    val largeTextForGlasses: Boolean = false,
    val useGpu: Boolean = false,
    val captureResIndex: Int = 1 // 0: 640x480, 1: 1280x720, 2: 1920x1080
)

data class TrussItem(
    val id: Int,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val ripeCount: Int,
    val turningCount: Int,
    val immatureCount: Int,
    val gradeCode: Int, // 0: 待检, 1: A, 2: B, 3: C, 4: D, 5: E
    val maturity: Float // < 0 when empty
) {
    val idText: String = "#$id"

    val totalFruits: Int get() = ripeCount + turningCount + immatureCount

    val gradeLetter: String
        get() = when (gradeCode) {
            1 -> "A"
            2 -> "B"
            3 -> "C"
            4 -> "D"
            5 -> "E"
            else -> "-"
        }

    val shortAdvice: String
        get() = when (gradeCode) {
            1 -> "立即采收"
            2 -> "短途物流"
            3 -> "中程备货"
            4 -> "延后/远途"
            5 -> "继续培育"
            else -> "待检"
        }

    val fullAdvice: String
        get() = when (gradeCode) {
            1 -> "熟果主导串 · 立即采收或销售"
            2 -> "晚转色主导串 · 可采收（短途物流）"
            3 -> "中转色主导串 · 可采收（中程备货）"
            4 -> "早转色主导串 · 延后采收或远途物流"
            5 -> "未熟主导串 · 不可采收，继续培育"
            else -> "串内暂未检出单果，请调整视角重检"
        }

    val maturityPercentText: String
        get() = if (totalFruits > 0 && maturity >= 0f) {
            String.format(Locale.US, "%.1f%%", maturity * 100f)
        } else {
            "--"
        }
}

data class FruitItem(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val cls: Int, // 1: ripe, 2: turning(pink), 3: immature(green)
    val parentTrussIndex: Int // -1 if unassigned
) {
    val stageName: String
        get() = when (cls) {
            1 -> "成熟"
            2 -> "转色"
            3 -> "未熟"
            else -> "未知"
        }
}

data class DetectionResult(
    val success: Boolean,
    val errorMessage: String,
    val inferMs: Float,
    val trusses: List<TrussItem>,
    val fruits: List<FruitItem>,
    val errorCode: Int = 0
) {
    val assignedRipe: Int = trusses.sumOf { it.ripeCount }
    val assignedTurning: Int = trusses.sumOf { it.turningCount }
    val assignedImmature: Int = trusses.sumOf { it.immatureCount }
    val assignedTotal: Int = assignedRipe + assignedTurning + assignedImmature
    val unassignedCount: Int = (fruits.size - assignedTotal).coerceAtLeast(0)

    val overallAssignedMaturity: Float?
        get() = if (assignedTotal > 0) {
            (assignedRipe + 0.5f * assignedTurning) / assignedTotal.toFloat()
        } else {
            null
        }

    val overallAssignedPercentText: String
        get() = overallAssignedMaturity?.let {
            String.format(Locale.US, "%.1f%%", it * 100f)
        } ?: "--"
}

class YoloDetector {
    companion object {
        const val ERR_MODEL_NOT_LOADED = 1
        const val ERR_BITMAP_INFO = 2
        const val ERR_BITMAP_FORMAT = 3
        const val ERR_PREPROCESS = 4
        const val ERR_NCNN_EXTRACT = 5
        const val ERR_RGBA_BUFFER = 6
        const val ERR_NO_CACHE = 7
        const val ERR_NATIVE_LOAD = 8

        @Volatile
        private var nativeLoaded = false

        @Volatile
        private var nativeLoadError: String? = null

        private val loadLock = Any()

        internal fun ensureNativeLoaded(): String? {
            if (nativeLoaded) return null
            synchronized(loadLock) {
                if (nativeLoaded) return null
                return try {
                    System.loadLibrary("dgtruss_yolo")
                    nativeLoaded = true
                    nativeLoadError = null
                    null
                } catch (e: UnsatisfiedLinkError) {
                    val msg = "原生推理库加载失败（仅支持 arm64-v8a 设备）：${e.message ?: "UnsatisfiedLinkError"}"
                    nativeLoadError = msg
                    msg
                } catch (e: SecurityException) {
                    val msg = "原生推理库加载受限：${e.message ?: "SecurityException"}"
                    nativeLoadError = msg
                    msg
                }
            }
        }

        private fun Float.safeRoundToInt(default: Int = 0): Int {
            return if (isFinite()) roundToInt() else default
        }

        internal fun parseDetectionResult(raw: FloatArray?): DetectionResult {
            if (raw == null || raw.isEmpty()) {
                return DetectionResult(false, "JNI 返回空结果", 0f, emptyList(), emptyList(), -1)
            }
            val header = raw[0]
            if (!header.isFinite()) {
                return DetectionResult(false, "JNI 返回非有限状态头", 0f, emptyList(), emptyList(), -2)
            }
            if (header < 0f) {
                val code = raw.getOrElse(1) { 0f }.safeRoundToInt()
                val ret = raw.getOrElse(2) { 0f }.safeRoundToInt()
                val dims = raw.getOrElse(3) { 0f }.safeRoundToInt()
                val h = raw.getOrElse(4) { 0f }.safeRoundToInt()
                val w = raw.getOrElse(5) { 0f }.safeRoundToInt()
                val msg = when (code) {
                    ERR_MODEL_NOT_LOADED -> "模型尚未加载完成"
                    ERR_BITMAP_INFO -> "无法读取 Bitmap 像素信息"
                    ERR_BITMAP_FORMAT -> "Bitmap 格式需为 RGBA_8888 (实际=$ret, ${w}x${h})"
                    ERR_PREPROCESS -> "图像缩放预处理失败"
                    ERR_NCNN_EXTRACT -> "NCNN 推理或输出形状校验失败 (ret=$ret, dims=$dims, shape=${h}x${w}, 期望=8x5376)"
                    ERR_RGBA_BUFFER -> "Direct RGBA 缓冲区或步长参数无效 (ret=$ret, dims=$dims, h=$h, w=$w)"
                    ERR_NO_CACHE -> "无可用推理输出缓存"
                    else -> "未知推理错误 ($code)"
                }
                return DetectionResult(false, msg, 0f, emptyList(), emptyList(), code)
            }
            if (header < 1f || raw.size < 4) {
                return DetectionResult(false, "JNI 返回数据头不完整", 0f, emptyList(), emptyList(), -2)
            }

            val inferMs = raw[1].let { if (it.isFinite() && it >= 0f) it else 0f }
            val trussCount = raw[2].safeRoundToInt().coerceAtLeast(0)
            val fruitCount = raw[3].safeRoundToInt().coerceAtLeast(0)
            val expectedSize = 4 + trussCount * 10 + fruitCount * 7
            if (raw.size < expectedSize) {
                return DetectionResult(
                    false,
                    "JNI 返回数据长度异常 (${raw.size} < $expectedSize)",
                    inferMs,
                    emptyList(),
                    emptyList(),
                    -3
                )
            }

            var offset = 4
            val trusses = ArrayList<TrussItem>(trussCount)
            for (i in 0 until trussCount) {
                trusses.add(
                    TrussItem(
                        id = i + 1,
                        x1 = raw[offset],
                        y1 = raw[offset + 1],
                        x2 = raw[offset + 2],
                        y2 = raw[offset + 3],
                        conf = raw[offset + 4],
                        ripeCount = raw[offset + 5].safeRoundToInt(),
                        turningCount = raw[offset + 6].safeRoundToInt(),
                        immatureCount = raw[offset + 7].safeRoundToInt(),
                        gradeCode = raw[offset + 8].safeRoundToInt(),
                        maturity = raw[offset + 9]
                    )
                )
                offset += 10
            }

            val fruits = ArrayList<FruitItem>(fruitCount)
            for (i in 0 until fruitCount) {
                fruits.add(
                    FruitItem(
                        x1 = raw[offset],
                        y1 = raw[offset + 1],
                        x2 = raw[offset + 2],
                        y2 = raw[offset + 3],
                        conf = raw[offset + 4],
                        cls = raw[offset + 5].safeRoundToInt(),
                        parentTrussIndex = raw[offset + 6].safeRoundToInt(-1)
                    )
                )
                offset += 7
            }

            return DetectionResult(true, "", inferMs, trusses, fruits, 0)
        }
    }

    private external fun initModel(assetManager: AssetManager, useGpu: Boolean): String

    private external fun detect(
        bitmap: Bitmap,
        confThresh: Float,
        iouThresh: Float,
        agnosticFruitNms: Boolean,
        assignNearestCenter: Boolean
    ): FloatArray?

    private external fun detectRgba(
        rgbaBuffer: ByteBuffer,
        bufPos: Int,
        bufLimit: Int,
        bufWidth: Int,
        bufHeight: Int,
        rowStride: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        rotationDegrees: Int,
        mirrorHorizontal: Boolean,
        confThresh: Float,
        iouThresh: Float,
        agnosticFruitNms: Boolean,
        assignNearestCenter: Boolean
    ): FloatArray?

    private external fun repostprocess(
        confThresh: Float,
        iouThresh: Float,
        agnosticFruitNms: Boolean,
        assignNearestCenter: Boolean
    ): FloatArray?

    private external fun clearCache()

    fun initialize(assetManager: AssetManager, useGpu: Boolean): InitResult {
        ensureNativeLoaded()?.let { err ->
            return InitResult(
                success = false,
                fallback = false,
                activeBackend = "未加载",
                reason = err
            )
        }
        val raw = initModel(assetManager, useGpu)
        val parts = raw.split("|", limit = 3)
        val status = parts.getOrElse(0) { "ERROR" }
        val backend = parts.getOrElse(1) { "未知" }
        val reason = parts.getOrElse(2) { "" }
        return InitResult(
            success = status == "OK" || status == "FALLBACK",
            fallback = status == "FALLBACK",
            activeBackend = backend,
            reason = reason
        )
    }

    fun detectFrame(bitmap: Bitmap, config: DetectorConfig): DetectionResult {
        ensureNativeLoaded()?.let { err ->
            return DetectionResult(false, err, 0f, emptyList(), emptyList(), ERR_NATIVE_LOAD)
        }
        val raw = detect(
            bitmap,
            config.confThresh,
            config.iouThresh,
            config.agnosticFruitNms,
            config.assignNearestCenter
        )
        return parseDetectionResult(raw)
    }

    fun detectRgbaFrame(
        rgbaBuffer: ByteBuffer,
        bufWidth: Int,
        bufHeight: Int,
        rowStride: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        rotationDegrees: Int,
        mirrorHorizontal: Boolean,
        config: DetectorConfig
    ): DetectionResult {
        ensureNativeLoaded()?.let { err ->
            return DetectionResult(false, err, 0f, emptyList(), emptyList(), ERR_NATIVE_LOAD)
        }
        val raw = detectRgba(
            rgbaBuffer,
            rgbaBuffer.position(),
            rgbaBuffer.limit(),
            bufWidth,
            bufHeight,
            rowStride,
            cropLeft,
            cropTop,
            cropWidth,
            cropHeight,
            rotationDegrees,
            mirrorHorizontal,
            config.confThresh,
            config.iouThresh,
            config.agnosticFruitNms,
            config.assignNearestCenter
        )
        return parseDetectionResult(raw)
    }

    fun repostprocessCached(config: DetectorConfig): DetectionResult {
        ensureNativeLoaded()?.let { err ->
            return DetectionResult(false, err, 0f, emptyList(), emptyList(), ERR_NATIVE_LOAD)
        }
        val raw = repostprocess(
            config.confThresh,
            config.iouThresh,
            config.agnosticFruitNms,
            config.assignNearestCenter
        )
        return parseDetectionResult(raw)
    }

    fun clearNativeCache() {
        if (!nativeLoaded) return
        try {
            clearCache()
        } catch (_: UnsatisfiedLinkError) {
        }
    }
}
