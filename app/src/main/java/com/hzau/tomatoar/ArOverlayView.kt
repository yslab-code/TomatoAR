package com.hzau.tomatoar

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class ArOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var currentResult: DetectionResult? = null
    private var srcWidth: Int = 0
    private var srcHeight: Int = 0
    private var config: DetectorConfig = DetectorConfig()
    private var selectedTrussIndex: Int? = null
    private var safeTopInsetPx: Float = 0f

    private val srcToViewMatrix = Matrix()
    private val liveScratch = RenderScratch()
    private val hitBoxes = HitTargetBuffer()
    private val hitBadges = HitTargetBuffer()

    var onTrussSelectedListener: ((Int?) -> Unit)? = null

    fun setSafeTopInsetPx(topPx: Float) {
        if (safeTopInsetPx != topPx) {
            safeTopInsetPx = topPx
            invalidate()
        }
    }

    fun updateScene(
        result: DetectionResult?,
        sourceWidth: Int,
        sourceHeight: Int,
        detectorConfig: DetectorConfig,
        selectedIndex: Int?
    ) {
        currentResult = result
        srcWidth = sourceWidth
        srcHeight = sourceHeight
        config = detectorConfig
        selectedTrussIndex = if (result != null && selectedIndex != null && selectedIndex in result.trusses.indices) {
            selectedIndex
        } else {
            null
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        hitBoxes.reset()
        hitBadges.reset()

        val result = currentResult ?: return
        if (!result.success || srcWidth <= 0 || srcHeight <= 0 || width <= 0 || height <= 0) {
            return
        }

        srcToViewMatrix.reset()
        srcToViewMatrix.setRectToRect(
            RectF(0f, 0f, srcWidth.toFloat(), srcHeight.toFloat()),
            RectF(0f, 0f, width.toFloat(), height.toFloat()),
            Matrix.ScaleToFit.CENTER
        )

        val density = resources.displayMetrics.density
        drawSceneAnnotations(
            canvas = canvas,
            scratch = liveScratch,
            boundsWidth = width.toFloat(),
            boundsHeight = height.toFloat(),
            safeTopPx = safeTopInsetPx,
            dstMatrix = srcToViewMatrix,
            result = result,
            config = config,
            selectedTrussIndex = selectedTrussIndex,
            isExportImage = false,
            scaleFactor = density,
            outBoxes = hitBoxes,
            outBadges = hitBadges
        )
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val result = currentResult
        if (result == null || !result.success || result.trusses.isEmpty()) {
            return false
        }
        if (event.action == MotionEvent.ACTION_UP) {
            val x = event.x
            val y = event.y
            val pad = 12f * resources.displayMetrics.density

            // Check badges first (smaller, easier to miss without priority)
            for (i in 0 until hitBadges.count) {
                val target = hitBadges.items[i]
                val rect = target.rect
                if (x >= rect.left - pad && x <= rect.right + pad &&
                    y >= rect.top - pad && y <= rect.bottom + pad
                ) {
                    val next = if (selectedTrussIndex == target.index) null else target.index
                    selectedTrussIndex = next
                    invalidate()
                    onTrussSelectedListener?.invoke(next)
                    performClick()
                    return true
                }
            }

            // Pick smallest containing truss box if overlapping
            var bestIdx: Int? = null
            var bestArea = Float.MAX_VALUE
            for (i in 0 until hitBoxes.count) {
                val target = hitBoxes.items[i]
                val rect = target.rect
                if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                    val area = rect.width() * rect.height()
                    if (area < bestArea) {
                        bestArea = area
                        bestIdx = target.index
                    }
                }
            }
            if (bestIdx != null) {
                val next = if (selectedTrussIndex == bestIdx) null else bestIdx
                selectedTrussIndex = next
                invalidate()
                onTrussSelectedListener?.invoke(next)
                performClick()
                return true
            }

            if (selectedTrussIndex != null) {
                selectedTrussIndex = null
                invalidate()
                onTrussSelectedListener?.invoke(null)
                performClick()
                return true
            }
        }
        return event.action == MotionEvent.ACTION_DOWN
    }

    private class HitTarget(
        var index: Int = 0,
        val rect: RectF = RectF()
    )

    private class HitTargetBuffer {
        val items = ArrayList<HitTarget>(16)
        var count = 0

        fun reset() {
            count = 0
        }

        fun add(index: Int, rect: RectF) {
            val entry = if (count < items.size) {
                items[count]
            } else {
                HitTarget().also { items.add(it) }
            }
            entry.index = index
            entry.rect.set(rect)
            count += 1
        }
    }

    private class RenderScratch {
        val fruitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
        }
        val fruitLabelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = COLOR_LABEL_BG
        }
        val fruitLabelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            typeface = Typeface.DEFAULT_BOLD
        }
        val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            color = COLOR_TRUSS_SHADOW
        }
        val trussPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
        }
        val badgeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            typeface = Typeface.DEFAULT_BOLD
        }
        val pillTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            typeface = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }
        val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = COLOR_BADGE_BG
        }
        val badgeStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = COLOR_BADGE_STROKE
        }
        val pillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        val mappedRect = RectF()
        val boxRect = RectF()
        val badgeRect = RectF()
        val pillRect = RectF()
        val labelBgRect = RectF()
        val placedBadges = ArrayList<RectF>(16)
        var placedCount = 0

        fun resetPlacedBadges() {
            placedCount = 0
        }

        fun recordPlacedBadge(src: RectF) {
            if (placedCount >= placedBadges.size) {
                placedBadges.add(RectF(src))
            } else {
                placedBadges[placedCount].set(src)
            }
            placedCount += 1
        }
    }

    companion object {
        private val COLOR_RIPE = Color.parseColor("#FF3B30")
        private val COLOR_TURNING = Color.parseColor("#FF9500")
        private val COLOR_IMMATURE = Color.parseColor("#34C759")
        private val COLOR_TRUSS_CORNER = Color.parseColor("#F5FFFFFF")
        private val COLOR_TRUSS_SHADOW = Color.parseColor("#73000000")
        private val COLOR_TRUSS_SELECTED = Color.parseColor("#FFD60A")
        private val COLOR_BADGE_BG = Color.parseColor("#E6141416")
        private val COLOR_BADGE_STROKE = Color.parseColor("#38FFFFFF")
        private val COLOR_LABEL_BG = Color.parseColor("#CC141416")

        fun gradeColor(gradeCode: Int): Int {
            return when (gradeCode) {
                1 -> Color.parseColor("#FF3B30") // A
                2 -> Color.parseColor("#FF6B22") // B
                3 -> Color.parseColor("#FF9500") // C
                4 -> Color.parseColor("#30B0C7") // D
                5 -> Color.parseColor("#34C759") // E
                else -> Color.parseColor("#8E8E93")
            }
        }

        private fun fruitColor(cls: Int): Int {
            return when (cls) {
                1 -> COLOR_RIPE
                2 -> COLOR_TURNING
                3 -> COLOR_IMMATURE
                else -> Color.WHITE
            }
        }

        fun renderExportBitmap(
            srcBitmap: Bitmap,
            result: DetectionResult,
            config: DetectorConfig,
            selectedTrussIndex: Int?
        ): Bitmap {
            val outBitmap = srcBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(outBitmap)
            val w = outBitmap.width.toFloat()
            val h = outBitmap.height.toFloat()
            val exportScale = (min(w, h) / 420f).coerceIn(1.2f, 4.0f)
            val exportScratch = RenderScratch()

            drawSceneAnnotations(
                canvas = canvas,
                scratch = exportScratch,
                boundsWidth = w,
                boundsHeight = h,
                safeTopPx = 8f * exportScale,
                dstMatrix = Matrix(),
                result = result,
                config = config,
                selectedTrussIndex = selectedTrussIndex,
                isExportImage = true,
                scaleFactor = exportScale,
                outBoxes = null,
                outBadges = null
            )

            drawExportSummaryFooter(canvas, w, h, result, exportScale)
            return outBitmap
        }

        private data class ExportLayoutLine(
            val text: String,
            val paint: Paint,
            val textSize: Float,
            val ascent: Float,
            val lineHeight: Float
        )

        private fun drawSceneAnnotations(
            canvas: Canvas,
            scratch: RenderScratch,
            boundsWidth: Float,
            boundsHeight: Float,
            safeTopPx: Float,
            dstMatrix: Matrix,
            result: DetectionResult,
            config: DetectorConfig,
            selectedTrussIndex: Int?,
            isExportImage: Boolean,
            scaleFactor: Float,
            outBoxes: HitTargetBuffer?,
            outBadges: HitTargetBuffer?
        ) {
            scratch.resetPlacedBadges()

            val fruitPaint = scratch.fruitPaint
            fruitPaint.strokeWidth = 1.5f * scaleFactor
            val fruitLabelBgPaint = scratch.fruitLabelBgPaint
            val fruitLabelTextPaint = scratch.fruitLabelTextPaint
            fruitLabelTextPaint.textSize = 10.5f * scaleFactor

            val mappedRect = scratch.mappedRect
            val labelBgRect = scratch.labelBgRect

            // 1. Single fruits (thin hollow borders so real fruit color shows through)
            if (config.showFruitBoxes) {
                val fruitFm = fruitLabelTextPaint.fontMetrics
                val fruitTh = fruitFm.descent - fruitFm.ascent
                val padH = 3f * scaleFactor
                val padV = 1.5f * scaleFactor

                for (f in result.fruits) {
                    mappedRect.set(f.x1, f.y1, f.x2, f.y2)
                    dstMatrix.mapRect(mappedRect)
                    fruitPaint.color = fruitColor(f.cls)
                    canvas.drawRect(mappedRect, fruitPaint)

                    if (config.showFruitLabels) {
                        val label = String.format(Locale.US, "%s %.2f", f.stageName, f.conf)
                        val tw = fruitLabelTextPaint.measureText(label)
                        val lLeft = mappedRect.left.coerceIn(2f, (boundsWidth - tw - 2 * padH - 2f).coerceAtLeast(2f))
                        val lTop = (mappedRect.top - fruitTh - 2 * padV).let {
                            if (it < safeTopPx) mappedRect.top + 2f else it
                        }
                        labelBgRect.set(lLeft, lTop, lLeft + tw + 2 * padH, lTop + fruitTh + 2 * padV)
                        canvas.drawRoundRect(labelBgRect, 3f * scaleFactor, 3f * scaleFactor, fruitLabelBgPaint)
                        canvas.drawText(label, lLeft + padH, lTop + padV - fruitFm.ascent, fruitLabelTextPaint)
                    }
                }
            }

            // 2. Trusses: corner brackets in live preview, full frame when selected or exporting
            val shadowPaint = scratch.shadowPaint
            val trussPaint = scratch.trussPaint

            val textScale = if (config.largeTextForGlasses && !isExportImage) 1.28f else 1.0f
            val badgeTextPaint = scratch.badgeTextPaint
            badgeTextPaint.textSize = 12.5f * scaleFactor * textScale
            val pillTextPaint = scratch.pillTextPaint
            pillTextPaint.textSize = 12.5f * scaleFactor * textScale
            val badgeBgPaint = scratch.badgeBgPaint
            val badgeStrokePaint = scratch.badgeStrokePaint
            badgeStrokePaint.strokeWidth = 1f * scaleFactor
            val pillPaint = scratch.pillPaint

            val boxRect = scratch.boxRect
            val badgeRect = scratch.badgeRect
            val pillRect = scratch.pillRect

            val fm = badgeTextPaint.fontMetrics
            val textH = fm.descent - fm.ascent
            val padX = 7f * scaleFactor * textScale
            val padY = 4f * scaleFactor * textScale
            val gap = 5f * scaleFactor * textScale
            val pillSize = textH + 2f * scaleFactor * textScale
            val badgeH = textH + padY * 2
            val margin = 4f * scaleFactor
            val radius = 7f * scaleFactor * textScale
            val pillRadius = 4f * scaleFactor * textScale
            val pillFm = pillTextPaint.fontMetrics
            val collisionGap = 3f * scaleFactor

            for ((idx, t) in result.trusses.withIndex()) {
                boxRect.set(t.x1, t.y1, t.x2, t.y2)
                dstMatrix.mapRect(boxRect)
                outBoxes?.add(idx, boxRect)

                val isSelected = selectedTrussIndex == idx
                val gColor = gradeColor(t.gradeCode)

                if (isExportImage) {
                    shadowPaint.strokeWidth = 4.5f * scaleFactor
                    canvas.drawRoundRect(boxRect, 6f * scaleFactor, 6f * scaleFactor, shadowPaint)
                    trussPaint.color = gColor
                    trussPaint.strokeWidth = 2.4f * scaleFactor
                    canvas.drawRoundRect(boxRect, 6f * scaleFactor, 6f * scaleFactor, trussPaint)
                } else if (isSelected) {
                    shadowPaint.strokeWidth = 5.2f * scaleFactor
                    canvas.drawRoundRect(boxRect, 6f * scaleFactor, 6f * scaleFactor, shadowPaint)
                    trussPaint.color = COLOR_TRUSS_SELECTED
                    trussPaint.strokeWidth = 2.8f * scaleFactor
                    canvas.drawRoundRect(boxRect, 6f * scaleFactor, 6f * scaleFactor, trussPaint)
                } else {
                    shadowPaint.strokeWidth = 4.6f * scaleFactor
                    val cornerLen = min(20f * scaleFactor, min(boxRect.width(), boxRect.height()) * 0.25f)
                    drawCornerBrackets(canvas, boxRect, cornerLen, shadowPaint)
                    trussPaint.color = COLOR_TRUSS_CORNER
                    trussPaint.strokeWidth = 2.6f * scaleFactor
                    drawCornerBrackets(canvas, boxRect, cornerLen, trussPaint)
                }

                // Short badge: #1 [A] 立即采收 (export includes maturity % too)
                val idText = "#${t.id}"
                val gradeText = t.gradeLetter
                val adviceText = if (isExportImage && t.totalFruits > 0) {
                    "${t.shortAdvice} ${t.maturityPercentText}"
                } else {
                    t.shortAdvice
                }

                val idW = badgeTextPaint.measureText(idText)
                val pillW = maxOf(pillSize, pillTextPaint.measureText(gradeText) + 8f * scaleFactor * textScale)
                val adviceW = badgeTextPaint.measureText(adviceText)
                val badgeW = padX * 2 + idW + gap + pillW + gap + adviceW

                val bLeft = boxRect.left.coerceIn(margin, (boundsWidth - badgeW - margin).coerceAtLeast(margin))
                val minTop = safeTopPx + margin
                val maxTop = (boundsHeight - badgeH - margin).coerceAtLeast(minTop)
                val preferredTop = (boxRect.top - badgeH - margin).let {
                    if (it < minTop) (boxRect.top + margin).coerceIn(minTop, maxTop) else it.coerceIn(minTop, maxTop)
                }

                val bTop = findNonOverlappingBadgeTop(
                    bLeft = bLeft,
                    badgeW = badgeW,
                    badgeH = badgeH,
                    preferredTop = preferredTop,
                    boxRect = boxRect,
                    minTop = minTop,
                    maxTop = maxTop,
                    margin = margin,
                    collisionGap = collisionGap,
                    scratch = scratch
                )

                badgeRect.set(bLeft, bTop, bLeft + badgeW, bTop + badgeH)
                scratch.recordPlacedBadge(badgeRect)
                outBadges?.add(idx, badgeRect)

                canvas.drawRoundRect(badgeRect, radius, radius, badgeBgPaint)
                badgeStrokePaint.color = if (isSelected) COLOR_TRUSS_SELECTED else COLOR_BADGE_STROKE
                canvas.drawRoundRect(badgeRect, radius, radius, badgeStrokePaint)

                val textBaseline = badgeRect.top + padY - fm.ascent
                var curX = badgeRect.left + padX
                canvas.drawText(idText, curX, textBaseline, badgeTextPaint)
                curX += idW + gap

                // Colored grade pill
                pillPaint.color = gColor
                val pillTop = badgeRect.centerY() - pillSize * 0.5f
                pillRect.set(curX, pillTop, curX + pillW, pillTop + pillSize)
                canvas.drawRoundRect(pillRect, pillRadius, pillRadius, pillPaint)
                val pillBaseline = pillRect.centerY() - (pillFm.ascent + pillFm.descent) * 0.5f
                canvas.drawText(gradeText, pillRect.centerX(), pillBaseline, pillTextPaint)
                curX += pillW + gap

                canvas.drawText(adviceText, curX, textBaseline, badgeTextPaint)
            }
        }

        private fun findNonOverlappingBadgeTop(
            bLeft: Float,
            badgeW: Float,
            badgeH: Float,
            preferredTop: Float,
            boxRect: RectF,
            minTop: Float,
            maxTop: Float,
            margin: Float,
            collisionGap: Float,
            scratch: RenderScratch
        ): Float {
            if (scratch.placedCount == 0) return preferredTop

            val bRight = bLeft + badgeW
            val step = badgeH + collisionGap

            // Helper to compute total intersection area with already-placed badges
            fun overlapScore(candTop: Float): Float {
                val candBottom = candTop + badgeH
                var totalOverlap = 0f
                for (i in 0 until scratch.placedCount) {
                    val p = scratch.placedBadges[i]
                    val ox = min(bRight, p.right + collisionGap) - max(bLeft, p.left - collisionGap)
                    if (ox <= 0f) continue
                    val oy = min(candBottom, p.bottom + collisionGap) - max(candTop, p.top - collisionGap)
                    if (oy > 0f) {
                        totalOverlap += ox * oy
                    }
                }
                return totalOverlap
            }

            if (overlapScore(preferredTop) <= 0f) return preferredTop

            val candidates = floatArrayOf(
                preferredTop,
                (boxRect.top + margin).coerceIn(minTop, maxTop),
                (boxRect.bottom - badgeH - margin).coerceIn(minTop, maxTop),
                (boxRect.bottom + margin).coerceIn(minTop, maxTop),
                (preferredTop + step).coerceIn(minTop, maxTop),
                (preferredTop - step).coerceIn(minTop, maxTop),
                (preferredTop + step * 2f).coerceIn(minTop, maxTop),
                (preferredTop - step * 2f).coerceIn(minTop, maxTop),
                (preferredTop + step * 3f).coerceIn(minTop, maxTop),
                (preferredTop - step * 3f).coerceIn(minTop, maxTop)
            )

            var bestTop = preferredTop
            var bestScore = Float.MAX_VALUE
            for (cand in candidates) {
                val score = overlapScore(cand)
                if (score <= 0f) return cand
                if (score < bestScore) {
                    bestScore = score
                    bestTop = cand
                }
            }
            return bestTop
        }

        private fun drawCornerBrackets(canvas: Canvas, r: RectF, len: Float, paint: Paint) {
            // Top-left
            canvas.drawLine(r.left, r.top, r.left + len, r.top, paint)
            canvas.drawLine(r.left, r.top, r.left, r.top + len, paint)
            // Top-right
            canvas.drawLine(r.right - len, r.top, r.right, r.top, paint)
            canvas.drawLine(r.right, r.top, r.right, r.top + len, paint)
            // Bottom-left
            canvas.drawLine(r.left, r.bottom - len, r.left, r.bottom, paint)
            canvas.drawLine(r.left, r.bottom, r.left + len, r.bottom, paint)
            // Bottom-right
            canvas.drawLine(r.right - len, r.bottom, r.right, r.bottom, paint)
            canvas.drawLine(r.right, r.bottom - len, r.right, r.bottom, paint)
        }

        private fun drawExportSummaryFooter(
            canvas: Canvas,
            width: Float,
            height: Float,
            result: DetectionResult,
            scale: Float
        ) {
            val baseTitleSize = 13f * scale
            val baseSubSize = 11.5f * scale

            val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE
                textSize = baseTitleSize
                typeface = Typeface.DEFAULT_BOLD
            }
            val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.parseColor("#E5E5EA")
                textSize = baseSubSize
            }
            val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                color = Color.parseColor("#D9141416")
            }

            val pad = 10f * scale
            val gap = 4f * scale
            val availW = (width - 2f * pad).coerceAtLeast(1f)

            val headerLines = ArrayList<String>(2)
            if (result.trusses.isEmpty()) {
                headerLines.add("DGTruss-YOLO · 暂未检测到番茄果穗 (单果 ${result.fruits.size} 颗)")
            } else {
                val singleHeader =
                    "果穗 ${result.trusses.size} 串 ｜ 检出 ${result.fruits.size} 颗 (已归属 ${result.assignedTotal} · 未归属 ${result.unassignedCount}) ｜ 果穗整体成熟度 ${result.overallAssignedPercentText}"
                if (titlePaint.measureText(singleHeader) <= availW) {
                    headerLines.add(singleHeader)
                } else {
                    headerLines.add("果穗 ${result.trusses.size} 串 ｜ 检出 ${result.fruits.size} 颗 (已归属 ${result.assignedTotal} · 未归属 ${result.unassignedCount})")
                    headerLines.add("果穗整体成熟度 ${result.overallAssignedPercentText}")
                }
            }

            val detailLines = result.trusses.take(4).map { t ->
                "#${t.id} ${t.gradeLetter}级 (${t.maturityPercentText}) [熟${t.ripeCount}/转${t.turningCount}/青${t.immatureCount}] · ${t.shortAdvice}"
            }.toMutableList()
            if (result.trusses.size > 4) {
                detailLines.add("…其余 ${result.trusses.size - 4} 串见设备详情")
            }

            fun fitLine(text: String, paint: Paint, baseSize: Float): ExportLayoutLine {
                paint.textSize = baseSize
                val measured = paint.measureText(text)
                val fittedSize = if (measured > availW) {
                    baseSize * (availW / measured).coerceAtLeast(0.65f)
                } else {
                    baseSize
                }
                paint.textSize = fittedSize
                val fm = paint.fontMetrics
                return ExportLayoutLine(text, paint, fittedSize, fm.ascent, fm.descent - fm.ascent)
            }

            val layoutLines = ArrayList<ExportLayoutLine>(headerLines.size + detailLines.size)
            for (hLine in headerLines) {
                layoutLines.add(fitLine(hLine, titlePaint, baseTitleSize))
            }
            for (dLine in detailLines) {
                layoutLines.add(fitLine(dLine, subPaint, baseSubSize))
            }

            var totalTextH = 0f
            for ((i, line) in layoutLines.withIndex()) {
                totalTextH += line.lineHeight
                if (i + 1 < layoutLines.size) totalTextH += gap
            }
            val totalH = pad * 2 + totalTextH
            val footerTop = (height - totalH).coerceAtLeast(0f)
            canvas.drawRect(0f, footerTop, width, height, bgPaint)

            var curTop = footerTop + pad
            for (line in layoutLines) {
                line.paint.textSize = line.textSize
                canvas.drawText(line.text, pad, curTop - line.ascent, line.paint)
                curTop += line.lineHeight + gap
            }
        }
    }
}
