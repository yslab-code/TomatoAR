package com.hzau.tomatoar

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.text.TextPaint
import android.text.TextUtils
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

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
    private var safeBottomInsetPx: Float = 0f

    private val srcToViewMatrix = Matrix()
    private val srcBoundsRect = RectF()
    private val viewBoundsRect = RectF()
    private var lastMatrixSrcW = -1
    private var lastMatrixSrcH = -1
    private var lastMatrixViewW = -1
    private var lastMatrixViewH = -1

    private val liveScratch = RenderScratch()
    private val hitBoxes = HitTargetBuffer()
    private val hitBadges = HitTargetBuffer()

    private val touchSlopSq: Float = ViewConfiguration.get(context).scaledTouchSlop.let { (it * it).toFloat() }
    private var downX = 0f
    private var downY = 0f
    private var pointerActive = false
    private var touchSlopExceeded = false

    var onTrussSelectedListener: ((Int?) -> Unit)? = null

    fun setSafeTopInsetPx(topPx: Float) {
        setSafeInsetsPx(topPx, safeBottomInsetPx)
    }

    fun setSafeInsetsPx(topPx: Float, bottomPx: Float) {
        val cleanTop = topPx.coerceAtLeast(0f)
        val cleanBottom = bottomPx.coerceAtLeast(0f)
        if (safeTopInsetPx != cleanTop || safeBottomInsetPx != cleanBottom) {
            safeTopInsetPx = cleanTop
            safeBottomInsetPx = cleanBottom
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

        if (lastMatrixSrcW != srcWidth || lastMatrixSrcH != srcHeight ||
            lastMatrixViewW != width || lastMatrixViewH != height
        ) {
            srcBoundsRect.set(0f, 0f, srcWidth.toFloat(), srcHeight.toFloat())
            viewBoundsRect.set(0f, 0f, width.toFloat(), height.toFloat())
            srcToViewMatrix.reset()
            srcToViewMatrix.setRectToRect(srcBoundsRect, viewBoundsRect, Matrix.ScaleToFit.CENTER)
            lastMatrixSrcW = srcWidth
            lastMatrixSrcH = srcHeight
            lastMatrixViewW = width
            lastMatrixViewH = height
        }

        val density = resources.displayMetrics.density
        val fontScale = resources.configuration.fontScale.coerceIn(0.85f, 1.45f)
        drawSceneAnnotations(
            canvas = canvas,
            scratch = liveScratch,
            boundsWidth = width.toFloat(),
            boundsHeight = height.toFloat(),
            safeTopPx = safeTopInsetPx,
            safeBottomPx = safeBottomInsetPx,
            dstMatrix = srcToViewMatrix,
            result = result,
            config = config,
            selectedTrussIndex = selectedTrussIndex,
            isExportImage = false,
            scaleFactor = density,
            textFontScale = fontScale,
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
            pointerActive = false
            return false
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = event.x
                downY = event.y
                touchSlopExceeded = false
                pointerActive = true
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (pointerActive && !touchSlopExceeded) {
                    val dx = event.x - downX
                    val dy = event.y - downY
                    if (dx * dx + dy * dy > touchSlopSq) {
                        touchSlopExceeded = true
                    }
                }
                return pointerActive
            }
            MotionEvent.ACTION_CANCEL -> {
                pointerActive = false
                touchSlopExceeded = false
                return false
            }
            MotionEvent.ACTION_UP -> {
                val validTap = pointerActive && !touchSlopExceeded
                pointerActive = false
                touchSlopExceeded = false
                if (!validTap) {
                    return false
                }

                val x = event.x
                val y = event.y
                val density = resources.displayMetrics.density
                val badgePad = 12f * density

                // Pass 1: exact bounds on badges in reverse draw order (visually topmost wins).
                for (i in (hitBadges.count - 1) downTo 0) {
                    val target = hitBadges.items[i]
                    val rect = target.rect
                    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                        return applyTapSelection(target.index)
                    }
                }

                // Pass 2: expanded hit slop on badges, resolving overlap by closest badge center.
                var bestBadgeIdx: Int? = null
                var bestBadgeDistSq = Float.MAX_VALUE
                for (i in (hitBadges.count - 1) downTo 0) {
                    val target = hitBadges.items[i]
                    val rect = target.rect
                    if (x >= rect.left - badgePad && x <= rect.right + badgePad &&
                        y >= rect.top - badgePad && y <= rect.bottom + badgePad
                    ) {
                        val dx = x - rect.centerX()
                        val dy = y - rect.centerY()
                        val distSq = dx * dx + dy * dy
                        if (distSq < bestBadgeDistSq) {
                            bestBadgeDistSq = distSq
                            bestBadgeIdx = target.index
                        }
                    }
                }
                if (bestBadgeIdx != null) {
                    return applyTapSelection(bestBadgeIdx)
                }

                // Pass 3: exact truss box hit (smallest area wins).
                var bestBoxIdx: Int? = null
                var bestBoxArea = Float.MAX_VALUE
                for (i in 0 until hitBoxes.count) {
                    val target = hitBoxes.items[i]
                    val rect = target.rect
                    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                        val area = rect.width() * rect.height()
                        if (area < bestBoxArea) {
                            bestBoxArea = area
                            bestBoxIdx = target.index
                        }
                    }
                }
                if (bestBoxIdx != null) {
                    return applyTapSelection(bestBoxIdx)
                }

                // Pass 4: small-box hit slop for distant trusses (< 48dp).
                val minTarget = 48f * density
                val smallPad = 8f * density
                for (i in 0 until hitBoxes.count) {
                    val target = hitBoxes.items[i]
                    val rect = target.rect
                    if (rect.width() < minTarget || rect.height() < minTarget) {
                        if (x >= rect.left - smallPad && x <= rect.right + smallPad &&
                            y >= rect.top - smallPad && y <= rect.bottom + smallPad
                        ) {
                            val area = rect.width() * rect.height()
                            if (area < bestBoxArea) {
                                bestBoxArea = area
                                bestBoxIdx = target.index
                            }
                        }
                    }
                }
                if (bestBoxIdx != null) {
                    return applyTapSelection(bestBoxIdx)
                }

                if (selectedTrussIndex != null) {
                    selectedTrussIndex = null
                    invalidate()
                    onTrussSelectedListener?.invoke(null)
                    performClick()
                    return true
                }
                return false
            }
        }
        return super.onTouchEvent(event)
    }

    private fun applyTapSelection(targetIndex: Int): Boolean {
        val next = if (selectedTrussIndex == targetIndex) null else targetIndex
        selectedTrussIndex = next
        invalidate()
        onTrussSelectedListener?.invoke(next)
        performClick()
        return true
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
            strokeJoin = Paint.Join.ROUND
            color = COLOR_TRUSS_SHADOW
        }
        val trussPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
        val leaderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            color = COLOR_LEADER_LINE
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

        val cornerPath = Path()
        val mappedRect = RectF()
        val boxRect = RectF()
        val badgeRect = RectF()
        val pillRect = RectF()
        val labelBgRect = RectF()
        val fruitFm = Paint.FontMetrics()
        val badgeFm = Paint.FontMetrics()
        val pillFm = Paint.FontMetrics()
        val candidates = FloatArray(10)
        val leftOptions = FloatArray(3)
        val labelSb = StringBuilder(24)
        val placedBadges = ArrayList<RectF>(16)
        var placedCount = 0
        var placedLeft = 0f
        var placedTop = 0f

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
        private const val COLOR_RIPE = 0xFFFF3B30.toInt()
        private const val COLOR_TURNING = 0xFFFF9500.toInt()
        private const val COLOR_IMMATURE = 0xFF34C759.toInt()
        private const val COLOR_GRADE_A = 0xFFFF3B30.toInt()
        private const val COLOR_GRADE_B = 0xFFFF6B22.toInt()
        private const val COLOR_GRADE_C = 0xFFFF9500.toInt()
        private const val COLOR_GRADE_D = 0xFF30B0C7.toInt()
        private const val COLOR_GRADE_E = 0xFF34C759.toInt()
        private const val COLOR_GRADE_NONE = 0xFF8E8E93.toInt()
        private const val COLOR_TRUSS_CORNER = 0xF5FFFFFF.toInt()
        private const val COLOR_TRUSS_SHADOW = 0x73000000.toInt()
        private const val COLOR_TRUSS_SELECTED = 0xFFFFD60A.toInt()
        private const val COLOR_BADGE_BG = 0xE6141416.toInt()
        private const val COLOR_BADGE_STROKE = 0x38FFFFFF.toInt()
        private const val COLOR_LABEL_BG = 0xCC141416.toInt()
        private const val COLOR_LEADER_LINE = 0xAAFFFFFF.toInt()
        private const val COLOR_EXPORT_FOOTER_BG = 0xFF141416.toInt()
        private const val COLOR_EXPORT_SUB_TEXT = 0xFFE5E5EA.toInt()
        private const val COLOR_EXPORT_DIVIDER = 0x2EFFFFFF.toInt()

        fun gradeColor(gradeCode: Int): Int {
            return when (gradeCode) {
                1 -> COLOR_GRADE_A
                2 -> COLOR_GRADE_B
                3 -> COLOR_GRADE_C
                4 -> COLOR_GRADE_D
                5 -> COLOR_GRADE_E
                else -> COLOR_GRADE_NONE
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
            config: DetectorConfig
        ): Bitmap? {
            if (srcBitmap.isRecycled || srcBitmap.width <= 0 || srcBitmap.height <= 0) {
                return null
            }
            val srcW = srcBitmap.width
            val srcH = srcBitmap.height
            val w = srcW.toFloat()
            val h = srcH.toFloat()
            val exportScale = (min(w, h) / 420f).coerceIn(0.65f, 4.0f)

            val footerPlan = buildExportFooterPlan(w, result, exportScale)
            val footerHeightPx = ceil(footerPlan.totalHeight).toInt().coerceAtLeast(1)

            val outBitmap = try {
                Bitmap.createBitmap(srcW, srcH + footerHeightPx, Bitmap.Config.ARGB_8888)
            } catch (_: OutOfMemoryError) {
                return null
            } catch (_: IllegalArgumentException) {
                return null
            }

            return try {
                val canvas = Canvas(outBitmap)
                canvas.drawBitmap(srcBitmap, 0f, 0f, null)

                val exportScratch = RenderScratch()
                drawSceneAnnotations(
                    canvas = canvas,
                    scratch = exportScratch,
                    boundsWidth = w,
                    boundsHeight = h,
                    safeTopPx = 8f * exportScale,
                    safeBottomPx = 8f * exportScale,
                    dstMatrix = Matrix(),
                    result = result,
                    config = config,
                    selectedTrussIndex = null,
                    isExportImage = true,
                    scaleFactor = exportScale,
                    textFontScale = 1.0f,
                    outBoxes = null,
                    outBadges = null
                )

                drawExportSummaryFooter(canvas, w, h, footerHeightPx.toFloat(), footerPlan)
                outBitmap
            } catch (_: OutOfMemoryError) {
                outBitmap.recycle()
                null
            } catch (_: Exception) {
                outBitmap.recycle()
                null
            }
        }

        private data class ExportLayoutLine(
            val text: String,
            val paint: TextPaint,
            val textSize: Float,
            val ascent: Float,
            val lineHeight: Float
        )

        private data class ExportFooterPlan(
            val titlePaint: TextPaint,
            val subPaint: TextPaint,
            val bgPaint: Paint,
            val dividerPaint: Paint,
            val pad: Float,
            val gap: Float,
            val lines: List<ExportLayoutLine>,
            val totalHeight: Float
        )

        private fun drawSceneAnnotations(
            canvas: Canvas,
            scratch: RenderScratch,
            boundsWidth: Float,
            boundsHeight: Float,
            safeTopPx: Float,
            safeBottomPx: Float,
            dstMatrix: Matrix,
            result: DetectionResult,
            config: DetectorConfig,
            selectedTrussIndex: Int?,
            isExportImage: Boolean,
            scaleFactor: Float,
            textFontScale: Float,
            outBoxes: HitTargetBuffer?,
            outBadges: HitTargetBuffer?
        ) {
            scratch.resetPlacedBadges()

            val glassesMode = config.largeTextForGlasses && !isExportImage
            val strokeScale = if (glassesMode) 1.25f else 1.0f
            val fruitLabelScale = if (glassesMode) 1.22f else 1.0f
            val textScale = (if (glassesMode) 1.28f else 1.0f) * textFontScale

            val fruitPaint = scratch.fruitPaint
            fruitPaint.strokeWidth = 1.5f * scaleFactor * strokeScale
            val fruitLabelBgPaint = scratch.fruitLabelBgPaint
            val fruitLabelTextPaint = scratch.fruitLabelTextPaint
            fruitLabelTextPaint.textSize = 10.5f * scaleFactor * fruitLabelScale * textFontScale

            val mappedRect = scratch.mappedRect
            val labelBgRect = scratch.labelBgRect

            // 1. Single fruits (thin hollow borders so real fruit color shows through)
            if (config.showFruitBoxes) {
                fruitLabelTextPaint.getFontMetrics(scratch.fruitFm)
                val fruitFm = scratch.fruitFm
                val fruitTh = fruitFm.descent - fruitFm.ascent
                val padH = 3f * scaleFactor
                val padV = 1.5f * scaleFactor
                val labelH = fruitTh + 2f * padV
                val labelRadius = 3f * scaleFactor
                val maxLabelTop = (boundsHeight - safeBottomPx - labelH - 2f).coerceAtLeast(safeTopPx)
                val sb = scratch.labelSb

                for (i in result.fruits.indices) {
                    val f = result.fruits[i]
                    mappedRect.set(f.x1, f.y1, f.x2, f.y2)
                    dstMatrix.mapRect(mappedRect)
                    fruitPaint.color = fruitColor(f.cls)
                    canvas.drawRect(mappedRect, fruitPaint)

                    if (config.showFruitLabels) {
                        sb.setLength(0)
                        sb.append(f.stageName).append(' ')
                        val pct = (f.conf * 100f).roundToInt().coerceIn(0, 100)
                        if (pct >= 100) {
                            sb.append("1.00")
                        } else {
                            sb.append("0.")
                            if (pct < 10) sb.append('0')
                            sb.append(pct)
                        }
                        val tw = fruitLabelTextPaint.measureText(sb, 0, sb.length)
                        val lLeft = mappedRect.left.coerceIn(
                            2f,
                            (boundsWidth - tw - 2f * padH - 2f).coerceAtLeast(2f)
                        )
                        val lTop = (mappedRect.top - labelH).let { above ->
                            if (above < safeTopPx) (mappedRect.top + 2f).coerceIn(safeTopPx, maxLabelTop)
                            else above.coerceIn(safeTopPx, maxLabelTop)
                        }
                        labelBgRect.set(lLeft, lTop, lLeft + tw + 2f * padH, lTop + labelH)
                        canvas.drawRoundRect(labelBgRect, labelRadius, labelRadius, fruitLabelBgPaint)
                        canvas.drawText(sb, 0, sb.length, lLeft + padH, lTop + padV - fruitFm.ascent, fruitLabelTextPaint)
                    }
                }
            }

            // 2. Trusses: corner brackets in live preview, full frame when selected or exporting
            val shadowPaint = scratch.shadowPaint
            val trussPaint = scratch.trussPaint
            val leaderPaint = scratch.leaderPaint
            leaderPaint.strokeWidth = 1.5f * scaleFactor

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

            badgeTextPaint.getFontMetrics(scratch.badgeFm)
            pillTextPaint.getFontMetrics(scratch.pillFm)
            val fm = scratch.badgeFm
            val pillFm = scratch.pillFm

            val textH = fm.descent - fm.ascent
            val padX = 7f * scaleFactor * textScale
            val padY = 4f * scaleFactor * textScale
            val gap = 5f * scaleFactor * textScale
            val pillSize = textH + 2f * scaleFactor * textScale
            val badgeH = textH + padY * 2f
            val margin = 4f * scaleFactor
            val radius = 7f * scaleFactor * textScale
            val pillRadius = 4f * scaleFactor * textScale
            val collisionGap = 3f * scaleFactor
            val minTop = safeTopPx + margin
            val maxTop = (boundsHeight - safeBottomPx - badgeH - margin).coerceAtLeast(minTop)
            val cornerRadius = 6f * scaleFactor
            val pillExtraW = 8f * scaleFactor * textScale

            for (idx in result.trusses.indices) {
                val t = result.trusses[idx]
                boxRect.set(t.x1, t.y1, t.x2, t.y2)
                dstMatrix.mapRect(boxRect)
                outBoxes?.add(idx, boxRect)

                val isSelected = selectedTrussIndex == idx
                val gColor = gradeColor(t.gradeCode)

                if (isExportImage) {
                    shadowPaint.strokeWidth = 4.5f * scaleFactor
                    canvas.drawRoundRect(boxRect, cornerRadius, cornerRadius, shadowPaint)
                    trussPaint.color = gColor
                    trussPaint.strokeWidth = 2.4f * scaleFactor
                    canvas.drawRoundRect(boxRect, cornerRadius, cornerRadius, trussPaint)
                } else if (isSelected) {
                    shadowPaint.strokeWidth = 5.2f * scaleFactor * strokeScale
                    canvas.drawRoundRect(boxRect, cornerRadius, cornerRadius, shadowPaint)
                    trussPaint.color = COLOR_TRUSS_SELECTED
                    trussPaint.strokeWidth = 2.8f * scaleFactor * strokeScale
                    canvas.drawRoundRect(boxRect, cornerRadius, cornerRadius, trussPaint)
                } else {
                    val cornerLen = min(20f * scaleFactor, min(boxRect.width(), boxRect.height()) * 0.25f)
                    buildCornerBracketsPath(scratch.cornerPath, boxRect, cornerLen)
                    shadowPaint.strokeWidth = 4.6f * scaleFactor * strokeScale
                    canvas.drawPath(scratch.cornerPath, shadowPaint)
                    trussPaint.color = COLOR_TRUSS_CORNER
                    trussPaint.strokeWidth = 2.6f * scaleFactor * strokeScale
                    canvas.drawPath(scratch.cornerPath, trussPaint)
                }

                val idText = t.idText
                val gradeText = t.gradeLetter
                val adviceText = if (isExportImage && t.totalFruits > 0) {
                    "${t.shortAdvice} ${t.maturityPercentText}"
                } else {
                    t.shortAdvice
                }

                val idW = badgeTextPaint.measureText(idText)
                val pillW = maxOf(pillSize, pillTextPaint.measureText(gradeText) + pillExtraW)
                val adviceW = badgeTextPaint.measureText(adviceText)
                val badgeW = padX * 2f + idW + gap + pillW + gap + adviceW

                val maxLeft = (boundsWidth - badgeW - margin).coerceAtLeast(margin)
                val preferredLeft = boxRect.left.coerceIn(margin, maxLeft)
                val preferredTop = (boxRect.top - badgeH - margin).let { above ->
                    if (above < minTop) (boxRect.top + margin).coerceIn(minTop, maxTop)
                    else above.coerceIn(minTop, maxTop)
                }

                findNonOverlappingBadgePlacement(
                    preferredLeft = preferredLeft,
                    maxLeft = maxLeft,
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

                val bLeft = scratch.placedLeft
                val bTop = scratch.placedTop
                badgeRect.set(bLeft, bTop, bLeft + badgeW, bTop + badgeH)
                scratch.recordPlacedBadge(badgeRect)
                outBadges?.add(idx, badgeRect)

                // Draw subtle leader line if collision avoidance displaced the badge noticeably away from its box
                val vertGap = when {
                    badgeRect.bottom < boxRect.top -> boxRect.top - badgeRect.bottom
                    badgeRect.top > boxRect.bottom -> badgeRect.top - boxRect.bottom
                    else -> 0f
                }
                if (vertGap > badgeH * 0.5f) {
                    val startX = badgeRect.centerX().coerceIn(badgeRect.left + radius, badgeRect.right - radius)
                    val startY = if (badgeRect.centerY() < boxRect.centerY()) badgeRect.bottom else badgeRect.top
                    val endX = startX.coerceIn(boxRect.left, boxRect.right)
                    val endY = if (badgeRect.centerY() < boxRect.centerY()) boxRect.top else boxRect.bottom
                    canvas.drawLine(startX, startY, endX, endY, leaderPaint)
                }

                canvas.drawRoundRect(badgeRect, radius, radius, badgeBgPaint)
                badgeStrokePaint.color = if (isSelected) COLOR_TRUSS_SELECTED else COLOR_BADGE_STROKE
                canvas.drawRoundRect(badgeRect, radius, radius, badgeStrokePaint)

                val textBaseline = badgeRect.top + padY - fm.ascent
                var curX = badgeRect.left + padX
                canvas.drawText(idText, curX, textBaseline, badgeTextPaint)
                curX += idW + gap

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

        private fun overlapScore(
            left: Float,
            top: Float,
            badgeW: Float,
            badgeH: Float,
            collisionGap: Float,
            scratch: RenderScratch
        ): Float {
            val right = left + badgeW
            val bottom = top + badgeH
            var totalOverlap = 0f
            for (i in 0 until scratch.placedCount) {
                val p = scratch.placedBadges[i]
                val ox = min(right, p.right + collisionGap) - max(left, p.left - collisionGap)
                if (ox <= 0f) continue
                val oy = min(bottom, p.bottom + collisionGap) - max(top, p.top - collisionGap)
                if (oy > 0f) {
                    totalOverlap += ox * oy
                }
            }
            return totalOverlap
        }

        private fun findNonOverlappingBadgePlacement(
            preferredLeft: Float,
            maxLeft: Float,
            badgeW: Float,
            badgeH: Float,
            preferredTop: Float,
            boxRect: RectF,
            minTop: Float,
            maxTop: Float,
            margin: Float,
            collisionGap: Float,
            scratch: RenderScratch
        ) {
            if (scratch.placedCount == 0 ||
                overlapScore(preferredLeft, preferredTop, badgeW, badgeH, collisionGap, scratch) <= 0f
            ) {
                scratch.placedLeft = preferredLeft
                scratch.placedTop = preferredTop
                return
            }

            val step = badgeH + collisionGap
            val cands = scratch.candidates
            cands[0] = preferredTop
            cands[1] = (boxRect.top + margin).coerceIn(minTop, maxTop)
            cands[2] = (boxRect.bottom - badgeH - margin).coerceIn(minTop, maxTop)
            cands[3] = (boxRect.bottom + margin).coerceIn(minTop, maxTop)
            cands[4] = (preferredTop + step).coerceIn(minTop, maxTop)
            cands[5] = (preferredTop - step).coerceIn(minTop, maxTop)
            cands[6] = (preferredTop + step * 2f).coerceIn(minTop, maxTop)
            cands[7] = (preferredTop - step * 2f).coerceIn(minTop, maxTop)
            cands[8] = (preferredTop + step * 3f).coerceIn(minTop, maxTop)
            cands[9] = (preferredTop - step * 3f).coerceIn(minTop, maxTop)

            val leftOpts = scratch.leftOptions
            leftOpts[0] = preferredLeft
            leftOpts[1] = (boxRect.right - badgeW).coerceIn(margin, maxLeft)
            leftOpts[2] = (boxRect.centerX() - badgeW * 0.5f).coerceIn(margin, maxLeft)

            var bestLeft = preferredLeft
            var bestTop = preferredTop
            var bestScore = Float.MAX_VALUE

            for (li in 0 until 3) {
                val candLeft = leftOpts[li]
                var dupLeft = false
                for (prevL in 0 until li) {
                    if (kotlin.math.abs(leftOpts[prevL] - candLeft) < 1f) {
                        dupLeft = true
                        break
                    }
                }
                if (dupLeft) continue

                val limitCand = if (li == 0) cands.size else 4
                for (ci in 0 until limitCand) {
                    val candTop = cands[ci]
                    var dupTop = false
                    for (prevT in 0 until ci) {
                        if (kotlin.math.abs(cands[prevT] - candTop) < 0.5f) {
                            dupTop = true
                            break
                        }
                    }
                    if (dupTop) continue

                    val score = overlapScore(candLeft, candTop, badgeW, badgeH, collisionGap, scratch)
                    if (score <= 0f) {
                        scratch.placedLeft = candLeft
                        scratch.placedTop = candTop
                        return
                    }
                    if (score < bestScore) {
                        bestScore = score
                        bestLeft = candLeft
                        bestTop = candTop
                    }
                }
            }

            scratch.placedLeft = bestLeft
            scratch.placedTop = bestTop
        }

        private fun buildCornerBracketsPath(path: Path, r: RectF, len: Float) {
            path.reset()
            // Top-left
            path.moveTo(r.left, r.top + len)
            path.lineTo(r.left, r.top)
            path.lineTo(r.left + len, r.top)
            // Top-right
            path.moveTo(r.right - len, r.top)
            path.lineTo(r.right, r.top)
            path.lineTo(r.right, r.top + len)
            // Bottom-left
            path.moveTo(r.left, r.bottom - len)
            path.lineTo(r.left, r.bottom)
            path.lineTo(r.left + len, r.bottom)
            // Bottom-right
            path.moveTo(r.right - len, r.bottom)
            path.lineTo(r.right, r.bottom)
            path.lineTo(r.right, r.bottom - len)
        }

        private fun buildExportFooterPlan(
            width: Float,
            result: DetectionResult,
            scale: Float
        ): ExportFooterPlan {
            val baseTitleSize = 13f * scale
            val baseSubSize = 11.5f * scale

            val titlePaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE
                textSize = baseTitleSize
                typeface = Typeface.DEFAULT_BOLD
            }
            val subPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
                color = COLOR_EXPORT_SUB_TEXT
                textSize = baseSubSize
            }
            val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                color = COLOR_EXPORT_FOOTER_BG
            }
            val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                color = COLOR_EXPORT_DIVIDER
            }

            val pad = 10f * scale
            val gap = 4f * scale
            val availW = (width - 2f * pad).coerceAtLeast(16f)
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())

            val headerLines = ArrayList<String>(3)
            if (result.trusses.isEmpty()) {
                headerLines.add("DGTruss-YOLO · $timestamp · 未检出完整果穗 (单果 ${result.fruits.size} 颗)")
            } else {
                val combined =
                    "DGTruss-YOLO · $timestamp ｜ 果穗 ${result.trusses.size} 串 · 检出 ${result.fruits.size} 颗 (归属 ${result.assignedTotal}/未归属 ${result.unassignedCount}) ｜ 整体成熟度 ${result.overallAssignedPercentText}"
                if (titlePaint.measureText(combined) <= availW) {
                    headerLines.add(combined)
                } else {
                    headerLines.add("DGTruss-YOLO · $timestamp ｜ 果穗 ${result.trusses.size} 串 · 整体成熟度 ${result.overallAssignedPercentText}")
                    headerLines.add("单果 ${result.fruits.size} 颗 (归属 ${result.assignedTotal} · 未归属 ${result.unassignedCount})")
                }
            }

            val detailLines = ArrayList<String>(6)
            detailLines.add("图例：红框=成熟果  橙框=转色果  绿框=未熟果")
            for (t in result.trusses.take(4)) {
                detailLines.add(
                    "#${t.id} ${t.gradeLetter}级 (${t.maturityPercentText}) [熟${t.ripeCount}/转${t.turningCount}/青${t.immatureCount}] · ${t.shortAdvice}"
                )
            }
            if (result.trusses.size > 4) {
                detailLines.add("…其余 ${result.trusses.size - 4} 串详见应用内清单")
            }

            fun fitLine(text: String, paint: TextPaint, baseSize: Float): ExportLayoutLine {
                paint.textSize = baseSize
                val measured = paint.measureText(text)
                val fittedSize = if (measured > availW) {
                    baseSize * (availW / measured).coerceAtLeast(0.70f)
                } else {
                    baseSize
                }
                paint.textSize = fittedSize
                val finalText = if (paint.measureText(text) > availW) {
                    TextUtils.ellipsize(text, paint, availW, TextUtils.TruncateAt.END).toString()
                } else {
                    text
                }
                val fm = paint.fontMetrics
                return ExportLayoutLine(finalText, paint, fittedSize, fm.ascent, fm.descent - fm.ascent)
            }

            val layoutLines = ArrayList<ExportLayoutLine>(headerLines.size + detailLines.size)
            for (hLine in headerLines) {
                layoutLines.add(fitLine(hLine, titlePaint, baseTitleSize))
            }
            for (dLine in detailLines) {
                layoutLines.add(fitLine(dLine, subPaint, baseSubSize))
            }

            var totalTextH = 0f
            for (i in layoutLines.indices) {
                totalTextH += layoutLines[i].lineHeight
                if (i + 1 < layoutLines.size) totalTextH += gap
            }
            val totalH = pad * 2f + totalTextH
            return ExportFooterPlan(titlePaint, subPaint, bgPaint, dividerPaint, pad, gap, layoutLines, totalH)
        }

        private fun drawExportSummaryFooter(
            canvas: Canvas,
            width: Float,
            photoHeight: Float,
            footerHeight: Float,
            plan: ExportFooterPlan
        ) {
            canvas.drawRect(0f, photoHeight, width, photoHeight + footerHeight, plan.bgPaint)
            canvas.drawRect(0f, photoHeight, width, photoHeight + 1f, plan.dividerPaint)

            var curTop = photoHeight + plan.pad
            for (i in plan.lines.indices) {
                val line = plan.lines[i]
                line.paint.textSize = line.textSize
                canvas.drawText(line.text, plan.pad, curTop - line.ascent, line.paint)
                curTop += line.lineHeight + plan.gap
            }
        }
    }
}
