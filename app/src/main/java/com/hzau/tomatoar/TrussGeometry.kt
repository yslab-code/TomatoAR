package com.hzau.tomatoar

import kotlin.math.max
import kotlin.math.min

data class NormBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

object TrussGeometry {
    fun toNormBox(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        srcW: Int,
        srcH: Int
    ): NormBox? {
        if (srcW <= 0 || srcH <= 0) return null
        if (!x1.isFinite() || !y1.isFinite() || !x2.isFinite() || !y2.isFinite()) return null
        val l = (min(x1, x2) / srcW.toFloat()).coerceIn(0f, 1f)
        val t = (min(y1, y2) / srcH.toFloat()).coerceIn(0f, 1f)
        val r = (max(x1, x2) / srcW.toFloat()).coerceIn(0f, 1f)
        val b = (max(y1, y2) / srcH.toFloat()).coerceIn(0f, 1f)
        if (r <= l || b <= t) return null
        return NormBox(l, t, r, b)
    }

    fun normIou(a: NormBox, b: NormBox): Float {
        val ix1 = max(a.left, b.left)
        val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right)
        val iy2 = min(a.bottom, b.bottom)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        val denom = areaA + areaB - inter
        return if (denom > 0f) inter / denom else 0f
    }

    fun matchTrussIndex(
        target: NormBox,
        trusses: List<TrussItem>,
        srcW: Int,
        srcH: Int,
        minIou: Float = 0.25f
    ): Int? {
        if (trusses.isEmpty() || srcW <= 0 || srcH <= 0) return null
        var bestIdx = -1
        var bestIou = 0f
        for (idx in trusses.indices) {
            val t = trusses[idx]
            val candidate = toNormBox(t.x1, t.y1, t.x2, t.y2, srcW, srcH) ?: continue
            val iou = normIou(target, candidate)
            if (iou > bestIou) {
                bestIou = iou
                bestIdx = idx
            }
        }
        return if (bestIdx >= 0 && bestIou >= minIou) bestIdx else null
    }

    fun scaleDetectionCoordinates(
        res: DetectionResult,
        fromW: Int,
        fromH: Int,
        toW: Int,
        toH: Int
    ): DetectionResult {
        if (!res.success || fromW <= 0 || fromH <= 0 || toW <= 0 || toH <= 0 ||
            (fromW == toW && fromH == toH)
        ) {
            return res
        }
        val sx = toW.toFloat() / fromW.toFloat()
        val sy = toH.toFloat() / fromH.toFloat()
        val scaledTrusses = res.trusses.map { t ->
            t.copy(x1 = t.x1 * sx, y1 = t.y1 * sy, x2 = t.x2 * sx, y2 = t.y2 * sy)
        }
        val scaledFruits = res.fruits.map { f ->
            f.copy(x1 = f.x1 * sx, y1 = f.y1 * sy, x2 = f.x2 * sx, y2 = f.y2 * sy)
        }
        return res.copy(trusses = scaledTrusses, fruits = scaledFruits)
    }
}
