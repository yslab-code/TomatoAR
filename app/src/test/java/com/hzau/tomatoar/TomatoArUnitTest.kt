package com.hzau.tomatoar

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TomatoArUnitTest {

    @Test
    fun parseDetectionResult_validPacket_parsesTrussesFruitsAndMaturity() {
        val raw = floatArrayOf(
            1f, 14.5f, 1f, 3f,
            // Truss 0: (10,20)-(110,220), conf=0.92, ripe=2, turning=1, immature=0, grade=1(A), maturity=2.5/3
            10f, 20f, 110f, 220f, 0.92f, 2f, 1f, 0f, 1f, 2.5f / 3f,
            // Fruit 0..2
            20f, 30f, 40f, 50f, 0.95f, 1f, 0f,
            50f, 60f, 70f, 80f, 0.88f, 1f, 0f,
            30f, 90f, 50f, 110f, 0.77f, 2f, 0f
        )

        val res = YoloDetector.parseDetectionResult(raw)
        assertTrue(res.success)
        assertEquals(14.5f, res.inferMs, 1e-4f)
        assertEquals(1, res.trusses.size)
        assertEquals(3, res.fruits.size)

        val t = res.trusses[0]
        assertEquals("#1", t.idText)
        assertEquals("A", t.gradeLetter)
        assertEquals("83.3%", t.maturityPercentText)
        assertEquals(3, t.totalFruits)
        assertEquals(3, res.assignedTotal)
        assertEquals(0, res.unassignedCount)
        assertEquals("83.3%", res.overallAssignedPercentText)
    }

    @Test
    fun parseDetectionResult_handlesErrorPacketAndNanSafety() {
        val errRaw = floatArrayOf(-1f, YoloDetector.ERR_RGBA_BUFFER.toFloat(), -2f, 0f, 100f, 200f)
        val errRes = YoloDetector.parseDetectionResult(errRaw)
        assertFalse(errRes.success)
        assertEquals(YoloDetector.ERR_RGBA_BUFFER, errRes.errorCode)

        val nanHeader = YoloDetector.parseDetectionResult(floatArrayOf(Float.NaN, 1f, 0f, 0f))
        assertFalse(nanHeader.success)
        assertEquals(-2, nanHeader.errorCode)

        val truncated = YoloDetector.parseDetectionResult(floatArrayOf(1f, 5f, 1f, 0f, 1f, 2f))
        assertFalse(truncated.success)
        assertEquals(-3, truncated.errorCode)
    }

    @Test
    fun trussGeometry_preservesSelectionAcrossLiveToFreezeResolutionScaling() {
        val liveW = 360
        val liveH = 640
        val freezeW = 1080
        val freezeH = 1920

        val liveTruss = TrussItem(
            id = 1,
            x1 = 60f,
            y1 = 120f,
            x2 = 180f,
            y2 = 360f,
            conf = 0.90f,
            ripeCount = 3,
            turningCount = 1,
            immatureCount = 1,
            gradeCode = 2,
            maturity = 0.70f
        )
        val liveResult = DetectionResult(
            success = true,
            errorMessage = "",
            inferMs = 12f,
            trusses = listOf(liveTruss),
            fruits = emptyList()
        )

        val selectedNorm = TrussGeometry.toNormBox(
            liveTruss.x1, liveTruss.y1, liveTruss.x2, liveTruss.y2,
            liveW, liveH
        )
        assertNotNull(selectedNorm)

        val scaled = TrussGeometry.scaleDetectionCoordinates(liveResult, liveW, liveH, freezeW, freezeH)
        assertEquals(180f, scaled.trusses[0].x1, 1e-3f)
        assertEquals(360f, scaled.trusses[0].y1, 1e-3f)
        assertEquals(540f, scaled.trusses[0].x2, 1e-3f)
        assertEquals(1080f, scaled.trusses[0].y2, 1e-3f)

        val matched = TrussGeometry.matchTrussIndex(selectedNorm!!, scaled.trusses, freezeW, freezeH)
        assertEquals(0, matched)
    }

    @Test
    fun trussGeometry_rejectsDisjointBoxes() {
        val normA = TrussGeometry.toNormBox(10f, 10f, 50f, 50f, 100, 100)!!
        val farTruss = TrussItem(
            id = 1, x1 = 60f, y1 = 60f, x2 = 90f, y2 = 90f,
            conf = 0.8f, ripeCount = 0, turningCount = 0, immatureCount = 0,
            gradeCode = 0, maturity = -1f
        )
        assertNull(TrussGeometry.matchTrussIndex(normA, listOf(farTruss), 100, 100))
    }
}
