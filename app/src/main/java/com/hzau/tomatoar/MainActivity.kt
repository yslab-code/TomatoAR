package com.hzau.tomatoar

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Rational
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.doOnLayout
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private enum class AppMode { LIVE, FROZEN, ALBUM }

    private data class LiveFrameState(
        val sessionId: Long,
        val srcWidth: Int,
        val srcHeight: Int,
        val result: DetectionResult
    )

    private data class FrameSnapshot(
        val sessionId: Long,
        val imageToken: Long,
        val bitmap: Bitmap,
        val result: DetectionResult
    )

    private val detector = YoloDetector()
    private val analyzerExecutor = Executors.newSingleThreadExecutor()
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val ioExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private val activeSessionId = AtomicLong(1L)
    private val cameraBindEpoch = AtomicLong(0L)
    private val imageTokenGenerator = AtomicLong(1L)
    private val cachedBitmapToken = AtomicLong(0L)
    private val freezeRequested = AtomicBoolean(false)
    private val reDetectSeq = AtomicLong(0L)
    private val initSeq = AtomicLong(0L)
    private val modelInitInProgress = AtomicBoolean(false)

    @Volatile private var mode: AppMode = AppMode.LIVE
    @Volatile private var config: DetectorConfig = DetectorConfig()
    @Volatile private var modelReady = false
    @Volatile private var activeBackendLabel = "初始化中…"
    @Volatile private var backendFallbackNotice = ""
    @Volatile private var lastFrameError = ""
    @Volatile private var latestLiveState: LiveFrameState? = null
    @Volatile private var lockedSnapshot: FrameSnapshot? = null
    @Volatile private var measuredFps = 0f

    // FPS tracking owned exclusively by analyzerExecutor
    private var fpsSessionId = -1L
    private var fpsWindowStartNs = 0L
    private var fpsFrameCount = 0

    // Reusable non-direct fallback scratch (owned exclusively by analyzerExecutor)
    private var fallbackTightBuffer: ByteBuffer? = null
    private var fallbackRowScratch: ByteArray? = null

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var cameraProvider: ProcessCameraProvider? = null
    private var currentImageAnalysis: ImageAnalysis? = null

    private var selectedTrussIndex: Int? = null
    private var selectedTrussNormBox: NormBox? = null
    private var techCardExpanded = false
    private var summaryExpanded = false
    private var settingsVisible = false
    private var pendingSaveAfterPermission = false
    private var summaryScrollMaxHeightPx = 0
    private var isHdRefining = false
    private var isExporting = false
    private var isSummaryScrolling = false
    private var lastLiveDrawerUpdateMs = 0L
    private var pendingReDetectRunnable: Runnable? = null
    private var selectableRowBgResId = 0

    // Views
    private lateinit var rootContainer: FrameLayout
    private lateinit var previewView: PreviewView
    private lateinit var frozenImageView: ImageView
    private lateinit var arOverlayView: ArOverlayView
    private lateinit var topHudContainer: LinearLayout
    private lateinit var bottomStack: LinearLayout

    private lateinit var btnStatusCapsule: LinearLayout
    private lateinit var statusDot: View
    private lateinit var tvStatusLabel: TextView
    private lateinit var tvStatusSub: TextView
    private lateinit var btnSettings: ImageButton

    private lateinit var techDetailCard: LinearLayout
    private lateinit var tvTechBackend: TextView
    private lateinit var tvTechResolution: TextView
    private lateinit var tvTechPerf: TextView
    private lateinit var tvTechNotice: TextView

    private lateinit var selectedTrussCard: LinearLayout
    private lateinit var tvSelectedGradeBadge: TextView
    private lateinit var tvSelectedTitle: TextView
    private lateinit var tvSelectedCounts: TextView
    private lateinit var btnCloseSelected: ImageButton

    private lateinit var summaryExpandedCard: LinearLayout
    private lateinit var summaryScrollView: ScrollView
    private lateinit var tvExpandedOverall: TextView
    private lateinit var trussListContainer: LinearLayout

    private lateinit var btnSummaryPill: LinearLayout
    private lateinit var tvSummaryPill: TextView
    private lateinit var ivSummaryChevron: ImageView

    private lateinit var liveControlRow: View
    private lateinit var btnGallery: ImageButton
    private lateinit var btnShutter: FrameLayout
    private lateinit var btnFlipCamera: ImageButton

    private lateinit var frozenControlRow: View
    private lateinit var btnResumeLive: Button
    private lateinit var btnSaveResult: Button

    private lateinit var settingsBackdrop: FrameLayout
    private lateinit var settingsSheet: LinearLayout
    private lateinit var settingsScrollView: ScrollView
    private lateinit var btnCloseSettings: ImageButton
    private lateinit var switchShowFruitBoxes: SwitchCompat
    private lateinit var switchShowFruitLabels: SwitchCompat
    private lateinit var switchLargeText: SwitchCompat
    private lateinit var tvConfValue: TextView
    private lateinit var seekConf: SeekBar
    private lateinit var tvIouValue: TextView
    private lateinit var seekIou: SeekBar
    private lateinit var btnFruitNmsAgnostic: Button
    private lateinit var btnFruitNmsPerClass: Button
    private lateinit var btnAssignNearest: Button
    private lateinit var btnAssignFirst: Button
    private lateinit var btnBackendCpu: Button
    private lateinit var btnBackendGpu: Button
    private lateinit var btnRes480: Button
    private lateinit var btnRes720: Button
    private lateinit var btnRes1080: Button
    private lateinit var btnLoadSample: Button

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                if (mode == AppMode.LIVE) {
                    startCamera()
                }
            } else {
                if (lockedSnapshot == null) {
                    Toast.makeText(this, "未授予相机权限，已切换为样本图/相册测图模式", Toast.LENGTH_LONG).show()
                    loadBuiltInSample()
                } else {
                    Toast.makeText(this, "未授予相机权限，保持当前图像模式", Toast.LENGTH_SHORT).show()
                }
            }
        }

    private val storagePermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted && pendingSaveAfterPermission) {
                pendingSaveAfterPermission = false
                exportCurrentSnapshot()
            } else {
                pendingSaveAfterPermission = false
                Toast.makeText(this, "未授予存储权限，无法保存到公共相册", Toast.LENGTH_SHORT).show()
            }
        }

    private val photoPickerLauncher =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
            if (uri != null) {
                loadAlbumImage(uri)
            }
        }

    private val legacyDocumentPickerLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) {
                loadAlbumImage(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = false
        }
        setContentView(R.layout.activity_main)

        val tv = TypedValue()
        if (theme.resolveAttribute(android.R.attr.selectableItemBackground, tv, true)) {
            selectableRowBgResId = tv.resourceId
        }

        config = loadPersistedConfig()
        bindViews()
        applyWindowInsets()
        setupListeners()
        setupBackNavigation()
        syncSettingsUi()
        updateModeUi()

        initializeModelAsync(useGpu = config.useGpu, reRunFrozenAfterInit = false)

        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onStop() {
        super.onStop()
        freezeRequested.set(false)
        cancelPendingReDetect()
    }

    override fun onDestroy() {
        super.onDestroy()
        cancelPendingReDetect()
        freezeRequested.set(false)
        activeSessionId.incrementAndGet()
        unbindCameraSafely()
        analyzerExecutor.shutdownNow()
        inferenceExecutor.shutdownNow()
        safeExecute(ioExecutor) {
            runCatching { detector.clearNativeCache() }
        }
        ioExecutor.shutdown()
    }

    private fun safeExecute(executor: ExecutorService, block: () -> Unit): Boolean {
        if (executor.isShutdown || executor.isTerminated) {
            return false
        }
        return try {
            executor.execute(block)
            true
        } catch (_: RejectedExecutionException) {
            false
        }
    }

    private fun unbindCameraSafely() {
        cameraBindEpoch.incrementAndGet()
        try {
            currentImageAnalysis?.clearAnalyzer()
        } catch (_: Exception) {
        }
        currentImageAnalysis = null
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
    }

    private fun cancelPendingReDetect() {
        pendingReDetectRunnable?.let { mainHandler.removeCallbacks(it) }
        pendingReDetectRunnable = null
    }

    private fun scheduleDebouncedReDetect() {
        cancelPendingReDetect()
        val runnable = Runnable {
            pendingReDetectRunnable = null
            if (!isDestroyed && !isFinishing && mode != AppMode.LIVE) {
                reDetectLockedSnapshot(forceFullInfer = false)
            }
        }
        pendingReDetectRunnable = runnable
        mainHandler.postDelayed(runnable, 40L)
    }

    private fun loadPersistedConfig(): DetectorConfig {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val rawConf = prefs.getFloat(KEY_CONF_THRESH, 0.35f)
        val rawIou = prefs.getFloat(KEY_IOU_THRESH, 0.70f)
        return DetectorConfig(
            confThresh = (if (rawConf.isFinite()) rawConf else 0.35f).coerceIn(0.10f, 0.85f),
            iouThresh = (if (rawIou.isFinite()) rawIou else 0.70f).coerceIn(0.30f, 0.90f),
            agnosticFruitNms = prefs.getBoolean(KEY_AGNOSTIC_NMS, true),
            assignNearestCenter = prefs.getBoolean(KEY_ASSIGN_NEAREST, true),
            showFruitBoxes = prefs.getBoolean(KEY_SHOW_FRUIT_BOXES, true),
            showFruitLabels = prefs.getBoolean(KEY_SHOW_FRUIT_LABELS, false),
            largeTextForGlasses = prefs.getBoolean(KEY_LARGE_TEXT, false),
            useGpu = prefs.getBoolean(KEY_USE_GPU, false),
            captureResIndex = prefs.getInt(KEY_CAPTURE_RES, 1).coerceIn(0, 2)
        )
    }

    private fun saveConfig(cfg: DetectorConfig) {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
            .putFloat(KEY_CONF_THRESH, cfg.confThresh)
            .putFloat(KEY_IOU_THRESH, cfg.iouThresh)
            .putBoolean(KEY_AGNOSTIC_NMS, cfg.agnosticFruitNms)
            .putBoolean(KEY_ASSIGN_NEAREST, cfg.assignNearestCenter)
            .putBoolean(KEY_SHOW_FRUIT_BOXES, cfg.showFruitBoxes)
            .putBoolean(KEY_SHOW_FRUIT_LABELS, cfg.showFruitLabels)
            .putBoolean(KEY_LARGE_TEXT, cfg.largeTextForGlasses)
            .putBoolean(KEY_USE_GPU, cfg.useGpu)
            .putInt(KEY_CAPTURE_RES, cfg.captureResIndex)
            .apply()
    }

    private inline fun updateConfig(persist: Boolean = true, transform: (DetectorConfig) -> DetectorConfig) {
        val next = transform(config)
        if (next != config) {
            config = next
            if (persist) {
                saveConfig(next)
            }
        }
    }

    private fun bindViews() {
        rootContainer = findViewById(R.id.rootContainer)
        previewView = findViewById(R.id.previewView)
        frozenImageView = findViewById(R.id.frozenImageView)
        arOverlayView = findViewById(R.id.arOverlayView)
        topHudContainer = findViewById(R.id.topHudContainer)
        bottomStack = findViewById(R.id.bottomStack)

        btnStatusCapsule = findViewById(R.id.btnStatusCapsule)
        statusDot = findViewById(R.id.statusDot)
        tvStatusLabel = findViewById(R.id.tvStatusLabel)
        tvStatusSub = findViewById(R.id.tvStatusSub)
        btnSettings = findViewById(R.id.btnSettings)

        techDetailCard = findViewById(R.id.techDetailCard)
        tvTechBackend = findViewById(R.id.tvTechBackend)
        tvTechResolution = findViewById(R.id.tvTechResolution)
        tvTechPerf = findViewById(R.id.tvTechPerf)
        tvTechNotice = findViewById(R.id.tvTechNotice)

        selectedTrussCard = findViewById(R.id.selectedTrussCard)
        tvSelectedGradeBadge = findViewById(R.id.tvSelectedGradeBadge)
        tvSelectedTitle = findViewById(R.id.tvSelectedTitle)
        tvSelectedCounts = findViewById(R.id.tvSelectedCounts)
        btnCloseSelected = findViewById(R.id.btnCloseSelected)

        summaryExpandedCard = findViewById(R.id.summaryExpandedCard)
        summaryScrollView = findViewById(R.id.summaryScrollView)
        tvExpandedOverall = findViewById(R.id.tvExpandedOverall)
        trussListContainer = findViewById(R.id.trussListContainer)

        btnSummaryPill = findViewById(R.id.btnSummaryPill)
        tvSummaryPill = findViewById(R.id.tvSummaryPill)
        ivSummaryChevron = findViewById(R.id.ivSummaryChevron)

        liveControlRow = findViewById(R.id.liveControlRow)
        btnGallery = findViewById(R.id.btnGallery)
        btnShutter = findViewById(R.id.btnShutter)
        btnFlipCamera = findViewById(R.id.btnFlipCamera)

        frozenControlRow = findViewById(R.id.frozenControlRow)
        btnResumeLive = findViewById(R.id.btnResumeLive)
        btnSaveResult = findViewById(R.id.btnSaveResult)

        settingsBackdrop = findViewById(R.id.settingsBackdrop)
        settingsSheet = findViewById(R.id.settingsSheet)
        settingsScrollView = findViewById(R.id.settingsScrollView)
        btnCloseSettings = findViewById(R.id.btnCloseSettings)
        switchShowFruitBoxes = findViewById(R.id.switchShowFruitBoxes)
        switchShowFruitLabels = findViewById(R.id.switchShowFruitLabels)
        switchLargeText = findViewById(R.id.switchLargeText)
        tvConfValue = findViewById(R.id.tvConfValue)
        seekConf = findViewById(R.id.seekConf)
        tvIouValue = findViewById(R.id.tvIouValue)
        seekIou = findViewById(R.id.seekIou)
        btnFruitNmsAgnostic = findViewById(R.id.btnFruitNmsAgnostic)
        btnFruitNmsPerClass = findViewById(R.id.btnFruitNmsPerClass)
        btnAssignNearest = findViewById(R.id.btnAssignNearest)
        btnAssignFirst = findViewById(R.id.btnAssignFirst)
        btnBackendCpu = findViewById(R.id.btnBackendCpu)
        btnBackendGpu = findViewById(R.id.btnBackendGpu)
        btnRes480 = findViewById(R.id.btnRes480)
        btnRes720 = findViewById(R.id.btnRes720)
        btnRes1080 = findViewById(R.id.btnRes1080)
        btnLoadSample = findViewById(R.id.btnLoadSample)
    }

    private fun applyWindowInsets() {
        val dm = resources.displayMetrics
        val dp = dm.density
        var lastBarsTop = 0
        var lastBarsBottom = 0

        fun updateOverlayAndSheetHeights() {
            val containerH = rootContainer.height.takeIf { it > 0 } ?: dm.heightPixels
            val usableH = (containerH - lastBarsTop - lastBarsBottom).coerceAtLeast((240f * dp).roundToInt())
            summaryScrollMaxHeightPx = min((220f * dp).roundToInt(), (usableH * 0.30f).roundToInt())
            val settingsScrollH = min((420f * dp).roundToInt(), (usableH * 0.56f).roundToInt())
            if (settingsScrollView.layoutParams.height != settingsScrollH) {
                settingsScrollView.layoutParams = settingsScrollView.layoutParams.apply {
                    height = settingsScrollH
                }
            }
            val topSafe = max(lastBarsTop.toFloat() + 66f * dp, topHudContainer.bottom.toFloat())
            val bottomSafe = if (rootContainer.height > 0 && bottomStack.top > 0) {
                (rootContainer.height - bottomStack.top).coerceAtLeast(lastBarsBottom).toFloat()
            } else {
                lastBarsBottom.toFloat() + 180f * dp
            }
            arOverlayView.setSafeInsetsPx(topSafe, bottomSafe)
        }

        ViewCompat.setOnApplyWindowInsetsListener(rootContainer) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            lastBarsTop = bars.top
            lastBarsBottom = bars.bottom
            val topPad = bars.top + (10f * dp).roundToInt()
            val sidePad = (14f * dp).roundToInt()
            val bottomPad = bars.bottom + (16f * dp).roundToInt()
            topHudContainer.setPadding(sidePad, topPad, sidePad, 0)
            bottomStack.setPadding(sidePad, 0, sidePad, bottomPad)

            val sheetSidePad = (18f * dp).roundToInt()
            settingsSheet.setPadding(
                sheetSidePad,
                (14f * dp).roundToInt(),
                sheetSidePad,
                bars.bottom + (22f * dp).roundToInt()
            )
            updateOverlayAndSheetHeights()
            insets
        }

        val layoutListener = View.OnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateOverlayAndSheetHeights()
        }
        rootContainer.addOnLayoutChangeListener(layoutListener)
        topHudContainer.addOnLayoutChangeListener(layoutListener)
        bottomStack.addOnLayoutChangeListener(layoutListener)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupListeners() {
        btnStatusCapsule.setOnClickListener {
            techCardExpanded = !techCardExpanded
            techDetailCard.visibility = if (techCardExpanded) View.VISIBLE else View.GONE
            if (techCardExpanded) {
                refreshTechHudFromActiveState()
            }
        }

        btnSettings.setOnClickListener {
            setSettingsVisible(true)
        }
        settingsBackdrop.setOnClickListener {
            setSettingsVisible(false)
        }
        btnCloseSettings.setOnClickListener {
            setSettingsVisible(false)
        }

        arOverlayView.onTrussSelectedListener = { idx ->
            val size = activeSourceSize()
            if (size != null) {
                setSelectedTruss(idx, activeResult(), size.first, size.second)
            } else {
                clearSelectedTruss()
            }
            if (idx != null && summaryExpanded) {
                summaryExpanded = false
                updateSummaryExpandedUi()
            }
            renderCurrentSnapshotDetails()
        }

        btnCloseSelected.setOnClickListener {
            clearSelectedTruss()
            renderCurrentSnapshotDetails()
            redrawActiveOverlay()
        }

        summaryScrollView.setOnTouchListener { _, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> isSummaryScrolling = true
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> isSummaryScrolling = false
            }
            false
        }

        var pillTouchStartY = 0f
        btnSummaryPill.setOnTouchListener { v, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> pillTouchStartY = ev.rawY
                MotionEvent.ACTION_CANCEL -> {
                    v.isPressed = false
                }
                MotionEvent.ACTION_UP -> {
                    val dy = ev.rawY - pillTouchStartY
                    val swipeThreshold = 24f * resources.displayMetrics.density
                    if (dy < -swipeThreshold && !summaryExpanded) {
                        v.isPressed = false
                        summaryExpanded = true
                        updateSummaryExpandedUi()
                        return@setOnTouchListener true
                    }
                    if (dy > swipeThreshold && summaryExpanded) {
                        v.isPressed = false
                        summaryExpanded = false
                        updateSummaryExpandedUi()
                        return@setOnTouchListener true
                    }
                }
            }
            false
        }
        btnSummaryPill.setOnClickListener {
            summaryExpanded = !summaryExpanded
            updateSummaryExpandedUi()
        }

        btnGallery.setOnClickListener { openGalleryPicker() }

        btnShutter.setOnClickListener {
            if (mode != AppMode.LIVE) return@setOnClickListener
            if (!modelReady) {
                Toast.makeText(this, "模型初始化中，请稍候…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val liveState = latestLiveState
            if (liveState == null || liveState.sessionId != activeSessionId.get()) {
                Toast.makeText(this, "等待首帧检测完成…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val rawPreviewBmp = previewView.bitmap
            if (rawPreviewBmp != null) {
                capturePreviewSnapshot(rawPreviewBmp, liveState)
            } else {
                freezeRequested.set(true)
            }
        }

        btnFlipCamera.setOnClickListener {
            if (!hasCameraPermission()) {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                return@setOnClickListener
            }
            val nextFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            val provider = cameraProvider
            if (provider != null) {
                val nextSelector = CameraSelector.Builder().requireLensFacing(nextFacing).build()
                val supported = try {
                    provider.hasCamera(nextSelector)
                } catch (_: Exception) {
                    false
                }
                if (!supported) {
                    Toast.makeText(this, "当前设备不支持切换该摄像头", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
            }
            lensFacing = nextFacing
            resetSessionAndPerf()
            startCamera()
        }

        btnResumeLive.setOnClickListener {
            returnToLiveCamera()
        }

        btnSaveResult.setOnClickListener {
            exportCurrentSnapshot()
        }

        // Display switches
        switchShowFruitBoxes.setOnCheckedChangeListener { _, isChecked ->
            if (config.showFruitBoxes != isChecked) {
                updateConfig { it.copy(showFruitBoxes = isChecked) }
                updateFruitLabelToggleState(isChecked)
                redrawActiveOverlay()
            }
        }
        switchShowFruitLabels.setOnCheckedChangeListener { _, isChecked ->
            if (config.showFruitLabels != isChecked) {
                updateConfig { it.copy(showFruitLabels = isChecked) }
                redrawActiveOverlay()
            }
        }
        switchLargeText.setOnCheckedChangeListener { _, isChecked ->
            if (config.largeTextForGlasses != isChecked) {
                updateConfig { it.copy(largeTextForGlasses = isChecked) }
                redrawActiveOverlay()
            }
        }

        // Detection SeekBars (persist only on release; debounce re-postprocess while dragging)
        seekConf.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val v = 0.10f + progress / 100f
                tvConfValue.text = String.format(Locale.US, "%.2f", v)
                if (fromUser) {
                    updateConfig(persist = false) { it.copy(confThresh = v) }
                    if (mode != AppMode.LIVE) {
                        scheduleDebouncedReDetect()
                    }
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                saveConfig(config)
                if (mode != AppMode.LIVE && pendingReDetectRunnable != null) {
                    cancelPendingReDetect()
                    reDetectLockedSnapshot(forceFullInfer = false)
                }
            }
        })

        seekIou.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val v = 0.30f + progress / 100f
                tvIouValue.text = String.format(Locale.US, "%.2f", v)
                if (fromUser) {
                    updateConfig(persist = false) { it.copy(iouThresh = v) }
                    if (mode != AppMode.LIVE) {
                        scheduleDebouncedReDetect()
                    }
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                saveConfig(config)
                if (mode != AppMode.LIVE && pendingReDetectRunnable != null) {
                    cancelPendingReDetect()
                    reDetectLockedSnapshot(forceFullInfer = false)
                }
            }
        })

        btnFruitNmsAgnostic.setOnClickListener {
            if (!config.agnosticFruitNms) {
                updateConfig { it.copy(agnosticFruitNms = true) }
                syncSettingsUi()
                if (mode != AppMode.LIVE) reDetectLockedSnapshot(forceFullInfer = false)
            }
        }
        btnFruitNmsPerClass.setOnClickListener {
            if (config.agnosticFruitNms) {
                updateConfig { it.copy(agnosticFruitNms = false) }
                syncSettingsUi()
                if (mode != AppMode.LIVE) reDetectLockedSnapshot(forceFullInfer = false)
            }
        }

        btnAssignNearest.setOnClickListener {
            if (!config.assignNearestCenter) {
                updateConfig { it.copy(assignNearestCenter = true) }
                syncSettingsUi()
                if (mode != AppMode.LIVE) reDetectLockedSnapshot(forceFullInfer = false)
            }
        }
        btnAssignFirst.setOnClickListener {
            if (config.assignNearestCenter) {
                updateConfig { it.copy(assignNearestCenter = false) }
                syncSettingsUi()
                if (mode != AppMode.LIVE) reDetectLockedSnapshot(forceFullInfer = false)
            }
        }

        // Device backend & resolution
        btnBackendCpu.setOnClickListener {
            if (!config.useGpu) return@setOnClickListener
            updateConfig { it.copy(useGpu = false) }
            syncSettingsUi()
            resetSessionAndPerf()
            initializeModelAsync(useGpu = false, reRunFrozenAfterInit = (mode != AppMode.LIVE))
        }
        btnBackendGpu.setOnClickListener {
            if (config.useGpu) return@setOnClickListener
            updateConfig { it.copy(useGpu = true) }
            syncSettingsUi()
            resetSessionAndPerf()
            initializeModelAsync(useGpu = true, reRunFrozenAfterInit = (mode != AppMode.LIVE))
        }

        btnRes480.setOnClickListener { changeResolution(0) }
        btnRes720.setOnClickListener { changeResolution(1) }
        btnRes1080.setOnClickListener { changeResolution(2) }

        btnLoadSample.setOnClickListener {
            setSettingsVisible(false)
            loadBuiltInSample()
        }
    }

    private fun setupBackNavigation() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    settingsVisible -> setSettingsVisible(false)
                    summaryExpanded -> {
                        summaryExpanded = false
                        updateSummaryExpandedUi()
                    }
                    selectedTrussIndex != null -> {
                        clearSelectedTruss()
                        renderCurrentSnapshotDetails()
                        redrawActiveOverlay()
                    }
                    mode != AppMode.LIVE && hasCameraPermission() -> returnToLiveCamera()
                    else -> {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        })
    }

    private fun setSettingsVisible(show: Boolean) {
        settingsVisible = show
        settingsBackdrop.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun updateSummaryExpandedUi() {
        summaryExpandedCard.visibility = if (summaryExpanded) View.VISIBLE else View.GONE
        ivSummaryChevron.setImageResource(if (summaryExpanded) R.drawable.ic_chevron_down else R.drawable.ic_chevron_up)
        if (summaryExpanded) {
            populateSummaryDrawer(activeResult(), force = true)
        }
    }

    private fun updateFruitLabelToggleState(fruitBoxesEnabled: Boolean) {
        switchShowFruitLabels.isEnabled = fruitBoxesEnabled
        switchShowFruitLabels.alpha = if (fruitBoxesEnabled) 1.0f else 0.45f
    }

    private fun syncSettingsUi() {
        val cfg = config
        switchShowFruitBoxes.isChecked = cfg.showFruitBoxes
        switchShowFruitLabels.isChecked = cfg.showFruitLabels
        updateFruitLabelToggleState(cfg.showFruitBoxes)
        switchLargeText.isChecked = cfg.largeTextForGlasses

        tvConfValue.text = String.format(Locale.US, "%.2f", cfg.confThresh)
        seekConf.progress = ((cfg.confThresh - 0.10f) * 100f).roundToInt().coerceIn(0, 75)

        tvIouValue.text = String.format(Locale.US, "%.2f", cfg.iouThresh)
        seekIou.progress = ((cfg.iouThresh - 0.30f) * 100f).roundToInt().coerceIn(0, 60)

        styleSegmentPair(btnFruitNmsAgnostic, btnFruitNmsPerClass, cfg.agnosticFruitNms)
        styleSegmentPair(btnAssignNearest, btnAssignFirst, cfg.assignNearestCenter)
        styleSegmentPair(btnBackendCpu, btnBackendGpu, !cfg.useGpu)

        styleSegmentButton(btnRes480, cfg.captureResIndex == 0)
        styleSegmentButton(btnRes720, cfg.captureResIndex == 1)
        styleSegmentButton(btnRes1080, cfg.captureResIndex == 2)
    }

    private fun styleSegmentPair(first: Button, second: Button, firstActive: Boolean) {
        styleSegmentButton(first, firstActive)
        styleSegmentButton(second, !firstActive)
    }

    private fun styleSegmentButton(btn: Button, active: Boolean) {
        btn.isSelected = active
        btn.setBackgroundResource(if (active) R.drawable.bg_pill_segment_active else R.drawable.bg_pill_segment_inactive)
        btn.setTextColor(ContextCompat.getColor(this, if (active) R.color.black else R.color.white))
        btn.setTypeface(null, if (active) Typeface.BOLD else Typeface.NORMAL)
    }

    private fun changeResolution(index: Int) {
        if (config.captureResIndex == index) return
        updateConfig { it.copy(captureResIndex = index) }
        syncSettingsUi()
        if (mode == AppMode.LIVE && hasCameraPermission()) {
            resetSessionAndPerf()
            startCamera()
        }
    }

    private fun resetSessionAndPerf() {
        activeSessionId.incrementAndGet()
        freezeRequested.set(false)
        measuredFps = 0f
        if (mode == AppMode.LIVE) {
            latestLiveState = null
            selectedTrussIndex = null
            selectedTrussNormBox = null
            arOverlayView.updateScene(null, 0, 0, config, null)
        }
    }

    private fun initModelOnInferenceThread(useGpu: Boolean, expectedSeq: Long? = null): InitResult {
        cachedBitmapToken.set(0L)
        val initRes = detector.initialize(assets, useGpu)
        if (expectedSeq == null || expectedSeq == initSeq.get()) {
            modelReady = initRes.success
            activeBackendLabel = initRes.activeBackend
            backendFallbackNotice = if (initRes.fallback) {
                "GPU 回退：${initRes.reason}"
            } else if (!initRes.success) {
                initRes.reason
            } else {
                ""
            }
        }
        return initRes
    }

    private fun initializeModelAsync(useGpu: Boolean, reRunFrozenAfterInit: Boolean) {
        modelReady = false
        modelInitInProgress.set(true)
        val seq = initSeq.incrementAndGet()
        activeBackendLabel = if (useGpu) "正在切换 Vulkan GPU…" else "正在初始化 CPU…"
        backendFallbackNotice = ""
        lastFrameError = ""
        updateStatusCapsuleUi()
        renderCurrentSnapshotDetails()
        if (techCardExpanded) {
            refreshTechHudFromActiveState()
        }

        val targetSession = activeSessionId.get()
        val submitted = safeExecute(inferenceExecutor) {
            if (seq != initSeq.get()) {
                return@safeExecute
            }
            val initRes = try {
                initModelOnInferenceThread(useGpu, expectedSeq = seq)
            } finally {
                if (seq == initSeq.get()) {
                    modelInitInProgress.set(false)
                }
            }
            runOnUiThread {
                if (isDestroyed || isFinishing || seq != initSeq.get()) return@runOnUiThread
                applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                if (reRunFrozenAfterInit && initRes.success && targetSession == activeSessionId.get() && mode != AppMode.LIVE) {
                    reDetectLockedSnapshot(forceFullInfer = true)
                }
            }
        }
        if (!submitted && seq == initSeq.get()) {
            modelInitInProgress.set(false)
        }
    }

    private fun applyInitUiFeedback(initRes: InitResult, showSuccessFallbackToast: Boolean) {
        if (initRes.fallback) {
            updateConfig { it.copy(useGpu = false) }
            syncSettingsUi()
            if (showSuccessFallbackToast) {
                Toast.makeText(this, "Vulkan 不可用，已自动回退 CPU (${initRes.reason})", Toast.LENGTH_LONG).show()
            }
        } else if (!initRes.success) {
            Toast.makeText(this, "模型加载失败：${initRes.reason}", Toast.LENGTH_LONG).show()
        }
        updateStatusCapsuleUi()
        renderCurrentSnapshotDetails()
        if (techCardExpanded) {
            refreshTechHudFromActiveState()
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun targetResolutionSize(): Size {
        return when (config.captureResIndex) {
            0 -> Size(640, 480)
            2 -> Size(1920, 1080)
            else -> Size(1280, 720)
        }
    }

    private fun analysisTargetSize(): Size {
        return if (config.captureResIndex == 0) {
            Size(640, 480)
        } else {
            Size(640, 360)
        }
    }

    private fun capturePreviewSnapshot(rawPreviewBmp: Bitmap, liveState: LiveFrameState) {
        val cropped = cropBitmapToAspect(rawPreviewBmp, liveState.srcWidth, liveState.srcHeight)
        val argbBmp = if (cropped.config != Bitmap.Config.ARGB_8888) {
            try {
                cropped.copy(Bitmap.Config.ARGB_8888, false)?.also { cropped.recycle() } ?: cropped
            } catch (_: OutOfMemoryError) {
                cropped
            }
        } else {
            cropped
        }

        val targetSession = activeSessionId.incrementAndGet()
        freezeRequested.set(false)
        val token = imageTokenGenerator.incrementAndGet()
        val scaledResult = TrussGeometry.scaleDetectionCoordinates(
            liveState.result,
            liveState.srcWidth,
            liveState.srcHeight,
            argbBmp.width,
            argbBmp.height
        )
        val initialSnap = FrameSnapshot(targetSession, token, argbBmp, scaledResult)
        lockedSnapshot = initialSnap
        mode = AppMode.FROZEN
        unbindCameraSafely()
        isHdRefining = true
        updateModeUi()
        renderSnapshot(initialSnap)
        updateSaveButtonState()

        val submitted = safeExecute(inferenceExecutor) {
            if (!modelReady || targetSession != activeSessionId.get()) {
                runOnUiThread {
                    if (!isDestroyed && !isFinishing) {
                        isHdRefining = false
                        updateSaveButtonState()
                    }
                }
                return@safeExecute
            }
            val cfgCopy = config
            val hdResult = detector.detectFrame(argbBmp, cfgCopy)
            if (hdResult.success) {
                cachedBitmapToken.set(token)
            }
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                isHdRefining = false
                if (targetSession == activeSessionId.get() && mode == AppMode.FROZEN) {
                    if (hdResult.success) {
                        val hdSnap = FrameSnapshot(targetSession, token, argbBmp, hdResult)
                        lockedSnapshot = hdSnap
                        renderSnapshot(hdSnap)
                    } else {
                        Toast.makeText(this, "高精度复检失败，已保留实时定格结果", Toast.LENGTH_SHORT).show()
                    }
                }
                updateSaveButtonState()
            }
        }
        if (!submitted) {
            isHdRefining = false
            updateSaveButtonState()
        }
    }

    private fun cropBitmapToAspect(src: Bitmap, targetW: Int, targetH: Int): Bitmap {
        if (targetW <= 0 || targetH <= 0 || src.width <= 0 || src.height <= 0) {
            return src
        }
        val targetRatio = targetW.toFloat() / targetH.toFloat()
        val srcRatio = src.width.toFloat() / src.height.toFloat()
        if (kotlin.math.abs(targetRatio - srcRatio) < 0.01f) {
            return src
        }
        val cropW: Int
        val cropH: Int
        if (targetRatio > srcRatio) {
            cropW = src.width
            cropH = (cropW / targetRatio).roundToInt().coerceIn(1, src.height)
        } else {
            cropH = src.height
            cropW = (cropH * targetRatio).roundToInt().coerceIn(1, src.width)
        }
        val left = ((src.width - cropW) / 2).coerceIn(0, (src.width - cropW).coerceAtLeast(0))
        val top = ((src.height - cropH) / 2).coerceIn(0, (src.height - cropH).coerceAtLeast(0))
        return try {
            val out = Bitmap.createBitmap(src, left, top, cropW, cropH)
            if (out !== src) {
                src.recycle()
            }
            out
        } catch (_: OutOfMemoryError) {
            src
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            if (isDestroyed || isFinishing) return@addListener
            try {
                cameraProvider = providerFuture.get()
                if (mode == AppMode.LIVE) {
                    bindCameraUseCases()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "相机启动失败：${e.message}", Toast.LENGTH_LONG).show()
                loadBuiltInSample()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        previewView.doOnLayout {
            if (isDestroyed || isFinishing || mode != AppMode.LIVE) return@doOnLayout
            if (previewView.width <= 0 || previewView.height <= 0) {
                previewView.post {
                    if (!isDestroyed && !isFinishing && mode == AppMode.LIVE) {
                        bindCameraUseCases()
                    }
                }
                return@doOnLayout
            }

            val previewTargetSize = targetResolutionSize()
            val analysisTarget = analysisTargetSize()
            val aspectStrategy = if (config.captureResIndex == 0) {
                AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            } else {
                AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
            }
            val previewResolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(aspectStrategy)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        previewTargetSize,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
            val analysisResolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(aspectStrategy)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        analysisTarget,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
            val preview = Preview.Builder()
                .setResolutionSelector(previewResolutionSelector)
                .setTargetRotation(rotation)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(analysisResolutionSelector)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            try {
                unbindCameraSafely()
                val bindEpoch = cameraBindEpoch.incrementAndGet()
                val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT

                imageAnalysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                    try {
                        if (mode != AppMode.LIVE || !modelReady || modelInitInProgress.get() ||
                            bindEpoch != cameraBindEpoch.get()
                        ) {
                            return@setAnalyzer
                        }
                        val frameSession = activeSessionId.get()
                        val cfgCopy = config
                        val shouldFreeze = freezeRequested.compareAndSet(true, false)

                        if (shouldFreeze) {
                            val upright = imageProxyToUprightBitmap(imageProxy, isFront)
                            if (upright == null) {
                                runOnUiThread {
                                    if (!isDestroyed && !isFinishing) {
                                        Toast.makeText(this, "定格采图失败，请重试", Toast.LENGTH_SHORT).show()
                                    }
                                }
                                return@setAnalyzer
                            }
                            if (frameSession != activeSessionId.get() || bindEpoch != cameraBindEpoch.get() || mode != AppMode.LIVE) {
                                upright.recycle()
                                return@setAnalyzer
                            }
                            val token = imageTokenGenerator.incrementAndGet()
                            val result = detector.detectFrame(upright, cfgCopy)
                            if (result.success) {
                                cachedBitmapToken.set(token)
                            }
                            if (frameSession != activeSessionId.get() || bindEpoch != cameraBindEpoch.get() || mode != AppMode.LIVE) {
                                upright.recycle()
                                return@setAnalyzer
                            }
                            val snap = FrameSnapshot(frameSession, token, upright, result)
                            runOnUiThread {
                                if (isDestroyed || isFinishing) {
                                    upright.recycle()
                                    return@runOnUiThread
                                }
                                if (mode == AppMode.LIVE && snap.sessionId == activeSessionId.get()) {
                                    lockedSnapshot = snap
                                    mode = AppMode.FROZEN
                                    unbindCameraSafely()
                                    updateModeUi()
                                    renderSnapshot(snap)
                                } else {
                                    upright.recycle()
                                }
                            }
                            return@setAnalyzer
                        }

                        val plane = imageProxy.planes.firstOrNull()
                        val canDirectRgba = plane != null && plane.pixelStride == 4 && plane.buffer.isDirect
                        val result: DetectionResult
                        val uprightW: Int
                        val uprightH: Int

                        if (canDirectRgba) {
                            val p = plane!!
                            val buf = p.buffer
                            buf.rewind()
                            val crop = imageProxy.cropRect
                            val imgW = imageProxy.width
                            val imgH = imageProxy.height
                            if (imgW <= 0 || imgH <= 0) return@setAnalyzer
                            val cropLeft = crop.left.coerceIn(0, imgW - 1)
                            val cropTop = crop.top.coerceIn(0, imgH - 1)
                            val cropW = crop.width().coerceIn(1, imgW - cropLeft)
                            val cropH = crop.height().coerceIn(1, imgH - cropTop)
                            val rot = imageProxy.imageInfo.rotationDegrees

                            cachedBitmapToken.set(0L)
                            result = detector.detectRgbaFrame(
                                rgbaBuffer = buf,
                                bufWidth = imgW,
                                bufHeight = imgH,
                                rowStride = p.rowStride,
                                cropLeft = cropLeft,
                                cropTop = cropTop,
                                cropWidth = cropW,
                                cropHeight = cropH,
                                rotationDegrees = rot,
                                mirrorHorizontal = isFront,
                                config = cfgCopy
                            )
                            val swapWh = (rot == 90 || rot == 270)
                            uprightW = if (swapWh) cropH else cropW
                            uprightH = if (swapWh) cropW else cropH
                        } else {
                            val upright = imageProxyToUprightBitmap(imageProxy, isFront) ?: return@setAnalyzer
                            cachedBitmapToken.set(0L)
                            result = detector.detectFrame(upright, cfgCopy)
                            uprightW = upright.width
                            uprightH = upright.height
                            upright.recycle()
                        }

                        if (frameSession != activeSessionId.get() || bindEpoch != cameraBindEpoch.get() || mode != AppMode.LIVE) {
                            return@setAnalyzer
                        }

                        updateFpsOnAnalyzerThread(frameSession)
                        val state = LiveFrameState(frameSession, uprightW, uprightH, result)
                        runOnUiThread {
                            if (!isDestroyed && !isFinishing && mode == AppMode.LIVE && state.sessionId == activeSessionId.get()) {
                                latestLiveState = state
                                renderLiveState(state)
                            }
                        }
                    } finally {
                        imageProxy.close()
                    }
                }

                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                val viewPort = previewView.viewPort ?: ViewPort.Builder(
                    Rational(previewView.width.coerceAtLeast(1), previewView.height.coerceAtLeast(1)),
                    rotation
                ).setScaleType(ViewPort.FIT).build()

                val group = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalysis)
                    .setViewPort(viewPort)
                    .build()

                currentImageAnalysis = imageAnalysis
                provider.bindToLifecycle(this, selector, group)
            } catch (e: Exception) {
                cameraBindEpoch.incrementAndGet()
                currentImageAnalysis = null
                Toast.makeText(this, "摄像头绑定失败：${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateFpsOnAnalyzerThread(sessionId: Long) {
        val nowNs = System.nanoTime()
        if (fpsSessionId != sessionId || fpsWindowStartNs == 0L) {
            fpsSessionId = sessionId
            fpsWindowStartNs = nowNs
            fpsFrameCount = 0
            return
        }
        fpsFrameCount += 1
        val elapsedNs = nowNs - fpsWindowStartNs
        if (elapsedNs >= 500_000_000L) {
            measuredFps = fpsFrameCount * 1_000_000_000f / elapsedNs.toFloat()
            fpsWindowStartNs = nowNs
            fpsFrameCount = 0
        }
    }

    private fun imageProxyToUprightBitmap(imageProxy: ImageProxy, mirrorHorizontal: Boolean): Bitmap? {
        val plane = imageProxy.planes.firstOrNull() ?: return null
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val w = imageProxy.width
        val h = imageProxy.height
        if (pixelStride != 4 || w <= 0 || h <= 0) return null

        val tightRowBytes = w * 4
        if (rowStride < tightRowBytes) return null
        val buffer = plane.buffer
        buffer.rewind()

        return try {
            val rawBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            if (rowStride == tightRowBytes && buffer.remaining() >= tightRowBytes * h) {
                rawBmp.copyPixelsFromBuffer(buffer)
            } else {
                val neededBytes = tightRowBytes * h
                val tightBuffer: ByteBuffer = fallbackTightBuffer
                    ?.takeIf { it.capacity() >= neededBytes }
                    ?: ByteBuffer.allocateDirect(neededBytes).also { fallbackTightBuffer = it }
                tightBuffer.clear()

                val rowTmp: ByteArray = fallbackRowScratch
                    ?.takeIf { it.size >= tightRowBytes }
                    ?: ByteArray(tightRowBytes).also { fallbackRowScratch = it }

                for (y in 0 until h) {
                    val rowOffset = y * rowStride
                    if (rowOffset + tightRowBytes > buffer.limit()) {
                        rawBmp.recycle()
                        return null
                    }
                    buffer.position(rowOffset)
                    buffer.get(rowTmp, 0, tightRowBytes)
                    tightBuffer.put(rowTmp, 0, tightRowBytes)
                }
                tightBuffer.rewind()
                rawBmp.copyPixelsFromBuffer(tightBuffer)
            }

            val crop = imageProxy.cropRect
            val cropLeft = crop.left.coerceIn(0, w - 1)
            val cropTop = crop.top.coerceIn(0, h - 1)
            val cropW = crop.width().coerceIn(1, w - cropLeft)
            val cropH = crop.height().coerceIn(1, h - cropTop)

            val matrix = Matrix()
            val rot = imageProxy.imageInfo.rotationDegrees
            if (rot != 0) {
                matrix.postRotate(rot.toFloat())
            }
            if (mirrorHorizontal) {
                matrix.postScale(-1f, 1f)
            }

            val upright = Bitmap.createBitmap(rawBmp, cropLeft, cropTop, cropW, cropH, matrix, true)
            if (upright !== rawBmp) {
                rawBmp.recycle()
            }
            upright
        } catch (_: OutOfMemoryError) {
            null
        } catch (_: Exception) {
            null
        }
    }

    private fun openGalleryPicker() {
        try {
            photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        } catch (_: Exception) {
            try {
                legacyDocumentPickerLauncher.launch(arrayOf("image/*"))
            } catch (_: Exception) {
                Toast.makeText(this, "当前设备未找到可用的系统图库选择器", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun prepareAlbumSession(loadingText: String): Long {
        cancelPendingReDetect()
        resetSessionAndPerf()
        val targetSession = activeSessionId.get()
        mode = AppMode.ALBUM
        isHdRefining = false
        unbindCameraSafely()
        lockedSnapshot = null
        clearSelectedTruss()
        frozenImageView.setImageDrawable(null)
        arOverlayView.updateScene(null, 0, 0, config, null)
        updateModeUi()
        tvSummaryPill.text = loadingText
        return targetSession
    }

    private fun loadBuiltInSample() {
        val targetSession = prepareAlbumSession("正在分析内置样本图…")

        safeExecute(ioExecutor) {
            if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                return@safeExecute
            }
            val decodeOpts = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val bmp = try {
                assets.open("sample_tomato.jpg").use { stream ->
                    BitmapFactory.decodeStream(stream, null, decodeOpts)
                }?.let { decoded ->
                    if (decoded.config != Bitmap.Config.ARGB_8888) {
                        decoded.copy(Bitmap.Config.ARGB_8888, false)?.also { decoded.recycle() }
                            ?: run { decoded.recycle(); null }
                    } else {
                        decoded
                    }
                }
            } catch (_: OutOfMemoryError) {
                null
            } catch (_: Exception) {
                null
            }

            if (bmp == null) {
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        tvSummaryPill.text = "内置样本图加载失败"
                    }
                    Toast.makeText(this, "内置样本图加载失败", Toast.LENGTH_SHORT).show()
                }
                return@safeExecute
            }

            if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                bmp.recycle()
                return@safeExecute
            }

            val dispatched = safeExecute(inferenceExecutor) {
                if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                    bmp.recycle()
                    return@safeExecute
                }

                var initRes: InitResult? = null
                var localInitSeq = 0L
                if (!modelReady) {
                    modelInitInProgress.set(true)
                    localInitSeq = initSeq.incrementAndGet()
                    try {
                        initRes = initModelOnInferenceThread(config.useGpu)
                    } finally {
                        modelInitInProgress.set(false)
                    }
                }
                if (!modelReady || targetSession != activeSessionId.get()) {
                    bmp.recycle()
                    val res = initRes
                    runOnUiThread {
                        if (isDestroyed || isFinishing) return@runOnUiThread
                        if (res != null && localInitSeq == initSeq.get()) {
                            applyInitUiFeedback(res, showSuccessFallbackToast = true)
                        }
                        if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                            tvSummaryPill.text = "模型未就绪：请点击左上状态查看详情"
                        }
                    }
                    return@safeExecute
                }

                val token = imageTokenGenerator.incrementAndGet()
                val result = detector.detectFrame(bmp, config)
                if (result.success) {
                    cachedBitmapToken.set(token)
                }
                val snap = FrameSnapshot(targetSession, token, bmp, result)
                runOnUiThread {
                    if (isDestroyed || isFinishing) {
                        bmp.recycle()
                        return@runOnUiThread
                    }
                    if (initRes != null && localInitSeq == initSeq.get()) {
                        applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                    }
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        lockedSnapshot = snap
                        frozenImageView.setImageBitmap(snap.bitmap)
                        updateSaveButtonState()
                        renderSnapshot(snap)
                    } else {
                        bmp.recycle()
                    }
                }
            }
            if (!dispatched) {
                bmp.recycle()
            }
        }
    }

    private fun loadAlbumImage(uri: Uri) {
        val targetSession = prepareAlbumSession("正在分析相册图片…")

        safeExecute(ioExecutor) {
            if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                return@safeExecute
            }
            val upright = decodeUriWithExif(uri)
            if (upright == null) {
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        tvSummaryPill.text = "无法读取所选图片"
                    }
                    Toast.makeText(this, "无法读取所选图片", Toast.LENGTH_SHORT).show()
                }
                return@safeExecute
            }

            if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                upright.recycle()
                return@safeExecute
            }

            val dispatched = safeExecute(inferenceExecutor) {
                if (targetSession != activeSessionId.get() || isDestroyed || isFinishing) {
                    upright.recycle()
                    return@safeExecute
                }

                var initRes: InitResult? = null
                var localInitSeq = 0L
                if (!modelReady) {
                    modelInitInProgress.set(true)
                    localInitSeq = initSeq.incrementAndGet()
                    try {
                        initRes = initModelOnInferenceThread(config.useGpu)
                    } finally {
                        modelInitInProgress.set(false)
                    }
                }
                if (!modelReady || targetSession != activeSessionId.get()) {
                    upright.recycle()
                    val res = initRes
                    runOnUiThread {
                        if (isDestroyed || isFinishing) return@runOnUiThread
                        if (res != null && localInitSeq == initSeq.get()) {
                            applyInitUiFeedback(res, showSuccessFallbackToast = true)
                        }
                        if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                            tvSummaryPill.text = "模型未就绪：请点击左上状态查看详情"
                        }
                    }
                    return@safeExecute
                }

                val token = imageTokenGenerator.incrementAndGet()
                val result = detector.detectFrame(upright, config)
                if (result.success) {
                    cachedBitmapToken.set(token)
                }
                val snap = FrameSnapshot(targetSession, token, upright, result)
                runOnUiThread {
                    if (isDestroyed || isFinishing) {
                        upright.recycle()
                        return@runOnUiThread
                    }
                    if (initRes != null && localInitSeq == initSeq.get()) {
                        applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                    }
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        lockedSnapshot = snap
                        frozenImageView.setImageBitmap(snap.bitmap)
                        updateSaveButtonState()
                        renderSnapshot(snap)
                    } else {
                        upright.recycle()
                    }
                }
            }
            if (!dispatched) {
                upright.recycle()
            }
        }
    }

    private fun decodeUriWithExif(uri: Uri): Bitmap? {
        val resolver = applicationContext.contentResolver
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(resolver, uri)
                val decoded = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val origW = info.size.width
                    val origH = info.size.height
                    if (origW > MAX_DECODE_DIM || origH > MAX_DECODE_DIM) {
                        val scale = min(
                            MAX_DECODE_DIM.toFloat() / origW.toFloat(),
                            MAX_DECODE_DIM.toFloat() / origH.toFloat()
                        )
                        val targetW = (origW * scale).roundToInt().coerceAtLeast(1)
                        val targetH = (origH * scale).roundToInt().coerceAtLeast(1)
                        decoder.setTargetSize(targetW, targetH)
                    }
                }
                return if (decoded.config != Bitmap.Config.ARGB_8888) {
                    decoded.copy(Bitmap.Config.ARGB_8888, false)?.also { decoded.recycle() }
                        ?: run { decoded.recycle(); null }
                } else {
                    decoded
                }
            }

            val orientation = try {
                resolver.openInputStream(uri)?.use { stream ->
                    ExifInterface(stream).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                } ?: ExifInterface.ORIENTATION_NORMAL
            } catch (_: Exception) {
                ExifInterface.ORIENTATION_NORMAL
            }

            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, boundsOpts)
            }
            if (boundsOpts.outWidth <= 0 || boundsOpts.outHeight <= 0) {
                return null
            }

            var sampleSize = 1
            while ((boundsOpts.outWidth / sampleSize) > MAX_DECODE_DIM ||
                (boundsOpts.outHeight / sampleSize) > MAX_DECODE_DIM
            ) {
                sampleSize *= 2
            }

            val decodeOpts = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inSampleSize = sampleSize
            }
            val decoded = resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOpts)
            } ?: return null

            val argb = if (decoded.config != Bitmap.Config.ARGB_8888) {
                decoded.copy(Bitmap.Config.ARGB_8888, false)?.also { decoded.recycle() }
                    ?: run { decoded.recycle(); return null }
            } else {
                decoded
            }

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
                ExifInterface.ORIENTATION_TRANSPOSE -> {
                    matrix.postRotate(90f)
                    matrix.postScale(-1f, 1f)
                }
                ExifInterface.ORIENTATION_TRANSVERSE -> {
                    matrix.postRotate(270f)
                    matrix.postScale(-1f, 1f)
                }
            }
            if (matrix.isIdentity) return argb
            val rotated = Bitmap.createBitmap(argb, 0, 0, argb.width, argb.height, matrix, true)
            if (rotated !== argb) argb.recycle()
            rotated
        } catch (_: OutOfMemoryError) {
            null
        } catch (_: Exception) {
            null
        }
    }

    private fun reDetectLockedSnapshot(forceFullInfer: Boolean) {
        val snap = lockedSnapshot ?: return
        val targetSession = activeSessionId.get()
        val seq = reDetectSeq.incrementAndGet()

        safeExecute(inferenceExecutor) {
            if (!modelReady || targetSession != activeSessionId.get()) return@safeExecute
            if (seq != reDetectSeq.get()) return@safeExecute

            val cfgLatest = config
            val newResult = if (!forceFullInfer && cachedBitmapToken.get() == snap.imageToken) {
                val cachedRes = detector.repostprocessCached(cfgLatest)
                if (!cachedRes.success && cachedRes.errorCode == YoloDetector.ERR_NO_CACHE) {
                    detector.detectFrame(snap.bitmap, cfgLatest).also {
                        if (it.success) cachedBitmapToken.set(snap.imageToken)
                    }
                } else {
                    cachedRes
                }
            } else {
                detector.detectFrame(snap.bitmap, cfgLatest).also {
                    if (it.success) cachedBitmapToken.set(snap.imageToken)
                }
            }

            val updated = FrameSnapshot(targetSession, snap.imageToken, snap.bitmap, newResult)
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                if (seq == reDetectSeq.get() &&
                    targetSession == activeSessionId.get() &&
                    mode != AppMode.LIVE &&
                    lockedSnapshot?.imageToken == snap.imageToken
                ) {
                    lockedSnapshot = updated
                    updateSaveButtonState()
                    renderSnapshot(updated)
                }
            }
        }
    }

    private fun returnToLiveCamera() {
        if (!hasCameraPermission()) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        cancelPendingReDetect()
        lockedSnapshot = null
        isHdRefining = false
        frozenImageView.setImageDrawable(null)
        clearSelectedTruss()
        lastFrameError = ""
        mode = AppMode.LIVE
        resetSessionAndPerf()
        updateModeUi()
        arOverlayView.updateScene(null, 0, 0, config, null)
        renderCurrentSnapshotDetails()
        if (cameraProvider == null) {
            startCamera()
        } else {
            bindCameraUseCases()
        }
    }

    private fun updateSaveButtonState() {
        val enabled = !isHdRefining && !isExporting && lockedSnapshot?.result?.success == true
        btnSaveResult.isEnabled = enabled
        btnSaveResult.alpha = if (enabled) 1.0f else 0.5f
        btnSaveResult.text = when {
            isExporting -> "保存中…"
            isHdRefining -> "精检中…"
            else -> getString(R.string.save_annotated)
        }
    }

    private fun updateModeUi() {
        when (mode) {
            AppMode.LIVE -> {
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                previewView.visibility = View.VISIBLE
                frozenImageView.visibility = View.GONE
                liveControlRow.visibility = View.VISIBLE
                frozenControlRow.visibility = View.GONE
            }
            AppMode.FROZEN -> {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                previewView.visibility = View.INVISIBLE
                frozenImageView.visibility = View.VISIBLE
                lockedSnapshot?.let { frozenImageView.setImageBitmap(it.bitmap) }
                liveControlRow.visibility = View.GONE
                frozenControlRow.visibility = View.VISIBLE
                btnResumeLive.setText(R.string.resume_live)
            }
            AppMode.ALBUM -> {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                previewView.visibility = View.INVISIBLE
                frozenImageView.visibility = View.VISIBLE
                lockedSnapshot?.let { frozenImageView.setImageBitmap(it.bitmap) }
                liveControlRow.visibility = View.GONE
                frozenControlRow.visibility = View.VISIBLE
                btnResumeLive.setText(R.string.back_to_camera)
            }
        }
        updateSaveButtonState()
        updateStatusCapsuleUi()
    }

    private fun updateStatusCapsuleUi() {
        tvStatusLabel.text = when (mode) {
            AppMode.LIVE -> getString(R.string.status_live)
            AppMode.FROZEN -> getString(R.string.status_frozen)
            AppMode.ALBUM -> getString(R.string.status_album)
        }

        val dotDrawable = (statusDot.background?.mutate() as? GradientDrawable)
        val dotColor = when {
            !modelReady || lastFrameError.isNotEmpty() -> ContextCompat.getColor(this, R.color.accent_red)
            mode == AppMode.LIVE -> ContextCompat.getColor(this, R.color.accent_green)
            else -> ContextCompat.getColor(this, R.color.accent_orange)
        }
        dotDrawable?.setColor(dotColor)

        val subText = when {
            !modelReady && backendFallbackNotice.isNotEmpty() -> "模型异常"
            lastFrameError.isNotEmpty() -> "推理异常"
            backendFallbackNotice.startsWith("GPU 回退") -> "已回退CPU"
            else -> ""
        }
        if (subText.isNotEmpty()) {
            tvStatusSub.text = subText
            tvStatusSub.visibility = View.VISIBLE
        } else {
            tvStatusSub.visibility = View.GONE
        }
    }

    private fun activeResult(): DetectionResult? {
        return if (mode == AppMode.LIVE) latestLiveState?.result else lockedSnapshot?.result
    }

    private fun activeSourceSize(): Pair<Int, Int>? {
        return if (mode == AppMode.LIVE) {
            latestLiveState?.let { it.srcWidth to it.srcHeight }
        } else {
            lockedSnapshot?.let { it.bitmap.width to it.bitmap.height }
        }
    }

    private fun setSelectedTruss(index: Int?, result: DetectionResult?, srcW: Int, srcH: Int) {
        if (index != null && result != null && index in result.trusses.indices) {
            val t = result.trusses[index]
            selectedTrussIndex = index
            selectedTrussNormBox = TrussGeometry.toNormBox(t.x1, t.y1, t.x2, t.y2, srcW, srcH)
        } else {
            clearSelectedTruss()
        }
    }

    private fun clearSelectedTruss() {
        selectedTrussIndex = null
        selectedTrussNormBox = null
    }

    private fun reconcileSelectedTruss(result: DetectionResult, srcW: Int, srcH: Int) {
        val targetBox = selectedTrussNormBox ?: run {
            if (selectedTrussIndex != null && selectedTrussIndex !in result.trusses.indices) {
                selectedTrussIndex = null
            }
            return
        }
        if (!result.success || result.trusses.isEmpty()) {
            clearSelectedTruss()
            return
        }
        val matchedIdx = TrussGeometry.matchTrussIndex(targetBox, result.trusses, srcW, srcH, minIou = 0.25f)
        if (matchedIdx != null) {
            val matched = result.trusses[matchedIdx]
            selectedTrussIndex = matchedIdx
            selectedTrussNormBox = TrussGeometry.toNormBox(matched.x1, matched.y1, matched.x2, matched.y2, srcW, srcH)
        } else {
            clearSelectedTruss()
        }
    }

    private fun redrawActiveOverlay() {
        val res = activeResult() ?: return
        val size = activeSourceSize() ?: return
        arOverlayView.updateScene(
            result = res,
            sourceWidth = size.first,
            sourceHeight = size.second,
            detectorConfig = config,
            selectedIndex = selectedTrussIndex
        )
    }

    private fun renderLiveState(state: LiveFrameState) {
        reconcileSelectedTruss(state.result, state.srcWidth, state.srcHeight)
        arOverlayView.updateScene(
            result = state.result,
            sourceWidth = state.srcWidth,
            sourceHeight = state.srcHeight,
            detectorConfig = config,
            selectedIndex = selectedTrussIndex
        )
        syncErrorNotice(state.result)
        if (techCardExpanded) {
            updateTechHudText(state.srcWidth, state.srcHeight, state.result)
        }
        renderCurrentSnapshotDetails()
    }

    private fun renderSnapshot(snapshot: FrameSnapshot) {
        reconcileSelectedTruss(snapshot.result, snapshot.bitmap.width, snapshot.bitmap.height)
        arOverlayView.updateScene(
            result = snapshot.result,
            sourceWidth = snapshot.bitmap.width,
            sourceHeight = snapshot.bitmap.height,
            detectorConfig = config,
            selectedIndex = selectedTrussIndex
        )
        syncErrorNotice(snapshot.result)
        if (techCardExpanded) {
            updateTechHudText(snapshot.bitmap.width, snapshot.bitmap.height, snapshot.result)
        }
        renderCurrentSnapshotDetails()
    }

    private fun syncErrorNotice(result: DetectionResult) {
        val nextErr = if (result.success) "" else result.errorMessage
        if (lastFrameError != nextErr) {
            lastFrameError = nextErr
            updateStatusCapsuleUi()
        }
    }

    private fun refreshTechHudFromActiveState() {
        val size = activeSourceSize()
        val res = activeResult()
        updateTechHudText(size?.first, size?.second, res)
    }

    private fun updateTechHudText(srcWidth: Int?, srcHeight: Int?, result: DetectionResult?) {
        tvTechBackend.text = "计算后端：$activeBackendLabel"
        if (srcWidth != null && srcHeight != null) {
            tvTechResolution.text = "输入尺寸：${srcWidth}×${srcHeight} → 模型 512×512"
        } else {
            tvTechResolution.text = "采集尺寸：-- → 模型 512×512"
        }

        if (result == null) {
            tvTechPerf.text = "单帧耗时：-- ms ｜ 实测 FPS：--"
        } else if (result.success) {
            val fpsText = if (mode == AppMode.LIVE && measuredFps > 0f) {
                String.format(Locale.US, "%.1f", measuredFps)
            } else {
                "--"
            }
            tvTechPerf.text = String.format(Locale.US, "单帧耗时：%.1f ms ｜ 实测 FPS：%s", result.inferMs, fpsText)
        } else {
            tvTechPerf.text = "单帧耗时：失败"
        }

        val combinedNotice = when {
            lastFrameError.isNotEmpty() -> lastFrameError
            backendFallbackNotice.isNotEmpty() -> backendFallbackNotice
            else -> ""
        }
        if (combinedNotice.isNotEmpty()) {
            tvTechNotice.text = combinedNotice
            tvTechNotice.visibility = View.VISIBLE
        } else {
            tvTechNotice.visibility = View.GONE
        }
    }

    private fun renderCurrentSnapshotDetails() {
        val res = activeResult()
        if (res == null) {
            tvSummaryPill.text = if (modelReady) "正在识别…" else "正在初始化模型…"
            selectedTrussCard.visibility = View.GONE
            if (summaryExpanded) {
                populateSummaryDrawer(null, force = false)
            }
            return
        }
        if (!res.success) {
            tvSummaryPill.text = "推理异常：点击左上状态查看详情"
            selectedTrussCard.visibility = View.GONE
            if (summaryExpanded) {
                populateSummaryDrawer(null, force = false)
            }
            return
        }

        tvSummaryPill.text = if (res.trusses.isEmpty()) {
            "暂未检测到果穗 · 检出单果 ${res.fruits.size} 颗"
        } else {
            "${res.trusses.size} 串果穗 · 已归属 ${res.assignedTotal} 颗"
        }

        val selIdx = selectedTrussIndex
        if (selIdx != null && selIdx in res.trusses.indices) {
            val t = res.trusses[selIdx]
            selectedTrussCard.visibility = View.VISIBLE
            tvSelectedGradeBadge.text = t.gradeLetter
            (tvSelectedGradeBadge.background?.mutate() as? GradientDrawable)?.setColor(ArOverlayView.gradeColor(t.gradeCode))
            tvSelectedTitle.text = "果穗 #${t.id} · ${t.gradeLetter}级 · ${t.fullAdvice}"
            tvSelectedCounts.text = "成熟度 ${t.maturityPercentText}  ｜  成熟 ${t.ripeCount} · 转色 ${t.turningCount} · 未熟 ${t.immatureCount}"
        } else {
            selectedTrussCard.visibility = View.GONE
        }

        if (summaryExpanded) {
            populateSummaryDrawer(res, force = false)
        }
    }

    private fun createPooledDrawerRow(): TextView {
        val dp = resources.displayMetrics.density
        return TextView(this).apply {
            textSize = 12.5f
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (6f * dp).roundToInt(), 0, (6f * dp).roundToInt())
            minHeight = (48f * dp).roundToInt()
            if (selectableRowBgResId != 0) {
                setBackgroundResource(selectableRowBgResId)
            }
            setOnClickListener { v ->
                val clickedIdx = v.tag as? Int ?: return@setOnClickListener
                val curRes = activeResult() ?: return@setOnClickListener
                val curSize = activeSourceSize() ?: return@setOnClickListener
                if (clickedIdx in curRes.trusses.indices) {
                    setSelectedTruss(clickedIdx, curRes, curSize.first, curSize.second)
                    summaryExpanded = false
                    updateSummaryExpandedUi()
                    renderCurrentSnapshotDetails()
                    redrawActiveOverlay()
                }
            }
        }
    }

    private fun populateSummaryDrawer(res: DetectionResult?, force: Boolean = true) {
        if (!force && mode == AppMode.LIVE) {
            val nowMs = SystemClock.uptimeMillis()
            if (isSummaryScrolling || nowMs - lastLiveDrawerUpdateMs < 300L) {
                return
            }
            lastLiveDrawerUpdateMs = nowMs
        }

        if (res == null || !res.success) {
            tvExpandedOverall.text = "暂无检测结果"
            for (i in 0 until trussListContainer.childCount) {
                trussListContainer.getChildAt(i).visibility = View.GONE
            }
            updateSummaryScrollHeight()
            return
        }

        tvExpandedOverall.text =
            "检出 ${res.fruits.size} 颗 ｜ 已归属 ${res.assignedTotal} 颗 [成熟 ${res.assignedRipe} · 转色 ${res.assignedTurning} · 未熟 ${res.assignedImmature}] ｜ 未归属 ${res.unassignedCount} 颗 ｜ 果穗整体成熟度 ${res.overallAssignedPercentText}"

        if (res.trusses.isEmpty()) {
            val row = if (trussListContainer.childCount > 0) {
                trussListContainer.getChildAt(0) as TextView
            } else {
                createPooledDrawerRow().also { trussListContainer.addView(it) }
            }
            row.visibility = View.VISIBLE
            row.tag = null
            row.isClickable = false
            row.text = "当前画面未识别到完整果穗"
            row.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            for (i in 1 until trussListContainer.childCount) {
                trussListContainer.getChildAt(i).visibility = View.GONE
            }
        } else {
            val normalColor = ContextCompat.getColor(this, R.color.text_primary)
            val selectedColor = Color.parseColor("#FFD60A")
            val count = res.trusses.size
            while (trussListContainer.childCount < count) {
                trussListContainer.addView(createPooledDrawerRow())
            }
            for (idx in 0 until count) {
                val t = res.trusses[idx]
                val row = trussListContainer.getChildAt(idx) as TextView
                row.visibility = View.VISIBLE
                row.isClickable = true
                row.tag = idx
                row.text =
                    "果穗 #${t.id}：${t.gradeLetter}级 (${t.maturityPercentText})  [熟 ${t.ripeCount} / 转 ${t.turningCount} / 青 ${t.immatureCount}]  →  ${t.shortAdvice}"
                row.setTextColor(if (selectedTrussIndex == idx) selectedColor else normalColor)
            }
            for (i in count until trussListContainer.childCount) {
                trussListContainer.getChildAt(i).visibility = View.GONE
            }
        }

        updateSummaryScrollHeight()
    }

    private fun updateSummaryScrollHeight() {
        if (summaryScrollMaxHeightPx <= 0) return
        val applyHeight = { contentHeight: Int ->
            if (!isDestroyed && !isFinishing && summaryScrollMaxHeightPx > 0) {
                val lp = summaryScrollView.layoutParams
                val targetH = if (contentHeight > summaryScrollMaxHeightPx) {
                    summaryScrollMaxHeightPx
                } else {
                    ViewGroup.LayoutParams.WRAP_CONTENT
                }
                if (lp.height != targetH) {
                    lp.height = targetH
                    summaryScrollView.layoutParams = lp
                }
            }
        }

        if (summaryScrollView.width > 0) {
            val wSpec = View.MeasureSpec.makeMeasureSpec(summaryScrollView.width, View.MeasureSpec.EXACTLY)
            val hSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
            trussListContainer.measure(wSpec, hSpec)
            applyHeight(trussListContainer.measuredHeight)
        } else {
            summaryScrollView.post {
                if (summaryScrollView.width > 0) {
                    val wSpec = View.MeasureSpec.makeMeasureSpec(summaryScrollView.width, View.MeasureSpec.EXACTLY)
                    val hSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
                    trussListContainer.measure(wSpec, hSpec)
                    applyHeight(trussListContainer.measuredHeight)
                } else {
                    applyHeight(trussListContainer.height)
                }
            }
        }
    }

    private fun exportCurrentSnapshot() {
        val snap = lockedSnapshot
        if (snap == null || !snap.result.success) {
            Toast.makeText(this, "当前没有可保存的检测结果", Toast.LENGTH_SHORT).show()
            return
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingSaveAfterPermission = true
            storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return
        }

        val cfgCopy = config
        isExporting = true
        updateSaveButtonState()

        val submitted = safeExecute(ioExecutor) {
            var outBmp: Bitmap? = null
            val savedPath = try {
                outBmp = ArOverlayView.renderExportBitmap(snap.bitmap, snap.result, cfgCopy)
                if (outBmp != null) saveBitmapToGallery(outBmp) else null
            } catch (_: Throwable) {
                null
            } finally {
                outBmp?.recycle()
            }

            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                isExporting = false
                updateSaveButtonState()
                if (savedPath != null) {
                    Toast.makeText(this, "已保存标注图：$savedPath", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "保存标注图失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
        if (!submitted) {
            isExporting = false
            updateSaveButtonState()
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap): String? {
        val name = "TomatoAR_" + SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date()) + ".jpg"
        val appCtx = applicationContext
        val resolver = appCtx.contentResolver
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/TomatoAR")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
                try {
                    val compressed = resolver.openOutputStream(uri)?.use { out ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                    } ?: false
                    if (!compressed) {
                        resolver.delete(uri, null, null)
                        return null
                    }
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                    "Pictures/TomatoAR/$name"
                } catch (_: Exception) {
                    try {
                        resolver.delete(uri, null, null)
                    } catch (_: Exception) {
                    }
                    null
                }
            } else {
                @Suppress("DEPRECATION")
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "TomatoAR")
                if (!dir.exists() && !dir.mkdirs()) {
                    return null
                }
                val file = File(dir, name)
                try {
                    val compressed = FileOutputStream(file).use { out ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                    }
                    if (!compressed) {
                        file.delete()
                        return null
                    }
                    MediaScannerConnection.scanFile(appCtx, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                    file.absolutePath
                } catch (_: Exception) {
                    file.delete()
                    null
                }
            }
        } catch (_: Exception) {
            null
        }
    }

    companion object {
        private const val PREFS_NAME = "tomato_ar_prefs"
        private const val KEY_CONF_THRESH = "conf_thresh"
        private const val KEY_IOU_THRESH = "iou_thresh"
        private const val KEY_AGNOSTIC_NMS = "agnostic_fruit_nms"
        private const val KEY_ASSIGN_NEAREST = "assign_nearest_center"
        private const val KEY_SHOW_FRUIT_BOXES = "show_fruit_boxes"
        private const val KEY_SHOW_FRUIT_LABELS = "show_fruit_labels"
        private const val KEY_LARGE_TEXT = "large_text_for_glasses"
        private const val KEY_USE_GPU = "use_gpu"
        private const val KEY_CAPTURE_RES = "capture_res_index"
        private const val MAX_DECODE_DIM = 2048
    }
}
