package com.hzau.tomatoar

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.drawable.GradientDrawable
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Rational
import android.util.Size
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
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
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
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val ioExecutor = Executors.newSingleThreadExecutor()
    private val activeSessionId = AtomicLong(1L)
    private val imageTokenGenerator = AtomicLong(1L)
    private val cachedBitmapToken = AtomicLong(0L)
    private val freezeRequested = AtomicBoolean(false)
    private val reDetectSeq = AtomicLong(0L)

    @Volatile private var mode: AppMode = AppMode.LIVE
    @Volatile private var config: DetectorConfig = DetectorConfig()
    @Volatile private var modelReady = false
    @Volatile private var activeBackendLabel = "初始化中…"
    @Volatile private var backendFallbackNotice = ""
    @Volatile private var lastFrameError = ""
    @Volatile private var latestLiveState: LiveFrameState? = null
    @Volatile private var lockedSnapshot: FrameSnapshot? = null
    @Volatile private var measuredFps = 0f

    // FPS tracking owned exclusively by inferenceExecutor
    private var fpsSessionId = -1L
    private var fpsWindowStartNs = 0L
    private var fpsFrameCount = 0

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var cameraProvider: ProcessCameraProvider? = null
    private var selectedTrussIndex: Int? = null
    private var selectedTrussBox: RectF? = null
    private var techCardExpanded = false
    private var summaryExpanded = false
    private var settingsVisible = false
    private var pendingSaveAfterPermission = false
    private var summaryScrollMaxHeightPx = 0

    // Views
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
                Toast.makeText(this, "未授予相机权限，已切换为样本图/相册测图模式", Toast.LENGTH_LONG).show()
                loadBuiltInSample()
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
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

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

    override fun onDestroy() {
        super.onDestroy()
        freezeRequested.set(false)
        cameraProvider?.unbindAll()
        inferenceExecutor.shutdownNow()
        ioExecutor.shutdownNow()
    }

    private fun loadPersistedConfig(): DetectorConfig {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        return DetectorConfig(
            confThresh = prefs.getFloat(KEY_CONF_THRESH, 0.35f).coerceIn(0.10f, 0.85f),
            iouThresh = prefs.getFloat(KEY_IOU_THRESH, 0.70f).coerceIn(0.30f, 0.90f),
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

    private inline fun updateConfig(transform: (DetectorConfig) -> DetectorConfig) {
        val next = transform(config)
        if (next != config) {
            config = next
            saveConfig(next)
        }
    }

    private fun bindViews() {
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
        val screenH = dm.heightPixels
        summaryScrollMaxHeightPx = min((220f * dp).roundToInt(), (screenH * 0.30f).roundToInt())
        val settingsScrollH = min((420f * dp).roundToInt(), (screenH * 0.56f).roundToInt())
        settingsScrollView.layoutParams = settingsScrollView.layoutParams.apply {
            height = settingsScrollH
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.rootContainer)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val topPad = bars.top + (10f * dp).roundToInt()
            val sidePad = (14f * dp).roundToInt()
            val bottomPad = bars.bottom + (16f * dp).roundToInt()
            topHudContainer.setPadding(sidePad, topPad, sidePad, 0)
            bottomStack.setPadding(sidePad, 0, sidePad, bottomPad)
            arOverlayView.setSafeTopInsetPx(bars.top + 66f * dp)
            insets
        }
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
            setSelectedTruss(idx, activeResult())
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

        var pillTouchStartY = 0f
        btnSummaryPill.setOnTouchListener { _, ev ->
            when (ev.action) {
                MotionEvent.ACTION_DOWN -> pillTouchStartY = ev.rawY
                MotionEvent.ACTION_UP -> {
                    val dy = ev.rawY - pillTouchStartY
                    val swipeThreshold = -24f * resources.displayMetrics.density
                    if (dy < swipeThreshold && !summaryExpanded) {
                        summaryExpanded = true
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
            if (latestLiveState == null) {
                Toast.makeText(this, "等待首帧检测完成…", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            freezeRequested.set(true)
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

        // Detection SeekBars
        seekConf.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val v = 0.10f + progress / 100f
                tvConfValue.text = String.format(Locale.US, "%.2f", v)
                if (fromUser) {
                    updateConfig { it.copy(confThresh = v) }
                    if (mode != AppMode.LIVE) {
                        reDetectLockedSnapshot(forceFullInfer = false)
                    }
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        seekIou.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val v = 0.30f + progress / 100f
                tvIouValue.text = String.format(Locale.US, "%.2f", v)
                if (fromUser) {
                    updateConfig { it.copy(iouThresh = v) }
                    if (mode != AppMode.LIVE) {
                        reDetectLockedSnapshot(forceFullInfer = false)
                    }
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
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
            populateSummaryDrawer(activeResult())
        }
    }

    private fun syncSettingsUi() {
        val cfg = config
        switchShowFruitBoxes.isChecked = cfg.showFruitBoxes
        switchShowFruitLabels.isChecked = cfg.showFruitLabels
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
        btn.setBackgroundResource(if (active) R.drawable.bg_pill_segment_active else R.drawable.bg_pill_segment_inactive)
        btn.setTextColor(ContextCompat.getColor(this, if (active) R.color.black else R.color.white))
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
        }
    }

    private fun initModelOnInferenceThread(useGpu: Boolean): InitResult {
        cachedBitmapToken.set(0L)
        val initRes = detector.initialize(assets, useGpu)
        modelReady = initRes.success
        activeBackendLabel = initRes.activeBackend
        backendFallbackNotice = if (initRes.fallback) {
            "GPU 回退：${initRes.reason}"
        } else if (!initRes.success) {
            initRes.reason
        } else {
            ""
        }
        return initRes
    }

    private fun initializeModelAsync(useGpu: Boolean, reRunFrozenAfterInit: Boolean) {
        modelReady = false
        activeBackendLabel = if (useGpu) "正在切换 Vulkan GPU…" else "正在初始化 CPU…"
        backendFallbackNotice = ""
        lastFrameError = ""
        updateStatusCapsuleUi()
        if (techCardExpanded) {
            refreshTechHudFromActiveState()
        }

        val targetSession = activeSessionId.get()
        inferenceExecutor.execute {
            val initRes = initModelOnInferenceThread(useGpu)
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                if (reRunFrozenAfterInit && initRes.success && targetSession == activeSessionId.get() && mode != AppMode.LIVE) {
                    reDetectLockedSnapshot(forceFullInfer = true)
                }
            }
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
        previewView.post {
            if (isDestroyed || isFinishing || mode != AppMode.LIVE) return@post

            val targetSize = targetResolutionSize()
            val aspectStrategy = if (config.captureResIndex == 0) {
                AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            } else {
                AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
            }
            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(aspectStrategy)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        targetSize,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(rotation)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            val boundSession = activeSessionId.get()
            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT

            imageAnalysis.setAnalyzer(inferenceExecutor) { imageProxy ->
                try {
                    if (mode != AppMode.LIVE || !modelReady || boundSession != activeSessionId.get()) {
                        return@setAnalyzer
                    }
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
                        if (boundSession != activeSessionId.get() || mode != AppMode.LIVE) {
                            upright.recycle()
                            return@setAnalyzer
                        }
                        val token = imageTokenGenerator.incrementAndGet()
                        val result = detector.detectFrame(upright, cfgCopy)
                        if (result.success) {
                            cachedBitmapToken.set(token)
                        }
                        if (boundSession != activeSessionId.get() || mode != AppMode.LIVE) {
                            upright.recycle()
                            return@setAnalyzer
                        }
                        val snap = FrameSnapshot(boundSession, token, upright, result)
                        runOnUiThread {
                            if (isDestroyed || isFinishing) {
                                upright.recycle()
                                return@runOnUiThread
                            }
                            if (mode == AppMode.LIVE && snap.sessionId == activeSessionId.get()) {
                                lockedSnapshot = snap
                                mode = AppMode.FROZEN
                                cameraProvider?.unbindAll()
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

                    if (boundSession != activeSessionId.get() || mode != AppMode.LIVE) {
                        return@setAnalyzer
                    }

                    updateFpsOnInferenceThread(boundSession)
                    val state = LiveFrameState(boundSession, uprightW, uprightH, result)
                    latestLiveState = state
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing && mode == AppMode.LIVE && state.sessionId == activeSessionId.get()) {
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

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, group)
            } catch (e: Exception) {
                Toast.makeText(this, "摄像头绑定失败：${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateFpsOnInferenceThread(sessionId: Long) {
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

        val rawBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        if (rowStride == tightRowBytes && buffer.remaining() >= tightRowBytes * h) {
            rawBmp.copyPixelsFromBuffer(buffer)
        } else {
            val tightBuffer = ByteBuffer.allocateDirect(tightRowBytes * h)
            val rowTmp = ByteArray(tightRowBytes)
            for (y in 0 until h) {
                val rowOffset = y * rowStride
                if (rowOffset + tightRowBytes > buffer.limit()) {
                    rawBmp.recycle()
                    return null
                }
                buffer.position(rowOffset)
                buffer.get(rowTmp, 0, tightRowBytes)
                tightBuffer.put(rowTmp)
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
        return upright
    }

    private fun openGalleryPicker() {
        try {
            photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        } catch (_: Exception) {
            legacyDocumentPickerLauncher.launch(arrayOf("image/*"))
        }
    }

    private fun prepareAlbumSession(loadingText: String): Long {
        resetSessionAndPerf()
        val targetSession = activeSessionId.get()
        mode = AppMode.ALBUM
        cameraProvider?.unbindAll()
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

        inferenceExecutor.execute {
            val bmp = try {
                assets.open("sample_tomato.jpg").use { stream ->
                    BitmapFactory.decodeStream(stream)
                }?.let { decoded ->
                    if (decoded.config != Bitmap.Config.ARGB_8888) {
                        decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
                    } else {
                        decoded
                    }
                }
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
                return@execute
            }

            var initRes: InitResult? = null
            if (!modelReady) {
                initRes = initModelOnInferenceThread(config.useGpu)
            }
            if (!modelReady) {
                bmp.recycle()
                val res = initRes
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    if (res != null) applyInitUiFeedback(res, showSuccessFallbackToast = true)
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        tvSummaryPill.text = "模型未就绪：请点击左上状态查看详情"
                    }
                }
                return@execute
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
                if (initRes != null) applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                    lockedSnapshot = snap
                    frozenImageView.setImageBitmap(snap.bitmap)
                    renderSnapshot(snap)
                } else {
                    bmp.recycle()
                }
            }
        }
    }

    private fun loadAlbumImage(uri: Uri) {
        val targetSession = prepareAlbumSession("正在分析相册图片…")

        inferenceExecutor.execute {
            val upright = decodeUriWithExif(uri)
            if (upright == null) {
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        tvSummaryPill.text = "无法读取所选图片"
                    }
                    Toast.makeText(this, "无法读取所选图片", Toast.LENGTH_SHORT).show()
                }
                return@execute
            }

            var initRes: InitResult? = null
            if (!modelReady) {
                initRes = initModelOnInferenceThread(config.useGpu)
            }
            if (!modelReady) {
                upright.recycle()
                val res = initRes
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    if (res != null) applyInitUiFeedback(res, showSuccessFallbackToast = true)
                    if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                        tvSummaryPill.text = "模型未就绪：请点击左上状态查看详情"
                    }
                }
                return@execute
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
                if (initRes != null) applyInitUiFeedback(initRes, showSuccessFallbackToast = true)
                if (targetSession == activeSessionId.get() && mode == AppMode.ALBUM) {
                    lockedSnapshot = snap
                    frozenImageView.setImageBitmap(snap.bitmap)
                    renderSnapshot(snap)
                } else {
                    upright.recycle()
                }
            }
        }
    }

    private fun decodeUriWithExif(uri: Uri): Bitmap? {
        return try {
            val orientation = try {
                contentResolver.openInputStream(uri)?.use { stream ->
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
            contentResolver.openInputStream(uri)?.use { stream ->
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
            val decoded = contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOpts)
            } ?: return null

            val argb = if (decoded.config != Bitmap.Config.ARGB_8888) {
                decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
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
        val cfgCopy = config
        val seq = reDetectSeq.incrementAndGet()

        inferenceExecutor.execute {
            if (!modelReady || targetSession != activeSessionId.get()) return@execute
            if (!forceFullInfer && seq != reDetectSeq.get()) return@execute

            val newResult = if (!forceFullInfer && cachedBitmapToken.get() == snap.imageToken) {
                val cachedRes = detector.repostprocessCached(cfgCopy)
                if (!cachedRes.success && cachedRes.errorCode == YoloDetector.ERR_NO_CACHE) {
                    detector.detectFrame(snap.bitmap, cfgCopy).also {
                        if (it.success) cachedBitmapToken.set(snap.imageToken)
                    }
                } else {
                    cachedRes
                }
            } else {
                detector.detectFrame(snap.bitmap, cfgCopy).also {
                    if (it.success) cachedBitmapToken.set(snap.imageToken)
                }
            }

            val updated = FrameSnapshot(targetSession, snap.imageToken, snap.bitmap, newResult)
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                if (targetSession == activeSessionId.get() && mode != AppMode.LIVE) {
                    lockedSnapshot = updated
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
        lockedSnapshot = null
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

    private fun updateModeUi() {
        when (mode) {
            AppMode.LIVE -> {
                previewView.visibility = View.VISIBLE
                frozenImageView.visibility = View.GONE
                liveControlRow.visibility = View.VISIBLE
                frozenControlRow.visibility = View.GONE
            }
            AppMode.FROZEN -> {
                previewView.visibility = View.INVISIBLE
                frozenImageView.visibility = View.VISIBLE
                lockedSnapshot?.let { frozenImageView.setImageBitmap(it.bitmap) }
                liveControlRow.visibility = View.GONE
                frozenControlRow.visibility = View.VISIBLE
                btnResumeLive.setText(R.string.resume_live)
            }
            AppMode.ALBUM -> {
                previewView.visibility = View.INVISIBLE
                frozenImageView.visibility = View.VISIBLE
                lockedSnapshot?.let { frozenImageView.setImageBitmap(it.bitmap) }
                liveControlRow.visibility = View.GONE
                frozenControlRow.visibility = View.VISIBLE
                btnResumeLive.setText(R.string.back_to_camera)
            }
        }
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

    private fun setSelectedTruss(index: Int?, result: DetectionResult?) {
        if (index != null && result != null && index in result.trusses.indices) {
            val t = result.trusses[index]
            selectedTrussIndex = index
            selectedTrussBox = RectF(t.x1, t.y1, t.x2, t.y2)
        } else {
            clearSelectedTruss()
        }
    }

    private fun clearSelectedTruss() {
        selectedTrussIndex = null
        selectedTrussBox = null
    }

    private fun reconcileSelectedTruss(result: DetectionResult) {
        val targetBox = selectedTrussBox ?: run {
            if (selectedTrussIndex != null && selectedTrussIndex !in result.trusses.indices) {
                selectedTrussIndex = null
            }
            return
        }
        if (!result.success || result.trusses.isEmpty()) {
            clearSelectedTruss()
            return
        }
        var bestIdx = -1
        var bestIou = 0f
        for ((idx, t) in result.trusses.withIndex()) {
            val iou = rectIou(targetBox, t.x1, t.y1, t.x2, t.y2)
            if (iou > bestIou) {
                bestIou = iou
                bestIdx = idx
            }
        }
        if (bestIdx >= 0 && bestIou >= 0.25f) {
            val matched = result.trusses[bestIdx]
            selectedTrussIndex = bestIdx
            targetBox.set(matched.x1, matched.y1, matched.x2, matched.y2)
        } else {
            clearSelectedTruss()
        }
    }

    private fun rectIou(a: RectF, bx1: Float, by1: Float, bx2: Float, by2: Float): Float {
        val ix1 = max(a.left, bx1)
        val iy1 = max(a.top, by1)
        val ix2 = min(a.right, bx2)
        val iy2 = min(a.bottom, by2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, bx2 - bx1) * max(0f, by2 - by1)
        val denom = areaA + areaB - inter
        return if (denom > 0f) inter / denom else 0f
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
        reconcileSelectedTruss(state.result)
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
        reconcileSelectedTruss(snapshot.result)
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
                populateSummaryDrawer(null)
            }
            return
        }
        if (!res.success) {
            tvSummaryPill.text = "推理异常：点击左上状态查看详情"
            selectedTrussCard.visibility = View.GONE
            if (summaryExpanded) {
                populateSummaryDrawer(null)
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
            populateSummaryDrawer(res)
        }
    }

    private fun populateSummaryDrawer(res: DetectionResult?) {
        trussListContainer.removeAllViews()
        if (res == null || !res.success) {
            tvExpandedOverall.text = "暂无检测结果"
            return
        }

        tvExpandedOverall.text =
            "检出 ${res.fruits.size} 颗 ｜ 已归属 ${res.assignedTotal} 颗 [成熟 ${res.assignedRipe} · 转色 ${res.assignedTurning} · 未熟 ${res.assignedImmature}] ｜ 未归属 ${res.unassignedCount} 颗 ｜ 果穗整体成熟度 ${res.overallAssignedPercentText}"

        if (res.trusses.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "当前画面未识别到完整果穗"
                setTextColor(ContextCompat.getColor(context, R.color.text_secondary))
                textSize = 12.5f
            }
            trussListContainer.addView(emptyTv)
        } else {
            val dp = resources.displayMetrics.density
            for ((idx, t) in res.trusses.withIndex()) {
                val row = TextView(this).apply {
                    text = "果穗 #${t.id}：${t.gradeLetter}级 (${t.maturityPercentText})  [熟 ${t.ripeCount} / 转 ${t.turningCount} / 青 ${t.immatureCount}]  →  ${t.shortAdvice}"
                    setTextColor(
                        if (selectedTrussIndex == idx) Color.parseColor("#FFD60A")
                        else ContextCompat.getColor(context, R.color.text_primary)
                    )
                    textSize = 12.5f
                    setPadding(0, (6f * dp).roundToInt(), 0, (6f * dp).roundToInt())
                    minHeight = (40f * dp).roundToInt()
                    setOnClickListener {
                        setSelectedTruss(idx, res)
                        summaryExpanded = false
                        updateSummaryExpandedUi()
                        renderCurrentSnapshotDetails()
                        redrawActiveOverlay()
                    }
                }
                trussListContainer.addView(row)
            }
        }

        summaryScrollView.post {
            if (isDestroyed || isFinishing || summaryScrollMaxHeightPx <= 0) return@post
            val lp = summaryScrollView.layoutParams
            val targetH = if (trussListContainer.height > summaryScrollMaxHeightPx) {
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
        val selCopy = selectedTrussIndex
        btnSaveResult.isEnabled = false

        ioExecutor.execute {
            val outBmp = ArOverlayView.renderExportBitmap(snap.bitmap, snap.result, cfgCopy, selCopy)
            val savedPath = saveBitmapToGallery(outBmp)
            outBmp.recycle()
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                btnSaveResult.isEnabled = true
                if (savedPath != null) {
                    Toast.makeText(this, "已保存标注图：$savedPath", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "保存标注图失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap): String? {
        val name = "TomatoAR_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/TomatoAR")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
                try {
                    val compressed = contentResolver.openOutputStream(uri)?.use { out ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                    } ?: false
                    if (!compressed) {
                        contentResolver.delete(uri, null, null)
                        return null
                    }
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    contentResolver.update(uri, values, null, null)
                    "Pictures/TomatoAR/$name"
                } catch (e: Exception) {
                    try {
                        contentResolver.delete(uri, null, null)
                    } catch (_: Exception) {
                    }
                    null
                }
            } else {
                @Suppress("DEPRECATION")
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "TomatoAR")
                if (!dir.exists()) dir.mkdirs()
                val file = File(dir, name)
                val compressed = FileOutputStream(file).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
                if (!compressed) {
                    file.delete()
                    return null
                }
                MediaScannerConnection.scanFile(this, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)
                file.absolutePath
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
