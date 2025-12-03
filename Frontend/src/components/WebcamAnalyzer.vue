<template>
  <div
    class="webcam-analyzer"
    :class="['o-' + orientation]"
    :style="orientation === 'portrait'
      ? { '--p-video-height': pSize.height, '--p-scale': pSize.scale, '--p-max-w': pSize.maxW }
      : { '--l-video-height': lSize.height, '--l-max-w': lSize.maxW }"
  >
    <div class="video-frame">
      <div class="video-container" ref="containerEl">
        <video
          ref="videoElement"
          class="video-feed"
          autoplay
          playsinline
          muted
        ></video>

        <canvas ref="canvasElement" class="pose-overlay"></canvas>

        <div 
          v-if="startupMode" 
          class="detection-frame-container"
        >
          <div class="detection-frame-text">請站入橘色框內</div>
        </div>

        <div v-if="!isStreaming" class="video-placeholder">
          <div class="placeholder-content">
            <p>📹 Camera Off</p>
            <button @click="startCamera" class="start-button">Camera On</button>
          </div>
        </div>

        <div v-if="error" class="error-message">
          {{ error }}
        </div>
      </div>
    </div>

    <div class="controls">
      <button @click="toggleCamera" class="control-button">
        {{ isStreaming ? 'Stop' : 'Begin' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'

const props = defineProps({
  analyzeInterval: { type: Number, default: 100 },
  orientation: {
    type: String,
    default: 'landscape',
    validator: v => ['portrait', 'landscape'].includes(v)
  },
  landscapeSize: { type: Object, default: () => ({ height: '56vh', maxW: '980px' }) },
  portraitSize:  { type: Object, default: () => ({ height: '52vh', scale: 1.08, maxW: '640px' }) },
  startupMode: { type: Boolean, default: false } 
})

const emit = defineEmits(['startup-confirmed', 'person-detected', 'person-lost', 'frame-analyzed'])

const lSize = computed(() => ({ height: props.landscapeSize.height, maxW: props.landscapeSize.maxW }))
const pSize = computed(() => ({ height: props.portraitSize.height, scale: props.portraitSize.scale ?? 1.08, maxW: props.portraitSize.maxW }))

const targetRatio = computed(() => props.orientation === 'portrait' ? 9/16 : 16/9)

const mediapipeStore = useMediapipeStore()

const containerEl = ref(null)
const videoElement = ref(null)
const canvasElement = ref(null)
const isStreaming = ref(false)
const mirror = ref(true)
const error = ref('')
const stream = ref(null)
const analyzeTimer = ref(null)
let ro

const personInFrame = ref(false)
let personDetectedFrames = 0
let personLostFrames = 0
const DETECTION_THRESHOLD = 3 
const LOST_THRESHOLD = 5 

function attachResizeObserver() {
  if (!containerEl.value || !canvasElement.value) return
  const canvas = canvasElement.value
  const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true })
  const update = () => {
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2))
    const rect = containerEl.value.getBoundingClientRect()
    canvas.style.width = rect.width + 'px'
    canvas.style.height = rect.height + 'px'
    canvas.width = Math.round(rect.width * dpr)
    canvas.height = Math.round(rect.height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }
  ro = new ResizeObserver(update)
  ro.observe(containerEl.value)
  update()
}

async function startCamera() {
  try {
    error.value = ''

    const base = {
      facingMode: 'user',
      frameRate: { ideal: 24, max: 24 },
      width:  { ideal: props.orientation === 'portrait' ? 720 : 1280, max: 1280 },
      height: { ideal: props.orientation === 'portrait' ? 1280 : 720, max: 1280 },
      resizeMode: 'none', 
    }

    const constraints = {
      audio: false,
      video: {
        ...base,
        aspectRatio: props.orientation === 'portrait' ? { ideal: 9/16 } : { ideal: 16/9 },
        resizeMode: 'crop-and-scale',
        advanced: [
          { exposureMode: 'continuous' },
          { focusMode: 'continuous' },
          { whiteBalanceMode: 'continuous' },
          { torch: false }
        ]
      }
    }

    stream.value = await navigator.mediaDevices.getUserMedia(constraints)

    if (videoElement.value) {
      videoElement.value.srcObject = stream.value
      videoElement.value.playsInline = true
      videoElement.value.muted = true
      await videoElement.value.play().catch(() => {})

      isStreaming.value = true
      startAnalysis()
    }
  } catch (err) {
    error.value = '无法访问摄像头：' + (err?.message || err)
  }
}

function stopCamera() {
  if (stream.value) {
    stream.value.getTracks().forEach(t => t.stop())
    stream.value = null
  }
  if (videoElement.value) videoElement.value.srcObject = null
  isStreaming.value = false
  stopAnalysis()
}

function toggleCamera() {
  isStreaming.value ? stopCamera() : startCamera()
}

function startAnalysis() {
  if (analyzeTimer.value) clearInterval(analyzeTimer.value)
  analyzeTimer.value = setInterval(analyzeFrame, props.analyzeInterval)
}

function stopAnalysis() {
  if (analyzeTimer.value) {
    clearInterval(analyzeTimer.value)
    analyzeTimer.value = null
  }
}

function computeCenteredCrop(sw, sh, targetRatio) {
  const srcRatio = sw / sh
  let sx = 0, sy = 0, sWidth = sw, sHeight = sh
  if (srcRatio > targetRatio) {
    sWidth = Math.round(sh * targetRatio)
    sx = Math.round((sw - sWidth) / 2)
  } else if (srcRatio < targetRatio) {
    sHeight = Math.round(sw / targetRatio)
    sy = Math.round((sh - sHeight) / 2)
  }
  return { sx, sy, sWidth, sHeight }
}

function checkPersonInFrame(landmarks, cw, ch) {
  if (!landmarks?.length) return false
  
  const frameLeft = cw * 0.2
  const frameRight = cw * 0.8
  const frameTop = ch * 0.15
  const frameBottom = ch * 0.85
  
  const keyPoints = [0, 11, 12, 23, 24]
  let inFrameCount = 0
  
  for (const idx of keyPoints) {
    const p = landmarks[idx]
    if (p?.visibility > 0.5) {
      const x = mirror.value ? cw - p.x * cw : p.x * cw
      const y = p.y * ch
      if (x >= frameLeft && x <= frameRight && y >= frameTop && y <= frameBottom) {
        inFrameCount++
      }
    }
  }
  
  return inFrameCount >= 3 
}

async function analyzeFrame() {
  const v = videoElement.value
  const canvas = canvasElement.value
  if (!v || !canvas || !isStreaming.value) return
  const ctx = canvas.getContext('2d')

  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2))
  const cw = canvas.width / dpr
  const ch = canvas.height / dpr

  const sw = v.videoWidth
  const sh = v.videoHeight
  if (!sw || !sh) return

  const { sx, sy, sWidth, sHeight } = computeCenteredCrop(sw, sh, targetRatio.value)

  ctx.clearRect(0, 0, cw, ch)
  ctx.save()
  if (mirror.value) {
    ctx.translate(cw, 0)
    ctx.scale(-1, 1)
  }
  ctx.drawImage(v, sx, sy, sWidth, sHeight, 0, 0, cw, ch)
  ctx.restore()

  try {
    const imageData = canvas.toDataURL('image/jpeg', 0.8)
    const result = await mediapipeStore.analyzeFrame(imageData)

    emit('frame-analyzed', result)
    
    if (result?.pose_landmarks) {
      drawPose(ctx, result.pose_landmarks, cw, ch)
      
      if (props.startupMode) {
        const isInFrame = checkPersonInFrame(result.pose_landmarks, cw, ch)
        
        if (isInFrame) {
          personDetectedFrames++
          personLostFrames = 0
          
          if (personDetectedFrames >= DETECTION_THRESHOLD && !personInFrame.value) {
            personInFrame.value = true
            emit('person-detected')
            console.log('[WebcamAnalyzer] ✅ 人体进入框内')
          }
        } else {
          personLostFrames++
          personDetectedFrames = 0
          
          if (personLostFrames >= LOST_THRESHOLD && personInFrame.value) {
            personInFrame.value = false
            emit('person-lost')
            console.log('[WebcamAnalyzer] ⚠️ 人体离开框内')
          }
        }
      }
    } else {
      if (props.startupMode) {
        personLostFrames++
        personDetectedFrames = 0
        
        if (personLostFrames >= LOST_THRESHOLD && personInFrame.value) {
          personInFrame.value = false
          emit('person-lost')
          console.log('[WebcamAnalyzer] ⚠️ 人体丢失')
        }
      }
    }
  } catch {
    // avoiding crash on analysis error
  }
}

function drawPose(ctx, landmarks, cw, ch) {
  if (!landmarks?.length) return
  ctx.strokeStyle = '#00ff00'
  ctx.lineWidth = 2
  ctx.fillStyle = '#ff0000'

  for (const p of landmarks) {
    if (p.visibility > 0.5) {
      const x = mirror.value ? cw - p.x * cw : p.x * cw
      const y = p.y * ch
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  
  const connections = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28]
  ]
  ctx.beginPath()
  for (const [a,b] of connections) {
    if (landmarks[a]?.visibility > 0.5 && landmarks[b]?.visibility > 0.5) {
      const x1 = mirror.value ? cw - landmarks[a].x * cw : landmarks[a].x * cw
      const y1 = landmarks[a].y * ch
      const x2 = mirror.value ? cw - landmarks[b].x * cw : landmarks[b].x * cw
      const y2 = landmarks[b].y * ch
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
    }
  }
  ctx.stroke()
}

defineExpose({ startAnalysis, stopAnalysis })

onMounted(() => {
  attachResizeObserver()
  startCamera()
})

watch(() => props.orientation, async () => {
  stopAnalysis()
  stopCamera()
  await startCamera()
})

watch(() => props.startupMode, (newVal) => {
  if (!newVal) {
    personInFrame.value = false
    personDetectedFrames = 0
    personLostFrames = 0
  }
})

onUnmounted(() => {
  stopCamera()
  if (ro) ro.disconnect()
})
</script>

<style scoped>
.webcam-analyzer {
  --l-video-height: 56vh;
  --l-max-w: 980px;
  --p-video-height: 60vh;
  --p-scale: 1.0;
  --p-max-w: 640px;

  width: 100%;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.video-frame {
  width: 100%;
  display: flex;
}

.video-container {
  position: relative;
  background: #000;
  overflow: hidden;
  height: auto;
  margin-inline: auto;
  width: 100%;
}

.o-landscape .video-container { aspect-ratio: 16 / 9; }
.o-portrait  .video-container { aspect-ratio: 9 / 16; }

.o-landscape .video-container {
  width: min(100%, calc(var(--l-video-height) * 16 / 9), var(--l-max-w));
}

.o-portrait .video-container {
  width: min(100%, calc(var(--p-scale) * var(--p-video-height) * 9 / 16), var(--p-max-w));
}

.video-feed { display: none !important; }

.pose-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.detection-frame-container {
  position: absolute;
  left: 20%;
  top: 15%;
  width: 60%;
  height: 70%;
  border: 5px solid #ff5e62; 
  border-radius: 12px;
  pointer-events: none;
}

.o-landscape .detection-frame-container {
  left: 15%;
  top: 8%;
  width: 70%;
  height: 84%;
}

.o-portrait .detection-frame-container {
  left: 10%;
  top: 5%;
  width: 80%;
  height: 90%;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.85;
    transform: scale(1.02);
  }
}

.detection-frame-text {
  font-weight: 700;
  font-size: 1.8rem;
  color: #ff5e62;
  text-shadow: 
    0 2px 4px rgba(0, 0, 0, 0.3),
    0 0 10px rgba(255, 94, 98, 0.5);
  animation: pulse 2s ease-in-out infinite;
  text-align: center;
  letter-spacing: 0.5px;
  white-space: nowrap;
  flex-shrink: 0;
  margin-bottom: 0.5rem;
  order: -1; 
}



@media (max-width: 1024px) {
  .webcam-analyzer { --l-video-height: 50vh; --p-video-height: 48vh; --p-scale: 1.05; }
  .detection-frame {
    border-width: 4px;
  }
}

@media (max-width: 640px) {
  .webcam-analyzer { --l-video-height: 44vh; --p-video-height: 42vh; --p-scale: 1.0; }
  .detection-frame {
    border-width: 3px;
  }
}

.video-placeholder {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  background: #f5f5f5;
}

.placeholder-content { text-align: center; }
.placeholder-content p { font-size: 18px; margin-bottom: 14px; color: #666; }

.start-button {
  padding: 10px 22px; background: #2f6fed; color: #fff; border: 0; border-radius: 8px; cursor: pointer;
}

.error-message {
  position: absolute; left: 20px; right: 20px; top: 20px;
  background: #ff4757; color: #fff; padding: 10px 15px; border-radius: 8px; font-size: 14px;
}

.controls { display: flex; gap: 10px; padding: 12px; background: #f8f9fa; }

.control-button {
  flex: 1; padding: 10px 20px; border: 1px solid #ddd; background: white; border-radius: 6px; font-size: 14px; cursor: pointer;
}

.control-button:hover { background: #f0f0f0; border-color: #2f6fed; color: #2f6fed; }
</style>