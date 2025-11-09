<template>
  <div
    v-if="showIntro && style === 'beginner' && exerciseData"
    class="intro-modal-overlay"
    @click.self="closeIntroAndStart"
  >
    <div class="intro-modal-content">
      <button @click="closeIntroAndStart" class="close-button">&times;</button>
      <div class="modal-body">
        <div class="image-container">
          <img :src="exerciseData.imageUrl" :alt="exerciseData.name" class="intro-image">
        </div>
        <div class="info-container">
          <h2>{{ exerciseData.name }}</h2>
          <p>{{ exerciseData.description }}</p>
          <h3 class="tips-title">動作要點</h3>
          <ul class="tips-list">
            <li v-for="tip in exerciseData.tips" :key="tip">{{ tip }}</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <div class="exercise-view" v-if="exerciseData">
    <div class="header">
      <router-link to="/" class="back-button">← Back</router-link>
      <h1>{{ exerciseData?.title || 'Analysis' }}</h1>
    </div>

    <div class="content">
      <div class="main-section">
        <div v-if="showStartupGuide" class="startup-guide-overlay" @click="showStartupGuide = false">
        <div class="startup-guide-content">
          <div class="startup-icon">
            {{ personDetected ? (thumbsUpDetected ? '🚀' : '👍') : '🧍' }}
          </div>
          <h2 class="startup-title">{{ startupTitle }}</h2>
          <p class="startup-instruction">{{ startupInstruction }}</p>
        </div>
      </div>
        <WebcamAnalyzer
          v-if="isWorkoutActive"
          ref="analyzer"
          :analyze-interval="50"
          :orientation="orientation"
          :startup-mode="showStartupGuide"
          @person-detected="handlePersonDetected"
          @person-lost="handlePersonLost"
          @frame-analyzed="handleFrameAnalyzed"
          @error="handleError"
        />
        <div v-else class="webcam-placeholder">
          <p>{{ style === 'motivator' ? 'Standby...' : 'Loading...' }}</p>
        </div>
      </div>

      <div class="stats-section">
        <div class="feedback-container" 
        :class="{ 
          'is-error': isOverextended,
          'is-paused': mediapipeStore.isPaused 
           }">
          <p class="feedback-text">{{ mediapipeStore.isPaused ? mediapipeStore.gestureMessage :feedbackText }}</p>
          <p v-if="mediapipeStore.lastGesture" class="gesture-indicator">
            检测到: {{ gestureEmoji(mediapipeStore.lastGesture) }}
          </p>
        </div>

        <section class="panel">
          <div class="section-title">動作示意</div>
          <div class="hint">當前訓練：{{ exerciseData.name }}</div>
          <div class="motion-box">
            <img
              v-if="exerciseData.motionUrl || exerciseData.imageUrl"
              :src="exerciseData.motionUrl || exerciseData.imageUrl"
              :alt="exerciseData.name"
              class="motion-img"
            />
            <div v-else class="motion-placeholder">請為該動作提供動圖/圖片</div>
          </div>
        </section>

        <section class="panel">
          <div class="progress-block">
            <div class="progress-label">
              <span>進度</span>
              <span class="progress-percent">{{ progressPercent }}%</span>
            </div>
            <div
              class="progress-bar"
              role="progressbar"
              :aria-valuenow="progressPercent"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              <div
                class="progress-fill"
                :class="{ done: progressPercent === 100 }"
                :style="{ width: progressPercent + '%' }"
              ></div>
            </div>
            <div class="progress-ratio">{{ mediapipeStore.count }}/{{ MAX_REPS }}</div>
            <div class="progress-actions"></div>
          </div>
        </section>
      </div>
    </div>

    <WorkoutSummary
      v-if="showSummary"
      :exercise-type="exerciseType"
      @continue="handleContinueWorkout"
      @end="handleEndWorkout"
    />
  </div>

  <div v-else>
    <h1>加载中或运动类型无效...</h1>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, computed, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'
import WebcamAnalyzer from '@/components/WebcamAnalyzer.vue'
import WorkoutSummary from '@/components/WorkoutSummary.vue'
import confetti from 'canvas-confetti'

const route = useRoute()
const router = useRouter()
const exerciseType = ref(route.params.type)
const style = ref(route.query.style)

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()
const analyzer = ref(null)
const currentAngles = ref(null)
const showSummary = ref(false)
const showIntro = ref(false)
const isWorkoutActive = ref(false)
const showStartupGuide = ref(false)
const exerciseData = computed(() => exerciseStore.getExerciseById(exerciseType.value))

const personDetected = ref(false)
const thumbsUpDetected = ref(false)

const feedbackText = ref('隨時準備！')
const isOverextended = ref(false)

// ✅ 修复：确保使用 .value 访问 ref
const MAX_REPS = computed(() => mediapipeStore.repetitionLimit || 15)

const orientation = computed(() => exerciseData.value?.orientation || 'landscape')

// ✅ 修复：确保进度条响应式更新
const progressPercent = computed(() => {
  const total = Number(MAX_REPS.value || 0)
  const done = Number(mediapipeStore.count || 0)
  if (!total) return 0
  const p = Math.round((done / total) * 100)
  console.log('[progressPercent] total:', total, 'done:', done, 'percent:', p)
  return Math.max(0, Math.min(100, p))
})

function gestureEmoji(gesture) {
  const emojiMap = {
    'like': '👍',
    'stop': '✋',
  }
  return emojiMap[gesture] || gesture
}

async function startWorkoutFlow() {
  console.log('[view] startWorkoutFlow route.style =', style.value)
  // 先设置 store（可先把它提前于 analyzer）
  await mediapipeStore.startExercise(exerciseType.value, String(style.value).toLowerCase())
  // 再启动本地流程
  isWorkoutActive.value = true
  showStartupGuide.value = true
  exerciseStore.startExercise()
  await nextTick()
  setTimeout(() => analyzer.value?.startAnalysis(), 100)
}

const startupTitle = computed(() => {
  if (!personDetected.value) {
    return '请站入橙色框内'
  } else if (!thumbsUpDetected.value) {
    return '准备开始'
  } else {
    return '正在启动...'
  }
})

const startupInstruction = computed(() => {
  if (!personDetected.value) {
    return '请调整位置，确保身体完整出现在橙色检测框内'
  } else if (!thumbsUpDetected.value) {
    return '做出 👍 手势以开始训练'
  } else {
    return '即将开始您的训练'
  }
})

function handlePersonDetected() {
  console.log('[ExerciseView] 检测到人体进入框内')
  personDetected.value = true
}

function handlePersonLost() {
  console.log('[ExerciseView] 人体离开框内')
  personDetected.value = false
  thumbsUpDetected.value = false
}

watch(() => mediapipeStore.gestureDetected, (gesture) => {
  if (showStartupGuide.value && personDetected.value && gesture === 'thumbs_up') {
    console.log('[ExerciseView] 检测到👍手势，准备启动训练')
    thumbsUpDetected.value = true
    
    setTimeout(() => {
      showStartupGuide.value = false
      personDetected.value = false
      thumbsUpDetected.value = false
      console.log('[ExerciseView] 启动引导结束，开始正式训练')
    }, 800)
  }
})

function closeIntroAndStart() {
  if (!showIntro.value) return
  showIntro.value = false
  startWorkoutFlow()
}

onMounted(() => {
  if (!exerciseType.value || !style.value) {
    router.push('/')
    return
  }
  exerciseStore.selectExercise(exerciseType.value)

  if (style.value === 'motivator') {
    showIntro.value = true
  } else {
    startWorkoutFlow()
  }
})

function handleFrameAnalyzed(result) {
  console.log('[handleFrameAnalyzed] received result keys:', Object.keys(result || {}))
  
  const byCurrent = result?.current
  const byType = result?.[exerciseType.value]
  const byName = exerciseData.value?.name ? result?.[exerciseData.value.name] : undefined
  const analysisData = byCurrent || byType || byName

  console.log('[handleFrameAnalyzed] selected data source:', 
    byCurrent ? 'current' : (byType ? 'type' : (byName ? 'name' : 'none')))
  console.log('[handleFrameAnalyzed] analysisData:', analysisData)

  if (analysisData && typeof analysisData === 'object') {
    if (analysisData.shoulder_angle !== undefined || analysisData.elbow_angle !== undefined) {
      currentAngles.value = {
        shoulder: analysisData.shoulder_angle,
        elbow: analysisData.elbow_angle
      }
    }

    mediapipeStore.updateAnalysisData(analysisData)

    if (!mediapipeStore.isPaused) {
      feedbackText.value = analysisData.feedback || feedbackText.value
      const nonStandard = analysisData.category === 'non_standard'
      isOverextended.value = Boolean(analysisData.overextended || nonStandard)
    }

    console.log('[handleFrameAnalyzed] after update - store.count:', mediapipeStore.count)
  } else {
    console.warn(
      '[handleFrameAnalyzed] 未命中 current/type/name。可用键:',
      Object.keys(result || {})
    )
  }

  if (result.gesture_detected) {
    console.log(`检测到手势: ${result.gesture_detected}`)
  }
}

function handleError(error) {
  console.error('Analysis error:', error)
}

function blastConfetti(options = {}) {
  // 全屏、在下一帧触发，避免与 UI 切换抢帧
  requestAnimationFrame(() => {
    const duration = options.duration ?? 2000 // ms
    const end = Date.now() + duration

    // 统一的发射参数
    const base = {
      particleCount: 50,
      spread: 60,
      startVelocity: 45,
      gravity: 0.9,
      ticks: 250,
      origin: { y: 0.6 }
    }

    // 连续喷射一段时间
    const interval = setInterval(() => {
      confetti({ ...base, angle: 60, origin: { x: 0, y: Math.random() * 0.3 + 0.1 } })
      confetti({ ...base, angle: 120, origin: { x: 1, y: Math.random() * 0.3 + 0.1 } })
      // 中央一波
      confetti({
        ...base,
        particleCount: 80,
        spread: 90,
        origin: { x: 0.5, y: 0.3 },
        scalar: 1.1,
        colors: ['#34d399', '#3b82f6', '#f59e0b', '#ef4444', '#a78bfa']
      })
      if (Date.now() > end) clearInterval(interval)
    }, 250)
  })
}

watch(
  () => mediapipeStore.count,
  (newCount) => {
    const limit = Number(mediapipeStore.repetitionLimit || 15) 
    console.log('[watch count] newCount:', newCount, 'limit:', limit)
    if (newCount >= limit && limit > 0) {
      console.log('[watch count] 🎉 达到目标次数，显示总结！')
      // 先停掉分析，避免继续加数
      try { analyzer.value?.stopAnalysis?.() } catch (e) { console.warn(e) }
      blastConfetti({ duration: 2500 })
      isWorkoutActive.value = false
      exerciseStore.endExercise()
      showSummary.value = true
      mediapipeStore.pause?.() // 如果有 pause 动作的话
    }
  }
)

async function handleContinueWorkout() {
  showSummary.value = false
  mediapipeStore.reset()
  await mediapipeStore.startExercise(exerciseType.value, String(style.value).toLowerCase())
  exerciseStore.startExercise()
  isWorkoutActive.value = true
  await nextTick()
  analyzer.value?.startAnalysis()
}

function handleEndWorkout() {
  showSummary.value = false
  mediapipeStore.reset()
  router.push('/')
}
</script>

<style scoped>
/* New */
.gesture-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: 12px;
  animation: fadeIn 0.3s ease;
}
.gesture-prompt {
  text-align: center;
  color: white;
  animation: pulse 2s ease-in-out infinite;
}
.gesture-text {
  font-size: 1.8rem;
  font-weight: 600;
  margin-bottom: 10px;
}

.gesture-hint {
  font-size: 1.2rem;
  opacity: 0.8;
}

.gesture-indicator {
  margin-top: 8px;
  font-size: 0.9rem;
  opacity: 0.7;
  animation: fadeIn 0.3s ease;
}
.intro-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  animation: fadeIn 0.3s ease;
}

.intro-modal-content {
  position: relative;
  background: white;
  border-radius: 16px;
  padding: 40px;
  max-width: 800px;
  width: 90%;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  animation: slideIn 0.4s ease-out;
}

.close-button {
  position: absolute;
  top: 15px;
  right: 15px;
  background: none;
  border: none;
  font-size: 2rem;
  line-height: 1;
  color: #aaa;
  cursor: pointer;
  transition: color 0.2s, transform 0.2s;
}
.close-button:hover {
  color: #333;
  transform: rotate(90deg);
}

.modal-body {
  display: flex;
  gap: 40px;
  align-items: center;
}

.image-container {
  flex-basis: 50%;
  flex-shrink: 0;
}

.intro-image {
  width: 100%;
  border-radius: 10px;
  background-color: #f0f0f0;
  display: block;
}

.info-container {
  flex-basis: 50%;
}

.info-container h2 {
  margin-top: 0;
  font-size: 2rem;
  color: #333;
}

.info-container p {
  font-size: 1rem;
  color: #666;
  margin-bottom: 24px;
}

.tips-title {
  font-size: 1.1rem;
  color: #444;
  margin-bottom: 12px;
  border-left: 3px solid #667eea;
  padding-left: 10px;
}

.tips-list {
  padding-left: 20px;
  margin: 0;
}

.tips-list li {
  margin-bottom: 8px;
  color: #555;
}

/* --- 响应式设计：在窄屏幕上垂直排列 --- */
@media (max-width: 768px) {
  .modal-body {
    flex-direction: column;
    gap: 20px;
  }
  .intro-modal-content {
    padding: 30px;
  }
  .info-container h2 {
    font-size: 1.5rem;
  }
}

.exercise-view { max-width: 1400px; margin: 0 auto; padding: 20px; }
.header { display: flex; align-items: center; gap: 20px; margin-bottom: 30px; }
.back-button { text-decoration: none; color: #667eea; font-size: 16px; }
.header h1 { margin: 0; color: #333; }
.content { display: grid; grid-template-columns: 1fr 350px; gap: 30px; }
.main-section, .stats-section { display: flex; flex-direction: column; gap: 20px; }

.camera-shell { width: 100%; }
.video-frame {
  position: relative;
  width: 100%;
  overflow: hidden;
  background: #000;
  border-radius: 12px;
}
.o-landscape .video-frame { aspect-ratio: 16 / 9; }
.o-portrait  .video-frame { aspect-ratio: 9 / 16; }

.video-frame video,
.video-frame canvas,
.video-frame .webcam-placeholder {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.feedback-container {
  background-color: #2d3748; /* 深灰色背景 */
  color: #edf2f7; /* 浅灰色文字 */
  padding: 16px 24px;
  border-radius: 12px;
  text-align: center;
  margin-top: 20px; /* 与上方视频区保持间距 */
  border: 2px solid transparent; /* 预留边框位置 */
  transition: background-color 0.3s ease, border-color 0.3s ease;
}
.feedback-text {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
  transition: color 0.3s ease;
}

/* 当 overextended 为 true 时的错误状态样式 */
.feedback-container.is-error {
  background-color: #451a1a; /* 暗红色背景 */
  border-color: #e53e3e; /* 鲜红色边框 */
  animation: shake 0.6s cubic-bezier(.36,.07,.19,.97) both;
}
.feedback-container.is-error .feedback-text {
  color: #fed7d7; /* 浅红色文字 */
  font-weight: 700;
}

/* Paused Container */
.feedback-container.is-paused {
  background-color: #2c5282;
  border-color: #4299e1;
}

.feedback-container.is-paused .feedback-text {
  color: #bee3f8;
}

.startup-guide-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(103, 58, 183, 0.75);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  animation: fadeIn 0.3s ease-out;
}

.startup-guide-content {
  text-align: center;
  color: white;
  max-width: 500px;
  padding: 40px 30px;
}

.startup-icon {
  font-size: 80px;
  margin-bottom: 24px;
  animation: bounce 1s ease-in-out infinite;
}

.startup-title {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 16px;
  text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.startup-instruction {
  font-size: 16px;
  line-height: 1.6;
  opacity: 0.95;
  text-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* 动画 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes shake {
  10%, 90% { transform: translate3d(-1px, 0, 0); }
  20%, 80% { transform: translate3d(2px, 0, 0); }
  30%, 50%, 70% { transform: translate3d(-3px, 0, 0); }
  40%, 60% { transform: translate3d(3px, 0, 0); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}
@media (max-width: 1024px) { .content { grid-template-columns: 1fr; } }
@media (max-width: 768px) {
  .modal-body { flex-direction: column; gap: 20px; }
  .intro-modal-content { padding: 30px; }
  .info-container h2 { font-size: 1.5rem; }
  .feedback-text { font-size: 1.2rem; }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

@media (max-width: 768px) {
  .startup-guide-content {
    padding: 30px 20px;
  }
  
  .startup-icon {
    font-size: 60px;
    margin-bottom: 20px;
  }
  
  .startup-title {
    font-size: 24px;
    margin-bottom: 12px;
  }
  
  .startup-instruction {
    font-size: 14px;
  }
}

.image-container {
flex-basis: 50%;
flex-shrink: 0;
}
.image-container {
display: flex;
align-items: center;
justify-content: center;

height: 280px; 
aspect-ratio: 16 / 9; 
background: #fff;
border-radius: 10px;
overflow: hidden;
}
.intro-image {
width: 100%;
height: 100%;
object-fit: contain; 
object-position: center;
display: block;
}

.motion-box {
width: 100%;
aspect-ratio: 16 / 9;
background: #fff;
border-radius: 12px;
border: 1px solid #e5e7eb;
overflow: hidden;
display: flex;
align-items: center;
justify-content: center;
}
.motion-img {
width: 100%;
height: 100%;
object-fit: contain;
object-position: center;
}
.motion-placeholder {
color: #9ca3af;
font-size: 40px;
}

.panel .section-title {
font-size: 40px; 
font-weight: 700; 
color: #111827;
}

.panel .hint {
font-size: 40px;
font-weight: 600;
color: #1f2937;
margin-top: 4px;
}

.progress-block {
margin-top: 10px;
}
.progress-label {
display: flex;
justify-content: space-between;
align-items: center;
font-size: 40px;
color: #374151;
margin-bottom: 6px;
}
.progress-percent {
font-size: 40px;
font-variant-numeric: tabular-nums;
color: #111827;
font-weight: 700;
}
.progress-bar {
width: 100%;
height: 30px;
background: #e5e7eb;
border-radius: 999px;
overflow: hidden;
}
.progress-fill {
height: 100%;
width: 0%;
background: linear-gradient(90deg, #60a5fa, #3b82f6);
border-radius: 999px;
transition: width 300ms ease;
}
.progress-fill.done {
background: linear-gradient(90deg, #34d399, #10b981);
}
.progress-ratio {
margin-top: 8px;
font-size: 40px;
color: #4b5563;
}
.progress-actions {
display: flex;
flex-direction: column;
gap: 8px;
margin-top: 10px;
}
.link-btn {
background: none;
border: none;
padding: 0;
color: #6b7280;
font-size: 12px;
cursor: pointer;
}
.link-btn:hover {
color: #111827;
text-decoration: underline;
}

/* 🆕 新增样式 - 启动引导遮罩 */
.startup-guide-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.95), rgba(118, 75, 162, 0.95));
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 15;
  border-radius: 12px;
  animation: fadeIn 0.5s ease;
}

.startup-guide-content {
  text-align: center;
  color: white;
  padding: 40px;
  max-width: 500px;
}

.startup-icon {
  font-size: 6rem;
  margin-bottom: 20px;
  animation: bounce 2s ease-in-out infinite;
}

.startup-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 20px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.startup-instruction {
  font-size: 1.3rem;
  margin-bottom: 12px;
  opacity: 0.95;
  line-height: 1.6;
}

/* 🆕 响应式 - 启动引导 */
@media (max-width: 768px) {
  .startup-icon {
    font-size: 4rem;
  }
  
  .startup-title {
    font-size: 2rem;
  }
  
  .startup-instruction {
    font-size: 1.1rem;
  }
  
  .startup-guide-content {
    padding: 30px 20px;
  }
}
</style>