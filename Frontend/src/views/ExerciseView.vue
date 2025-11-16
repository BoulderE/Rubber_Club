<template>
  <div
    v-if="showIntro && style === 'motivator' && exerciseData"
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
      <router-link to="/" class="back-button">← 返回</router-link>
      <h1>{{ exerciseData?.title || '訓練分析' }}</h1>
    </div>

    <div class="content">
      <div class="main-section">
        <div class="feedback-container" 
        :class="{ 
          'is-error': isOverextended,
          'is-paused': mediapipeStore.isPaused,
          'is-ready': showReadyMessage
           }">
          <p class="feedback-text">{{ displayFeedback }}</p>
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
          <p>載入中...</p>
        </div>
      </div>

      <div class="stats-section">
        
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
    <h1>載入中或運動類型無效...</h1>
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

// 🔥 新增：控制準備好訊息的顯示
const showReadyMessage = ref(false)
const readyMessageTimer = ref(null)

const MAX_REPS = computed(() => mediapipeStore.repetitionLimit || 15)
const orientation = computed(() => exerciseData.value?.orientation || 'landscape')

const progressPercent = computed(() => {
  const total = Number(MAX_REPS.value || 0)
  const done = Number(mediapipeStore.count || 0)
  if (!total) return 0
  const p = Math.round((done / total) * 100)
  console.log('[progressPercent] total:', total, 'done:', done, 'percent:', p)
  return Math.max(0, Math.min(100, p))
})

// 新增：計算顯示的反饋文字
const displayFeedback = computed(() => {
  // 如果處於暫停狀態，優先顯示手勢訊息
  if (mediapipeStore.isPaused) {
    return mediapipeStore.gestureMessage
  }
  
  // 如果顯示準備好訊息
  if (showReadyMessage.value) {
    return '你已準備好，隨時開始運動'
  }
  
  // 否則顯示正常的反饋
  return feedbackText.value
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
  await mediapipeStore.startExercise(exerciseType.value, String(style.value).toLowerCase())
  isWorkoutActive.value = true
  showStartupGuide.value = true
  exerciseStore.startExercise()
  await nextTick()
  setTimeout(() => analyzer.value?.startAnalysis(), 100)
}

const startupTitle = computed(() => {
  if (!personDetected.value) {
    return '請站入橙色框內'
  } else if (!thumbsUpDetected.value) {
    return '準備開始'
  } else {
    return '正在啟動...'
  }
})

const startupInstruction = computed(() => {
  if (!personDetected.value) {
    return '請調整位置，確保身體完整出現在橙色檢測框內'
  } else if (!thumbsUpDetected.value) {
    return '做出👍手勢以開始訓練'
  } else {
    return '即將開始你的訓練'
  }
})

function handlePersonDetected() {
  console.log('[ExerciseView] 檢測到人體進入框內')
  personDetected.value = true
}

function handlePersonLost() {
  console.log('[ExerciseView] 人體離開框內')
  personDetected.value = false
  thumbsUpDetected.value = false
}

// 修改：監聽手勢檢測
watch(() => mediapipeStore.gestureDetected, (gesture) => {
  if (showStartupGuide.value && personDetected.value && gesture === 'thumbs_up') {
    console.log('[ExerciseView] 檢測到👍手勢，準備啟動訓練')
    thumbsUpDetected.value = true
    
    setTimeout(() => {
      showStartupGuide.value = false
      personDetected.value = false
      thumbsUpDetected.value = false
      console.log('[ExerciseView] 啟動引導結束，開始正式訓練')
      
      // 新增：顯示準備好訊息 3 秒
      showReadyMessage.value = true
      if (readyMessageTimer.value) {
        clearTimeout(readyMessageTimer.value)
      }
      readyMessageTimer.value = setTimeout(() => {
        showReadyMessage.value = false
      }, 3000)
    }, 800)
  }
})

// 新增：監聽 count 變化，當開始運動時隱藏準備好訊息
watch(() => mediapipeStore.count, (newCount) => {
  // 當第一次計數時（從 0 變為 1），立即隱藏準備好訊息
  if (newCount > 0 && showReadyMessage.value) {
    showReadyMessage.value = false
    if (readyMessageTimer.value) {
      clearTimeout(readyMessageTimer.value)
      readyMessageTimer.value = null
    }
  }
  
  // 原有的完成檢測邏輯
  const limit = Number(mediapipeStore.repetitionLimit || 15) 
  console.log('[watch count] newCount:', newCount, 'limit:', limit)
  if (newCount >= limit && limit > 0) {
    console.log('[watch count] 🎉 達到目標次數，顯示總結！')
    try { analyzer.value?.stopAnalysis?.() } catch (e) { console.warn(e) }
    blastConfetti({ duration: 2500 })
    isWorkoutActive.value = false
    exerciseStore.endExercise()
    showSummary.value = true
    mediapipeStore.pause?.()
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

    // 🔥 修改：只在不顯示準備好訊息時更新反饋
    if (!mediapipeStore.isPaused && !showReadyMessage.value) {
      feedbackText.value = analysisData.feedback || feedbackText.value
      const nonStandard = analysisData.category === 'non_standard'
      isOverextended.value = Boolean(analysisData.overextended || nonStandard)
    }

    console.log('[handleFrameAnalyzed] after update - store.count:', mediapipeStore.count)
  } else {
    console.warn(
      '[handleFrameAnalyzed] 未命中 current/type/name。可用鍵:',
      Object.keys(result || {})
    )
  }

  if (result.gesture_detected) {
    console.log(`檢測到手勢: ${result.gesture_detected}`)
  }
}

function handleError(error) {
  console.error('分析錯誤:', error)
}

function blastConfetti(options = {}) {
  requestAnimationFrame(() => {
    const duration = options.duration ?? 2000
    const end = Date.now() + duration

    const base = {
      particleCount: 50,
      spread: 60,
      startVelocity: 45,
      gravity: 0.9,
      ticks: 250,
      origin: { y: 0.6 }
    }

    const interval = setInterval(() => {
      confetti({ ...base, angle: 60, origin: { x: 0, y: Math.random() * 0.3 + 0.1 } })
      confetti({ ...base, angle: 120, origin: { x: 1, y: Math.random() * 0.3 + 0.1 } })
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

async function handleContinueWorkout() {
  showSummary.value = false
  mediapipeStore.reset()
  
  // 🔥 重置準備好訊息狀態
  showReadyMessage.value = false
  if (readyMessageTimer.value) {
    clearTimeout(readyMessageTimer.value)
    readyMessageTimer.value = null
  }
  
  await mediapipeStore.startExercise(exerciseType.value, String(style.value).toLowerCase())
  exerciseStore.startExercise()
  isWorkoutActive.value = true
  await nextTick()
  analyzer.value?.startAnalysis()
}

function handleEndWorkout() {
  showSummary.value = false
  mediapipeStore.reset()
  
  // 清理準備好訊息狀態
  showReadyMessage.value = false
  if (readyMessageTimer.value) {
    clearTimeout(readyMessageTimer.value)
    readyMessageTimer.value = null
  }
  
  router.push('/')
}
</script>

<style scoped>
/* ==================== 啟動提示框 ==================== */
.startup-hint {
  background: linear-gradient(135deg, #ff9966, #ff5e62);
  border-radius: 12px;
  padding: 20px 30px;
  text-align: center;
  box-shadow: 0 4px 15px rgba(255, 94, 98, 0.3);
  margin-bottom: 20px;
  animation: fadeInDown 0.5s ease;
  border: 5px solid #ff5722;
}

.startup-hint-title {
  font-size: 40px;
  font-weight: 700;
  color: white;
  margin-bottom: 8px;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.startup-hint-text {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.95);
  line-height: 1.5;
}

/* ==================== 引導彈窗 ==================== */
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
  display: flex;
  align-items: center;
  justify-content: center;
  height: 280px;
  aspect-ratio: 4 / 3;
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

/* ==================== 主布局 ==================== */
.exercise-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 30px;
}

.back-button {
  text-decoration: none;
  color: #667eea;
  font-size: 16px;
}

.header h1 {
  margin: 0;
  color: #333;
}

.content {
  display: grid;
  grid-template-columns: 650px 1fr;
  gap: 30px;
}

.main-section,
.stats-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ==================== 視頻區域 ==================== */
.webcam-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  border-radius: 12px;
  aspect-ratio: 16 / 9;
  color: white;
  font-size: 1.2rem;
}

/* ==================== 反饋容器 ==================== */
.feedback-container {
  background-color: #2d3748;
  color: #edf2f7;
  padding: 16px 24px;
  border-radius: 12px;
  text-align: center;
  border: 2px solid transparent;
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.feedback-text {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
  transition: color 0.3s ease;
}

.feedback-container.is-error {
  background-color: #451a1a;
  border-color: #e53e3e;
  animation: shake 0.6s cubic-bezier(.36,.07,.19,.97) both;
}

.feedback-container.is-error .feedback-text {
  color: #fed7d7;
  font-weight: 700;
}

.feedback-container.is-paused {
  background-color: #2c5282;
  border-color: #4299e1;
}

.feedback-container.is-paused .feedback-text {
  color: #bee3f8;
}

/* 🔥 新增：準備好狀態的樣式 */
.feedback-container.is-ready {
  background: linear-gradient(135deg, #10b981, #34d399);
  border-color: #34d399;
  animation: pulse 1.5s ease-in-out infinite;
}

.feedback-container.is-ready .feedback-text {
  color: white;
  font-weight: 700;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.gesture-indicator {
  margin-top: 8px;
  font-size: 0.9rem;
  opacity: 0.7;
  animation: fadeIn 0.3s ease;
}

/* ==================== 動作示意面板 ==================== */
.panel {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.panel .section-title {
  font-size: 40px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 8px;
}

.panel .hint {
  font-size: 40px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
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
  min-height: 400px;
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

/* ==================== 進度條 ==================== */
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

/* ==================== 動畫 ==================== */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideIn {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes shake {
  10%, 90% { transform: translate3d(-1px, 0, 0); }
  20%, 80% { transform: translate3d(2px, 0, 0); }
  30%, 50%, 70% { transform: translate3d(-3px, 0, 0); }
  40%, 60% { transform: translate3d(3px, 0, 0); }
}

@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

/* 新增：脈衝動畫 */
@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
  }
}

/* ==================== 響應式設計 ==================== */
@media (max-width: 1024px) {
  .content {
    grid-template-columns: 1fr;
  }
}

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
  
  .startup-hint {
    padding: 16px 20px;
  }
  
  .startup-hint-title {
    font-size: 28px;
  }
  
  .startup-hint-text {
    font-size: 1rem;
  }
  
  .feedback-text {
    font-size: 1.2rem;
  }
  
  .panel .section-title,
  .panel .hint,
  .progress-label,
  .progress-percent,
  .progress-ratio,
  .motion-placeholder {
    font-size: 24px;
  }
}
</style>