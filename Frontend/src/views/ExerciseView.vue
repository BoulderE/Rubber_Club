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
      <router-link to="/" class="back-button">← Back</router-link>
      <h1>{{ exerciseData?.title || 'Analysis' }}</h1>
    </div>
    
    <div class="content">
      <div class="main-section">
        <WebcamAnalyzer 
          v-if="isWorkoutActive"
          ref="analyzer"
          :analyze-interval="50"
          @frame-analyzed="handleFrameAnalyzed"
          @error="handleError"
        />
        <div v-else class="webcam-placeholder">
          <p>{{ style === 'motivator' ? 'Standby...' : 'Loading...' }}</p>
        </div>
      </div>
      
      <div class="stats-section">
        <!-- Motion preview -->
         <div class="feedback-container" :class="{ 'is-error': isOverextended }">
          <p class="feedback-text">{{ feedbackText }}</p>
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
            <div class="progress-bar" role="progressbar" :aria-valuenow="progressPercent" aria-valuemin="0" aria-valuemax="100">
            <div class="progress-fill" :class="{ done: progressPercent === 100 }" :style="{ width: progressPercent + '%' }"></div>
            </div>
            <div class="progress-ratio">{{ mediapipeStore.count }}/{{ MAX_REPS }}</div>
            <div class="progress-actions">
            </div>
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
// import WorkoutStats from '@/components/WorkoutStats.vue'
// import AngleDisplay from '@/components/AngleDisplay.vue'
import WorkoutSummary from '@/components/WorkoutSummary.vue'

// --- 1. 从 URL 获取动态数据 ---
const route = useRoute();
const router = useRouter();
const exerciseType = ref(route.params.type);
const style = ref(route.query.style);       // 'motivator' 或 'guide'

// --- 2. 状态管理和组件引用 ---
const mediapipeStore = useMediapipeStore();
const exerciseStore = useExerciseStore();
const analyzer = ref(null);
const currentAngles = ref(null);
const showSummary = ref(false);
const showIntro = ref(false);
const isWorkoutActive = ref(false);
const exerciseData = computed(() => exerciseStore.getExerciseById(exerciseType.value));

const feedbackText = ref("隨時準備！");
const isOverextended = ref(false);

const MAX_REPS = 15;

const progressPercent = computed(() => {
const total = Number(MAX_REPS || 0);
const done = Number(mediapipeStore.count || 0);
if (!total) return 0;
const p = Math.round((done / total) * 100);
return Math.max(0, Math.min(100, p));
});


async function startWorkoutFlow() {
  isWorkoutActive.value = true;
    await nextTick();

  exerciseStore.startExercise();
  setTimeout(() => {
    analyzer.value?.startAnalysis(); // 开始摄像头分析
  }, 100);
  mediapipeStore.startExercise(exerciseType.value, style.value);
}

function closeIntroAndStart() {
  if (!showIntro.value) return;
  showIntro.value = false; // 关闭弹窗
  startWorkoutFlow();      // 执行开始流程
}

onMounted(() => {
  if (!exerciseType.value || !style.value) {
    router.push('/');
    return;
  }
  
  exerciseStore.selectExercise(exerciseType.value);
  
  if (style.value === 'motivator') {
    showIntro.value = true;
  } else {
    startWorkoutFlow();
  }
});

function handleFrameAnalyzed(result) {
    const analysisData = result[exerciseType.value];
    if (analysisData) {
      currentAngles.value = {
        shoulder: analysisData.shoulder_angle,
        elbow: analysisData.elbow_angle
      };
      mediapipeStore.updateAnalysisData(analysisData);
      feedbackText.value = analysisData.feedback;
      const nonStandard = analysisData.category === 'non_standard';
      isOverextended.value = Boolean(analysisData.overextended || nonStandard);
    } else {
      console.log("当前帧未返回有效的分析数据");
    }

    if (result.gesture_detected) {
      console.log(`检测到手势: ${result.gesture_detected}`);
    }
}

function handleError(error) {
  console.error('Analysis error:', error);
}

watch(() => mediapipeStore.count, (newCount, oldCount) => {
  if (newCount > 0 && newCount % mediapipeStore.repetitionLimit === 0 && newCount !== oldCount) {
    isWorkoutActive.value = false;
    exerciseStore.endExercise()
    showSummary.value = true;
    // analyzer.value?.stopAnalysis();
  }
});

function handleContinueWorkout() {
  showSummary.value = false;
  isWorkoutActive.value = true;
  mediapipeStore.reset();
  exerciseStore.startExercise(); 
  setTimeout(() => {
    analyzer.value?.startAnalysis();
  }, 100);
  mediapipeStore.startExercise(exerciseType.value, style.value);
}

function handleEndWorkout() {
  showSummary.value = false;
  mediapipeStore.reset();
  router.push('/');
}
</script>

<style scoped>
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
@media (max-width: 1024px) { .content { grid-template-columns: 1fr; } }
@media (max-width: 768px) {
  .modal-body { flex-direction: column; gap: 20px; }
  .intro-modal-content { padding: 30px; }
  .info-container h2 { font-size: 1.5rem; }
  .feedback-text { font-size: 1.2rem; }
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
font-size: 40px; /* “當前訓練：xxx” */
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
</style>