<template>
  <div class="exercise-view" v-if="exerciseData">
    <div class="header">
      <router-link to="/" class="back-button">← 返回</router-link>
      <h1>{{ exerciseData?.title || 'Analysis' }}</h1>
    </div>
    
    <div class="content">
      <div class="main-section">
        <WebcamAnalyzer 
          ref="analyzer"
          :analyze-interval="50"
          @frame-analyzed="handleFrameAnalyzed"
          @error="handleError"
        />
        
        <div class="tips" v-if="exerciseData?.tips?.length">
          <h3>动作要点</h3>
          <ul>
            <li v-for="tip in exerciseData.tips" :key="tip">{{ tip }}</li>
          </ul>
        </div>
      </div>
      
      <div class="stats-section">
        <!-- exerciseType 是动态的 -->
        <WorkoutStats :exercise-type="exerciseType" />
        <AngleDisplay 
          v-if="currentAngles"
          :angles="currentAngles"
          :exercise-type="exerciseType"
        />
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
import { ref, watch, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'
import WebcamAnalyzer from '@/components/WebcamAnalyzer.vue'
import WorkoutStats from '@/components/WorkoutStats.vue'
import AngleDisplay from '@/components/AngleDisplay.vue'
import WorkoutSummary from '@/components/WorkoutSummary.vue'

// --- 1. 从 URL 获取动态数据 ---
const route = useRoute();
const router = useRouter();
const exerciseType = ref(route.params.type); // 'chest_pull' 或 'lateral_raise'
const style = ref(route.query.style);       // 'motivator' 或 'guide'

// --- 2. 状态管理和组件引用 ---
const mediapipeStore = useMediapipeStore();
const exerciseStore = useExerciseStore();
const analyzer = ref(null);
const currentAngles = ref(null);
const showSummary = ref(false);

const exerciseData = computed(() => exerciseStore.getExerciseById(exerciseType.value));

onMounted(() => {
  if (!exerciseType.value || !style.value) {
    console.error("运动类型或难度模式缺失!");
    router.push('/'); // 如果缺少参数，返回主页
    return;
  }
  
  console.log(`准备开始运动: ${exerciseType.value}, 模式: ${style.value}`);
  exerciseStore.selectExercise(exerciseType.value);
  mediapipeStore.startExercise(exerciseType.value, style.value);
});

// --- 5. 其他逻辑 (与您之前的代码基本相同) ---
function handleFrameAnalyzed(result) {
    const analysisData = result[exerciseType.value];

  if (analysisData) {
    // a. 更新角度显示 (如果您的后端返回了角度数据)
    currentAngles.value = {
      shoulder: analysisData.shoulder_angle,
      elbow: analysisData.elbow_angle
    };

    // b. 【关键】将提取出的、正确的分析数据提交给 Pinia Store
    mediapipeStore.updateAnalysisData(analysisData);

  } else {
    console.log("当前帧未返回有效的分析数据");
  }

  // 额外处理手势检测结果
  if (result.gesture_detected) {
    console.log(`检测到手势: ${result.gesture_detected}`);
    // 你可以在这里添加一些UI反馈，比如显示一个图标
  }
}

function handleError(error) {
  console.error('Analysis error:', error);
}

watch(() => mediapipeStore.count, (newCount, oldCount) => {
  if (newCount > 0 && newCount % mediapipeStore.repetitionLimit === 0 && newCount !== oldCount) {
    showSummary.value = true;
    analyzer.value?.stopAnalysis();
  }
});

function handleContinueWorkout() {
  showSummary.value = false;
  mediapipeStore.reset();
  setTimeout(() => {
    analyzer.value?.startAnalysis();
    mediapipeStore.startExercise(exerciseType.value, style.value);
  }, 100);
}

function handleEndWorkout() {
  showSummary.value = false;
  mediapipeStore.reset();
  router.push('/');
}
</script>

<style scoped>
.lateral-raise-view {
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
  display: flex;
  align-items: center;
  gap: 5px;
  transition: color 0.3s;
}

.back-button:hover {
  color: #5a54d8;
}

.header h1 {
  margin: 0;
  color: #333;
}

.content {
  display: grid;
  grid-template-columns: 1fr 350px;
  gap: 30px;
}

.main-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.tips {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.tips h3 {
  margin-top: 0;
  color: #333;
}

.tips ul {
  margin: 0;
  padding-left: 20px;
  color: #666;
}

.tips li {
  margin-bottom: 8px;
}

.stats-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

@media (max-width: 1024px) {
  .content {
    grid-template-columns: 1fr;
  }
}
</style>