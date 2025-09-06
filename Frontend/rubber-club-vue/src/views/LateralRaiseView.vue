<template>
  <div class="lateral-raise-view">
    <div class="header">
      <router-link to="/" class="back-button">← 返回</router-link>
      <h1>侧平举分析</h1>
    </div>
    
    <div class="content">
      <div class="main-section">
        <WebcamAnalyzer 
          ref="analyzer"
          :analyze-interval="50"
          @frame-analyzed="handleFrameAnalyzed"
          @error="handleError"
        />
        
        <div class="tips">
          <h3>动作要点</h3>
          <ul>
            <li>保持身体直立，核心收紧</li>
            <li>手臂从身体两侧举起至肩膀高度</li>
            <li>控制下放速度，避免自由落体</li>
            <li>保持肘部微屈，避免完全伸直</li>
          </ul>
        </div>
      </div>
      
      <div class="stats-section">
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
      @close="handleSummaryClose"
    />
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'
import WebcamAnalyzer from '@/components/WebcamAnalyzer.vue'
import WorkoutStats from '@/components/WorkoutStats.vue'
import AngleDisplay from '@/components/AngleDisplay.vue'
import WorkoutSummary from '@/components/WorkoutSummary.vue'

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()

const analyzer = ref(null)
const exerciseType = 'lateral_raise'
const currentAngles = ref(null)
const showSummary = ref(false)

// 设置当前运动类型
onMounted(() => {
  exerciseStore.selectExercise(exerciseType)
  mediapipeStore.startExercise(exerciseType)
})

// 处理帧分析结果
function handleFrameAnalyzed(result) {
  if (result.lateral_raise) {
    currentAngles.value = {
      shoulder: result.lateral_raise.shoulder_angle,
      elbow: result.lateral_raise.elbow_angle
    }
  }
}

// 处理错误
function handleError(error) {
  console.error('Analysis error:', error)
}

// 监听是否需要显示总结
watch(() => mediapipeStore.count, (newCount, oldCount) => {
  if (newCount > 0 && newCount % mediapipeStore.repetitionLimit === 0 && newCount !== oldCount) {
    showSummary.value = true
    analyzer.value?.stopAnalysis()
  }
})

// 关闭总结
function handleSummaryClose() {
  showSummary.value = false
  mediapipeStore.reset()
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