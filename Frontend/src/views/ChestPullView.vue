<template>
  <div class="chest-pull-view">
    <div class="header">
      <router-link to="/" class="back-button">← 返回</router-link>
      <h1>拉胸分析</h1>
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
            <li>双手握住弹力带或拉力器把手</li>
            <li>保持背部挺直，核心收紧</li>
            <li>将把手拉向胸部，肩胛骨向后收缩</li>
            <li>控制速度，避免借力</li>
            <li>呼吸节奏：拉时呼气，还原时吸气</li>
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
      @continue="handleContinueWorkout"
      @end="handleEndWorkout"
    />
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'
import WebcamAnalyzer from '@/components/WebcamAnalyzer.vue'
import WorkoutStats from '@/components/WorkoutStats.vue'
import AngleDisplay from '@/components/AngleDisplay.vue'
import WorkoutSummary from '@/components/WorkoutSummary.vue'

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()
const router = useRouter()

const analyzer = ref(null)
const exerciseType = 'chest_pull'
const currentAngles = ref(null)
const showSummary = ref(false)

// 设置当前运动类型
onMounted(() => {
  exerciseStore.selectExercise(exerciseType)
  mediapipeStore.startExercise(exerciseType)
})

// 处理帧分析结果
function handleFrameAnalyzed(result) {
  if (result.chest_pull) {
    currentAngles.value = {
      shoulder: result.chest_pull.shoulder_angle,
      elbow: result.chest_pull.elbow_angle
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

function handleContinueWorkout() {
  console.log('继续运动被触发')
  showSummary.value = false
  mediapipeStore.reset()
  
  setTimeout(() => {
    if (analyzer.value) {
      console.log('重新启动分析器')
      analyzer.value.startAnalysis()
    }

  // 重新启动 MediaPipe
  mediapipeStore.startExercise(exerciseType)
    }, 100)
}

function handleEndWorkout() {
  showSummary.value = false       // 1. 关闭总结窗口
  mediapipeStore.reset()          // 2. 重置数据
  router.push('/')                // 3. 返回主页
}
</script>

<style scoped>
.chest-pull-view {
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