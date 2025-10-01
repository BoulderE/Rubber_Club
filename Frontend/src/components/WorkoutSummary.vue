<template>
  <div class="workout-summary-overlay" @click.self="$emit('close')">
    <div class="workout-summary">
      <h2>Summary</h2>
      
      <div class="summary-stats">
        <div class="stat-card">
          <div class="stat-icon">🎯</div>
          <div class="stat-info">
            <div class="stat-value">{{ mediapipeStore.count }}</div>
            <div class="stat-label">Counts</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon">⚡</div>
          <div class="stat-info">
            <div class="stat-value">{{ mediapipeStore.energy }}%</div>
            <div class="stat-label">Energy</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon">⏱️</div>
          <div class="stat-info">
            <div class="stat-value">{{ duration }}</div>
            <div class="stat-label">Duration</div>
          </div>
        </div>
      </div>
      
      <div class="performance-analysis">
        <h3>Analysis</h3>
        <div class="analysis-item">
          <span class="analysis-label">Accuracy</span>
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: accuracy + '%' }"></div>
          </div>
          <span class="analysis-value">{{ accuracy }}%</span>
        </div>
        
        <div class="analysis-item">
          <span class="analysis-label">Smoothness</span>
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: fluency + '%' }"></div>
          </div>
          <span class="analysis-value">{{ fluency }}%</span>
        </div>
      </div>
      
      <div class="feedback">
        <h3>Advice</h3>
        <ul>
          <li v-for="tip in tips" :key="tip">{{ tip }}</li>
        </ul>
      </div>
      
      <div class="actions">
        <button @click="$emit('continue')" class="btn-primary">Again</button>
        <button @click="$emit('end')" class="btn-secondary">Finish</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed} from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'

// const emit = defineEmits(['end', 'continue'])

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()  

// 从 store 获取当前运动类型
const exerciseType = computed(() => exerciseStore.currentExercise)

const duration = computed(() => {
if (!exerciseStore.startTime || !exerciseStore.endTime) {
    return '0:00'
  }

  const seconds = Math.floor((exerciseStore.endTime - exerciseStore.startTime) / 1000)
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
})

const accuracy = computed(() => {
  // 基于能量值计算准确度（示例）
  return Math.min(95, 70 + mediapipeStore.energy * 0.25)
})

const fluency = computed(() => {
  // 基于次数计算流畅度（示例）
  return Math.min(90, 60 + mediapipeStore.count * 2)
})

const tips = computed(() => {
  const tipsList = []
  
  if (accuracy.value < 80) {
    tipsList.push('注意保持动作标准，可以适当降低速度')
  }
  
  if (mediapipeStore.count < 10) {
    tipsList.push('建议增加运动次数，以达到更好的锻炼效果')
  }
  
  if (mediapipeStore.energy < 50) {
    tipsList.push('能量消耗较低，可以尝试增加动作幅度')
  }
  
  // 根据不同运动类型添加特定建议
  if (exerciseType.value === 'lateral_raise') {
    if (accuracy.value < 85) {
      tipsList.push('侧平举时注意保持手臂在身体侧面')
    }
  } else if (exerciseType.value === 'chest_pull') {
    if (accuracy.value < 85) {
      tipsList.push('拉胸时注意肩胛骨的收缩')
    }
  }
  
  if (tipsList.length === 0) {
    tipsList.push('Good Job! Keep it up!')
  }
  
  return tipsList
})
</script>

<style scoped>
.workout-summary-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.workout-summary {
  background: white;
  border-radius: 20px;
  padding: 40px;
  max-width: 600px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.workout-summary h2 {
  margin-top: 0;
  margin-bottom: 30px;
  text-align: center;
  color: #333;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 30px;
}

.stat-card {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
}

.stat-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #667eea;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-top: 5px;
}

.performance-analysis {
  margin-bottom: 30px;
}

.performance-analysis h3 {
  margin-bottom: 20px;
  color: #333;
}

.analysis-item {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.analysis-label {
  width: 120px;
  color: #666;
  font-size: 14px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #f0f0f0;
  border-radius: 4px;
  margin: 0 15px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.5s ease;
}

.analysis-value {
  width: 50px;
  text-align: right;
  font-weight: 600;
  color: #333;
}

.feedback {
  margin-bottom: 30px;
}

.feedback h3 {
  margin-bottom: 15px;
  color: #333;
}

.feedback ul {
  margin: 0;
  padding-left: 20px;
}

.feedback li {
  color: #666;
  margin-bottom: 8px;
}

.actions {
  display: flex;
  gap: 15px;
  justify-content: center;
}

.btn-primary,
.btn-secondary {
  padding: 12px 30px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.btn-secondary {
  background: #f0f0f0;
  color: #666;
}

.btn-secondary:hover {
  background: #e0e0e0;
  color: #333;
}

@media (max-width: 600px) {
  .summary-stats {
    grid-template-columns: 1fr;
  }
  
  .actions {
    flex-direction: column;
  }
  
  .btn-primary,
  .btn-secondary {
    width: 100%;
  }
}
</style>