<template>
  <div class="workout-summary-overlay" @click.self="$emit('close')">
    <div class="workout-summary">
      <h2>Summary</h2>

      <!-- 顶部三卡：Accuracy / Smoothness / Duration -->
      <div class="summary-stats">
        <div class="stat-card">
          <div class="stat-info">
            <div class="stat-value">{{ accuracy }}%</div>
            <div class="stat-label">動作質素指數</div>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-info">
            <div class="stat-value">{{ smoothnessPercent }}%</div>
            <div class="stat-label">動作流暢指數</div>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-info">
            <div class="stat-value">{{ duration }}</div>
            <div class="stat-label">運動時長</div>
          </div>
        </div>
      </div>

      <!-- 将原 Analysis 位置改为 Advice -->
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
import { computed } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()

const exerciseType = computed(() => exerciseStore.currentExercise)

const duration = computed(() => {
  if (!exerciseStore.startTime || !exerciseStore.endTime) return '0:00'
  const seconds = Math.floor((exerciseStore.endTime - exerciseStore.startTime) / 1000)
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
})

const accuracy = computed(() => Number(mediapipeStore.accuracy) || 0)
const smoothnessPercent = computed(() => Number(mediapipeStore.smoothness) || 0)

// 基于 accuracy 与 smoothness 的针对性建议
const tips = computed(() => {
  const t = []

  // Accuracy 建议
  if (accuracy.value < 60) {
    t.push('動作準確度偏低：放慢節奏，專注關節對齊與完整活動範圍。')
  } else if (accuracy.value < 80) {
    t.push('準確度可再提升：注意核心穩定，保持關節角度在提示範圍內。')
  } else {
    t.push('準確度良好：維持當前節奏，逐步增加組數或阻力。')
  }

  // Smoothness 建議
  if (smoothnessPercent.value < 50) {
    t.push('動作不夠順暢：嘗試均勻呼吸，控制離心階段，避免忽快忽慢。')
  } else if (smoothnessPercent.value < 80) {
    t.push('順暢度中等：在最高點短暫停留，感受肌肉張力後再回程。')
  } else {
    t.push('動作流暢：可微幅加快但保持穩定節奏，追求更佳效率。')
  }

  // 依運動類型附加一條提示（示例）
  if (exerciseType.value === 'lateral_raise') {
    t.push('側平舉：肩膀放鬆下沉，手肘略彎，手腕中立，抬至與肩同高即可。')
  } else if (exerciseType.value === 'chest_pull') {
    t.push('拉胸：啟動肩胛骨後收，胸口打開，避免聳肩代償。')
  }

  return t
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
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
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

/* 移除了原 Analysis 區塊樣式，直接使用 Advice 區塊 */
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