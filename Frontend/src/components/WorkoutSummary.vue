<template>
  <div class="workout-summary-overlay" @click.self="$emit('close')">
    <div class="workout-summary">
      <h2>訓練總結</h2>

      <!-- new -->
      <div v-if="exerciseStore.isPlaylistMode" class="playlist-progress">
        <span class="progress-text">
          訓練進度：{{ exerciseStore.playlistProgress.current }} / {{ exerciseStore.playlistProgress.total }}
        </span>
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            :style="{ width: `${(exerciseStore.playlistProgress.current / exerciseStore.playlistProgress.total) * 100}%` }"
          ></div>
        </div>
      </div>

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

      <div class="feedback">
        <h3>訓練建議</h3>
        <ul>
          <li v-for="tip in tips" :key="tip">{{ tip }}</li>
        </ul>
      </div>

      <div class="actions">
        <button @click="$emit('continue')" class="btn-primary">再來一組</button>
        <!-- new -->
        <button 
          v-if="exerciseStore.hasNextInPlaylist" 
          @click="goToNext" 
          class="btn-next"
        >
          下一項：{{ nextExerciseName }}
        </button>
        <button @click="handleEnd" class="btn-secondary">
          {{ exerciseStore.isPlaylistMode ? '結束整組訓練' : '結束訓練' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'
import { useExerciseStore } from '@/stores/exercise'
import confetti from 'canvas-confetti'

const mediapipeStore = useMediapipeStore()
const exerciseStore = useExerciseStore()

// new
const emit = defineEmits(['continue', 'end', 'next', 'close'])
const nextExerciseName = computed(() => {
  const nextId = exerciseStore.playlist[exerciseStore.playlistIndex + 1]
  if (!nextId) return ''
  const exercise = exerciseStore.getExerciseById(nextId)
  return exercise?.name || nextId
})

function goToNext() {
  emit('next')
}

function handleEnd() {
  exerciseStore.clearPlaylist()
  emit('end')
}

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

const tips = computed(() => {
  const t = []

  if (accuracy.value < 60) {
    t.push('動作準確度偏低：放慢節奏，專注關節對齊與完整活動範圍。')
  } else if (accuracy.value < 80) {
    t.push('準確度可再提升：注意核心穩定，保持關節角度在提示範圍內。')
  } else {
    t.push('準確度良好：維持當前節奏，逐步增加組數或阻力。')
  }

  if (smoothnessPercent.value < 50) {
    t.push('動作不夠順暢：嘗試均勻呼吸，控制離心階段，避免忽快忽慢。')
  } else if (smoothnessPercent.value < 80) {
    t.push('順暢度中等：在最高點短暫停留，感受肌肉張力後再回程。')
  } else {
    t.push('動作流暢：可微幅加快但保持穩定節奏，追求更佳效率。')
  }

  if (exerciseType.value === 'lateral_raise') {
    t.push('側平舉：肩膀放鬆下沉，手肘略彎，手腕中立，抬至與肩同高即可。')
  } else if (exerciseType.value === 'chest_pull') {
    t.push('拉胸：啟動肩胛骨後收，胸口打開，避免聳肩代償。')
  }

  return t
})

function blastConfetti() {
  setTimeout(() => {
    const duration = 2500
    const end = Date.now() + duration

    const base = {
      particleCount: 50,
      spread: 60,
      startVelocity: 45,
      gravity: 0.9,
      ticks: 250,
      origin: { y: 0.6 },
      zIndex: 9999 
    }

    const interval = setInterval(() => {

      confetti({ 
        ...base, 
        angle: 60, 
        origin: { x: 0, y: Math.random() * 0.3 + 0.1 } 
      })
      
      confetti({ 
        ...base, 
        angle: 120, 
        origin: { x: 1, y: Math.random() * 0.3 + 0.1 } 
      })
      
      confetti({
        ...base,
        particleCount: 80,
        spread: 90,
        origin: { x: 0.5, y: 0.3 },
        scalar: 1.1,
        colors: ['#34d399', '#3b82f6', '#f59e0b', '#ef4444', '#a78bfa']
      })
      
      if (Date.now() > end) {
        clearInterval(interval)
      }
    }, 250)
  }, 100) 
}

onMounted(() => {
  console.log('🎉 Summary 组件已挂载，准备发射烟花')
  blastConfetti()
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
  position: relative; 
  z-index: 1001; 
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
  font-size: 40px;
  font-weight: 700;
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
  font-size: 40px;
  font-weight: 700;
  color: #667eea;
}

.stat-label {
  font-size: 18px;
  color: #666;
  margin-top: 5px;
  font-weight: 500;
}

.feedback {
  margin-bottom: 30px;
}

.feedback h3 {
  margin-bottom: 20px;
  color: #333;
  font-size: 32px;
  font-weight: 700;
}

.feedback ul {
  margin: 0;
  padding-left: 24px;
}

.feedback li {
  color: #555;
  margin-bottom: 16px;
  font-size: 20px;
  line-height: 1.7;
  font-weight: 500;
}

.actions {
  display: flex;
  gap: 15px;
  justify-content: center;
}

.btn-primary,
.btn-secondary {
  padding: 14px 35px;
  border: none;
  border-radius: 8px;
  font-size: 18px;
  font-weight: 600;
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
  
  .workout-summary h2 {
    font-size: 32px;
  }
  
  .stat-value {
    font-size: 32px;
  }
  
  .stat-label {
    font-size: 16px;
  }
  
  .feedback h3 {
    font-size: 26px;
  }
  
  .feedback li {
    font-size: 17px;
    margin-bottom: 14px;
  }
  
  .actions {
    flex-direction: column;
  }
  
  .btn-primary,
  .btn-secondary {
    width: 100%;
    font-size: 16px;
  }
}

.btn-next {
  padding: 14px 35px;
  border: none;
  border-radius: 8px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
  color: white;
}

.btn-next:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
}
</style>