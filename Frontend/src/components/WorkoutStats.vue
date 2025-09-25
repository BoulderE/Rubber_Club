<template>
  <div class="workout-stats">
    <h3>Stats</h3>
    
    <div class="stat-item">
      <span class="stat-label">Status</span>
      <span :class="['stat-value', 'status', statusClass]">{{ mediapipeStore.status }}</span>
    </div>
    
    <div class="stat-item">
      <span class="stat-label">Count</span>
      <span class="stat-value count">{{ mediapipeStore.count }}</span>
    </div>
    
    <div class="stat-item">
      <span class="stat-label">Energy</span>
      <div class="energy-bar">
        <div class="energy-fill" :style="{ width: energyPercentage + '%' }"></div>
      </div>
      <span class="stat-value">{{ mediapipeStore.energy }}%</span>
    </div>
    
    <div v-if="mediapipeStore.isPaused" class="pause-indicator">
      <span class="pause-icon">⏸️</span>
      <span>Paused</span>
    </div>
    
    <div v-if="mediapipeStore.overextension" class="warning">
      <span class="warning-icon">⚠️</span>
      <span>Over-extension</span>
    </div>
    
    <button @click="handleReset" class="reset-button">
      Reset
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'

const mediapipeStore = useMediapipeStore()

const statusClass = computed(() => {
  switch (mediapipeStore.status) {
    case 'Ready': return 'ready'
    case 'Analyzing': return 'analyzing'
    case 'Paused': return 'paused'
    default: return ''
  }
})

const energyPercentage = computed(() => {
  return Math.min(mediapipeStore.energy, 100)
})

function handleReset() {
  mediapipeStore.reset()
}
</script>

<style scoped>
.workout-stats {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.workout-stats h3 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #333;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 15px;
  border-bottom: 1px solid #f0f0f0;
}

.stat-label {
  color: #666;
  font-size: 14px;
}

.stat-value {
  font-weight: 600;
  color: #333;
}

.stat-value.count {
  font-size: 24px;
  color: #667eea;
}

.status.ready { color: #2ed573; }
.status.analyzing { color: #667eea; }
.status.paused { color: #ffa502; }

.energy-bar {
  flex: 1;
  height: 8px;
  background: #f0f0f0;
  border-radius: 4px;
  margin: 0 15px;
  overflow: hidden;
}

.energy-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
}

.pause-indicator,
.warning {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px;
  border-radius: 8px;
  margin-top: 15px;
  font-size: 14px;
}

.pause-indicator {
  background: rgba(255, 165, 2, 0.1);
  color: #ffa502;
}

.warning {
  background: rgba(255, 71, 87, 0.1);
  color: #ff4757;
}

.reset-button {
  width: 100%;
  padding: 10px;
  margin-top: 20px;
  border: none;
  border-radius: 8px;
  background: #f0f0f0;
  color: #666;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s;
}

.reset-button:hover {
  background: #e0e0e0;
  color: #333;
}
</style>