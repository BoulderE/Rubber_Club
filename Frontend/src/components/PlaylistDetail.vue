<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-container">
      <header class="modal-header">
        <div class="header-info">
          <span class="playlist-icon">{{ playlist.is_routine ? '⭐' : '📋' }}</span>
          <div>
            <h2>{{ playlist.playlist_name }}</h2>
            <span class="playlist-meta">
              {{ exercises.length }} 個動作
              <span v-if="playlist.is_routine" class="routine-badge">常用訓練</span>
            </span>
          </div>
        </div>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </header>

      <div class="modal-body">
        <!-- Progress Summary -->
        <div v-if="progressPercent > 0" class="progress-section">
          <div class="progress-header">
            <span class="progress-label">完成進度</span>
            <span class="progress-value">{{ progressPercent }}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <p class="progress-detail">
            已完成 {{ completedCount }} / {{ exercises.length }} 個動作
          </p>
        </div>

        <!-- Exercise List -->
        <div class="exercise-section">
          <h3 class="section-title">運動項目</h3>
          <div class="exercise-list">
            <div
              v-for="(exercise, index) in exercises"
              :key="exercise.id || index"
              class="exercise-item"
              :class="{ completed: exercise.status === 'completed' }"
            >
              <div class="exercise-order">
                <span v-if="exercise.status === 'completed'" class="check-icon">✓</span>
                <span v-else>{{ index + 1 }}</span>
              </div>
              
              <div class="exercise-info">
                <span class="exercise-name">{{ exercise.exercise_name }}</span>
                <span class="exercise-target">
                  目標：{{ exercise.target_reps }} 次 × {{ exercise.target_sets }} 組
                </span>
              </div>

              <div v-if="exercise.status === 'completed'" class="exercise-result">
                <span class="result-value">
                  {{ exercise.completed_reps_total || 0 }} 次
                </span>
                <span v-if="exercise.avg_smoothness" class="result-smoothness">
                  流暢度 {{ Math.round(exercise.avg_smoothness) }}%
                </span>
              </div>
              
              <div v-else class="exercise-status">
                <span v-if="exercise.status === 'in_progress'" class="status-badge in-progress">
                  進行中
                </span>
                <span v-else class="status-badge pending">
                  待完成
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Notes Section -->
        <div v-if="playlist.admin_notes || playlist.user_notes" class="notes-section">
          <div v-if="playlist.admin_notes" class="note-item">
            <h4>治療師備註</h4>
            <p>{{ playlist.admin_notes }}</p>
          </div>
          <div v-if="playlist.user_notes" class="note-item">
            <h4>我的備註</h4>
            <p>{{ playlist.user_notes }}</p>
          </div>
        </div>

        <!-- Playlist Info -->
        <div class="info-section">
          <div v-if="playlist.assigned_date" class="info-item">
            <span class="info-label">指派日期</span>
            <span class="info-value">{{ formatDate(playlist.assigned_date) }}</span>
          </div>
          <div v-if="playlist.due_date" class="info-item">
            <span class="info-label">截止日期</span>
            <span class="info-value" :class="{ overdue: isOverdue }">
              {{ formatDate(playlist.due_date) }}
              <span v-if="isOverdue" class="overdue-text">（已過期）</span>
            </span>
          </div>
          <div v-if="playlist.created_by" class="info-item">
            <span class="info-label">建立者</span>
            <span class="info-value">
              {{ playlist.created_by === 'admin' ? '治療師' : '我' }}
            </span>
          </div>
        </div>
      </div>

      <footer class="modal-footer">
        <button class="edit-btn" @click="$emit('edit')">
          ✏️ 編輯
        </button>
        <button class="start-btn" @click="$emit('start')">
          ▶ {{ hasStarted ? '繼續訓練' : '開始訓練' }}
        </button>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  playlist: {
    type: Object,
    required: true
  },
  exercises: {
    type: Array,
    default: () => []
  }
})

defineEmits(['close', 'start', 'edit'])

// Computed
const completedCount = computed(() => {
  return props.exercises.filter(e => e.status === 'completed').length
})

const progressPercent = computed(() => {
  if (props.exercises.length === 0) return 0
  return Math.round((completedCount.value / props.exercises.length) * 100)
})

const hasStarted = computed(() => {
  return props.exercises.some(e => e.status === 'completed' || e.status === 'in_progress')
})

const isOverdue = computed(() => {
  if (!props.playlist.due_date) return false
  const dueDate = new Date(props.playlist.due_date)
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return dueDate < today
})

// Methods
function formatDate(dateStr) {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-TW', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  z-index: 1000;
}

.modal-container {
  background: #1a1a2e;
  border-radius: 20px;
  width: 100%;
  max-width: 500px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  gap: 1rem;
}

.header-info {
  display: flex;
  align-items: center;
  gap: 1rem;
  min-width: 0;
}

.playlist-icon {
  font-size: 2.5rem;
  flex-shrink: 0;
}

.modal-header h2 {
  color: #fff;
  margin: 0 0 0.25rem;
  font-size: 1.25rem;
  word-break: break-word;
}

.playlist-meta {
  color: #a0a0a0;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.routine-badge {
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
  padding: 0.125rem 0.5rem;
  border-radius: 10px;
  font-size: 0.75rem;
}

.close-btn {
  background: none;
  border: none;
  color: #a0a0a0;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0.25rem;
  flex-shrink: 0;
}

.close-btn:hover {
  color: #fff;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

/* Progress Section */
.progress-section {
  background: rgba(102, 126, 234, 0.1);
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1.5rem;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.progress-label {
  color: #a0a0a0;
  font-size: 0.875rem;
}

.progress-value {
  color: #667eea;
  font-weight: 700;
  font-size: 1.1rem;
}

.progress-bar {
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 4px;
  transition: width 0.3s;
}

.progress-detail {
  color: #a0a0a0;
  font-size: 0.75rem;
  margin: 0.5rem 0 0;
  text-align: center;
}

/* Exercise Section */
.exercise-section {
  margin-bottom: 1.5rem;
}

.section-title {
  color: #fff;
  font-size: 1rem;
  margin: 0 0 0.75rem;
}

.exercise-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.exercise-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all 0.2s;
}

.exercise-item.completed {
  background: rgba(34, 197, 94, 0.1);
  border-color: rgba(34, 197, 94, 0.2);
}

.exercise-order {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  color: #a0a0a0;
  font-size: 0.875rem;
  font-weight: 600;
  flex-shrink: 0;
}

.exercise-item.completed .exercise-order {
  background: #22c55e;
  color: #fff;
}

.check-icon {
  font-size: 0.875rem;
}

.exercise-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.exercise-name {
  color: #fff;
  font-size: 0.9rem;
}

.exercise-target {
  color: #666;
  font-size: 0.75rem;
}

.exercise-result {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.125rem;
}

.result-value {
  color: #22c55e;
  font-weight: 600;
  font-size: 0.875rem;
}

.result-smoothness {
  color: #a0a0a0;
  font-size: 0.7rem;
}

.exercise-status {
  flex-shrink: 0;
}

.status-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 500;
}

.status-badge.pending {
  background: rgba(255, 255, 255, 0.1);
  color: #a0a0a0;
}

.status-badge.in-progress {
  background: rgba(234, 179, 8, 0.2);
  color: #eab308;
}

/* Notes Section */
.notes-section {
  margin-bottom: 1.5rem;
}

.note-item {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 10px;
  padding: 1rem;
  margin-bottom: 0.75rem;
}

.note-item:last-child {
  margin-bottom: 0;
}

.note-item h4 {
  color: #a0a0a0;
  font-size: 0.8rem;
  margin: 0 0 0.5rem;
  font-weight: 500;
}

.note-item p {
  color: #fff;
  font-size: 0.9rem;
  margin: 0;
  line-height: 1.5;
}

/* Info Section */
.info-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.info-label {
  color: #666;
  font-size: 0.8rem;
}

.info-value {
  color: #a0a0a0;
  font-size: 0.85rem;
}

.info-value.overdue {
  color: #ef4444;
}

.overdue-text {
  font-size: 0.75rem;
}

/* Footer */
.modal-footer {
  display: flex;
  gap: 1rem;
  padding: 1.25rem 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.edit-btn,
.start-btn {
  flex: 1;
  padding: 0.875rem 1rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.edit-btn {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.edit-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.start-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
}

.start-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
</style>