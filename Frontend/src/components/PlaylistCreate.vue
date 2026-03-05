<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-container">
      <header class="modal-header">
        <h2>建立新清單</h2>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </header>

      <div class="modal-body">
        <div class="form-group">
          <label>清單名稱</label>
          <input
            v-model="playlistName"
            type="text"
            placeholder="例如：上肢訓練、每日伸展"
            class="text-input"
            maxlength="50"
          />
        </div>

        <div class="form-group toggle-group">
          <label>
            <input v-model="isRoutine" type="checkbox" />
            <span class="toggle-label">儲存為常用訓練</span>
          </label>
          <p class="hint">常用訓練可重複使用，不會因完成而消失</p>
        </div>

        <div class="form-group">
          <label>選擇運動項目</label>
          <div class="exercise-list">
            <div
              v-for="exercise in availableExercises"
              :key="exercise.exercise_key"
              class="exercise-option"
              :class="{ selected: isSelected(exercise) }"
              @click="toggleExercise(exercise)"
            >
              <span class="exercise-name">{{ exercise.exercise_name }}</span>
              <span v-if="isSelected(exercise)" class="check-icon">✓</span>
            </div>
          </div>
        </div>

        <div v-if="selectedExercises.length > 0" class="form-group">
          <label>已選擇 ({{ selectedExercises.length }}) - 拖曳調整順序</label>
          <div class="selected-list">
            <div
              v-for="(exercise, index) in selectedExercises"
              :key="exercise.exercise_key"
              class="selected-item"
              draggable="true"
              @dragstart="onDragStart(index)"
              @dragover.prevent
              @drop="onDrop(index)"
            >
              <span class="drag-handle">⋮⋮</span>
              <span class="order-num">{{ index + 1 }}</span>
              <span class="exercise-name">{{ exercise.exercise_name }}</span>
              
              <div class="exercise-config">
                <div class="config-item">
                  <label>次數</label>
                  <input
                    v-model.number="exercise.target_reps"
                    type="number"
                    min="1"
                    max="100"
                    class="small-input"
                  />
                </div>
                <div class="config-item">
                  <label>組數</label>
                  <input
                    v-model.number="exercise.target_sets"
                    type="number"
                    min="1"
                    max="20"
                    class="small-input"
                  />
                </div>
              </div>
              
              <button class="remove-btn" @click="removeExercise(index)">✕</button>
            </div>
          </div>
        </div>
      </div>

      <footer class="modal-footer">
        <button class="cancel-btn" @click="$emit('close')">取消</button>
        <button
          class="submit-btn"
          :disabled="!canSubmit || submitting"
          @click="handleCreate"
        >
          {{ submitting ? '建立中...' : '建立清單' }}
        </button>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { createPlaylist } from '@/api/tasks'

// const props = defineProps({
//   availableExercises: {
//     type: Array,
//     default: () => []
//   }
// })

const emit = defineEmits(['close', 'created'])

const playlistName = ref('')
const isRoutine = ref(false)
const selectedExercises = ref([])
const submitting = ref(false)
const dragIndex = ref(null)

const canSubmit = computed(() => {
  return playlistName.value.trim() && selectedExercises.value.length > 0
})

function isSelected(exercise) {
  return selectedExercises.value.some(e => e.exercise_key === exercise.exercise_key)
}

function toggleExercise(exercise) {
  const index = selectedExercises.value.findIndex(
    e => e.exercise_key === exercise.exercise_key
  )
  
  if (index >= 0) {
    selectedExercises.value.splice(index, 1)
  } else {
    selectedExercises.value.push({
      ...exercise,
      target_reps: 10,
      target_sets: 1
    })
  }
}

function removeExercise(index) {
  selectedExercises.value.splice(index, 1)
}

function onDragStart(index) {
  dragIndex.value = index
}

function onDrop(targetIndex) {
  if (dragIndex.value === null || dragIndex.value === targetIndex) return
  
  const item = selectedExercises.value.splice(dragIndex.value, 1)[0]
  selectedExercises.value.splice(targetIndex, 0, item)
  dragIndex.value = null
}

async function handleCreate() {
  if (!canSubmit.value || submitting.value) return
  
  submitting.value = true
  try {
    const exercises = selectedExercises.value.map((ex, index) => ({
      exercise_key: ex.exercise_key,
      exercise_name: ex.exercise_name,
      target_reps: ex.target_reps,
      target_sets: ex.target_sets,
      sort_order: index
    }))
    
    await createPlaylist({
      name: playlistName.value.trim(),
      exercises,
      is_routine: isRoutine.value
    })
    
    emit('created')
  } catch (err) {
    console.error('Failed to create playlist:', err)
    alert('建立失敗，請稍後再試')
  } finally {
    submitting.value = false
  }
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
  align-items: center;
  justify-content: space-between;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-header h2 {
  color: #fff;
  margin: 0;
  font-size: 1.25rem;
}

.close-btn {
  background: none;
  border: none;
  color: #a0a0a0;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0.25rem;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  color: #a0a0a0;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.text-input {
  width: 100%;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  color: #fff;
  font-size: 1rem;
}

.text-input:focus {
  outline: none;
  border-color: #667eea;
}

.toggle-group label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.toggle-group input[type="checkbox"] {
  width: 20px;
  height: 20px;
  accent-color: #667eea;
}

.toggle-label {
  color: #fff;
  font-size: 1rem;
}

.hint {
  color: #666;
  font-size: 0.75rem;
  margin: 0.5rem 0 0;
}

.exercise-list {
  display: grid;
  gap: 0.5rem;
  max-height: 200px;
  overflow-y: auto;
}

.exercise-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.exercise-option:hover {
  background: rgba(255, 255, 255, 0.08);
}

.exercise-option.selected {
  background: rgba(102, 126, 234, 0.2);
  border-color: #667eea;
}

.exercise-option .exercise-name {
  color: #fff;
}

.check-icon {
  color: #667eea;
  font-weight: bold;
}

.selected-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.selected-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  cursor: grab;
}

.selected-item:active {
  cursor: grabbing;
}

.drag-handle {
  color: #666;
  font-size: 1rem;
}

.order-num {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #667eea;
  color: #fff;
  border-radius: 50%;
  font-size: 0.75rem;
  font-weight: bold;
}

.selected-item .exercise-name {
  flex: 1;
  color: #fff;
  font-size: 0.9rem;
}

.exercise-config {
  display: flex;
  gap: 0.5rem;
}

.config-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.config-item label {
  font-size: 0.65rem;
  color: #666;
  margin: 0;
}

.small-input {
  width: 50px;
  padding: 0.25rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  color: #fff;
  text-align: center;
  font-size: 0.875rem;
}

.remove-btn {
  background: none;
  border: none;
  color: #ef4444;
  cursor: pointer;
  padding: 0.25rem;
  font-size: 1rem;
}

.modal-footer {
  display: flex;
  gap: 1rem;
  padding: 1.25rem 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.cancel-btn,
.submit-btn {
  flex: 1;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-btn {
  background: rgba(255, 255, 255, 0.1);
  border: none;
  color: #fff;
}

.cancel-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: #fff;
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.submit-btn:not(:disabled):hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
</style>