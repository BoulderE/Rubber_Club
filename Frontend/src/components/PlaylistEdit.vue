<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-container">
      <header class="modal-header">
        <h2>編輯清單</h2>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </header>

      <div class="modal-body">
        <!-- Playlist Name -->
        <div class="form-group">
          <label>清單名稱</label>
          <input
            v-model="localName"
            type="text"
            placeholder="例如：上肢訓練、每日伸展"
            class="text-input"
            maxlength="50"
          />
        </div>

        <!-- Save as Routine Toggle -->
        <div class="form-group toggle-group">
          <label>
            <input v-model="localIsRoutine" type="checkbox" />
            <span class="toggle-label">儲存為常用訓練</span>
          </label>
          <p class="hint">常用訓練可重複使用，不會因完成而消失</p>
        </div>

        <!-- Current Exercises -->
        <div class="form-group">
          <label>目前運動項目 ({{ localExercises.length }})</label>
          <div v-if="localExercises.length > 0" class="selected-list">
            <div
              v-for="(exercise, index) in localExercises"
              :key="exercise.id || exercise.exercise_key"
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
          <div v-else class="empty-exercises">
            <p>尚未新增任何運動</p>
          </div>
        </div>

        <!-- Add More Exercises -->
        <div class="form-group">
          <label>新增運動項目</label>
          <div class="add-exercise-section">
            <button 
              class="toggle-add-btn"
              @click="showExerciseSelector = !showExerciseSelector"
            >
              {{ showExerciseSelector ? '收起' : '+ 新增運動' }}
            </button>
            
            <div v-if="showExerciseSelector" class="exercise-list">
              <div
                v-for="exercise in availableToAdd"
                :key="exercise.exercise_key"
                class="exercise-option"
                @click="addExercise(exercise)"
              >
                <span class="exercise-name">{{ exercise.exercise_name }}</span>
                <span class="add-icon">+</span>
              </div>
              <div v-if="availableToAdd.length === 0" class="no-more">
                所有運動項目都已加入
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer class="modal-footer">
        <button class="cancel-btn" @click="$emit('close')">取消</button>
        <button
          class="submit-btn"
          :disabled="!canSubmit || submitting"
          @click="handleUpdate"
        >
          {{ submitting ? '儲存中...' : '儲存變更' }}
        </button>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { updatePlaylist } from '@/api/tasks'

const props = defineProps({
  playlist: {
    type: Object,
    required: true
  },
  playlistExercises: {
    type: Array,
    default: () => []
  },
  availableExercises: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['close', 'updated'])

// Local state
const localName = ref('')
const localIsRoutine = ref(false)
const localExercises = ref([])
const showExerciseSelector = ref(false)
const submitting = ref(false)
const dragIndex = ref(null)

// Initialize local state from props
watch(
  () => props.playlist,
  (newPlaylist) => {
    if (newPlaylist) {
      localName.value = newPlaylist.playlist_name || ''
      localIsRoutine.value = newPlaylist.is_routine || false
    }
  },
  { immediate: true }
)

watch(
  () => props.playlistExercises,
  (newExercises) => {
    localExercises.value = (newExercises || []).map(ex => ({
      ...ex,
      target_reps: ex.target_reps || 10,
      target_sets: ex.target_sets || 3
    }))
  },
  { immediate: true }
)

// Computed
const canSubmit = computed(() => {
  return localName.value.trim() && localExercises.value.length > 0
})

const availableToAdd = computed(() => {
  const currentKeys = new Set(localExercises.value.map(e => e.exercise_key))
  return props.availableExercises.filter(e => !currentKeys.has(e.exercise_key))
})

// Methods
function addExercise(exercise) {
  localExercises.value.push({
    exercise_key: exercise.exercise_key,
    exercise_name: exercise.exercise_name,
    target_reps: 10,
    target_sets: 3,
    isNew: true
  })
}

function removeExercise(index) {
  localExercises.value.splice(index, 1)
}

function onDragStart(index) {
  dragIndex.value = index
}

function onDrop(targetIndex) {
  if (dragIndex.value === null || dragIndex.value === targetIndex) return
  
  const item = localExercises.value.splice(dragIndex.value, 1)[0]
  localExercises.value.splice(targetIndex, 0, item)
  dragIndex.value = null
}

async function handleUpdate() {
  if (!canSubmit.value || submitting.value) return
  
  submitting.value = true
  try {
    const exercises = localExercises.value.map((ex, index) => ({
      id: ex.id,
      exercise_key: ex.exercise_key,
      exercise_name: ex.exercise_name,
      target_reps: ex.target_reps,
      target_sets: ex.target_sets,
      sort_order: index
    }))
    
    await updatePlaylist(props.playlist.playlist_id, {
      name: localName.value.trim(),
      exercises,
      is_routine: localIsRoutine.value
    })
    
    emit('updated')
  } catch (err) {
    console.error('Failed to update playlist:', err)
    alert('儲存失敗，請稍後再試')
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

.close-btn:hover {
  color: #fff;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group > label {
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
  box-sizing: border-box;
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

.selected-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.selected-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  cursor: grab;
  flex-wrap: wrap;
}

.selected-item:active {
  cursor: grabbing;
  background: rgba(255, 255, 255, 0.08);
}

.drag-handle {
  color: #666;
  font-size: 1rem;
  cursor: grab;
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
  flex-shrink: 0;
}

.selected-item .exercise-name {
  flex: 1;
  color: #fff;
  font-size: 0.9rem;
  min-width: 80px;
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

.small-input:focus {
  outline: none;
  border-color: #667eea;
}

.remove-btn {
  background: none;
  border: none;
  color: #ef4444;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  font-size: 1rem;
  opacity: 0.7;
  transition: opacity 0.2s;
}

.remove-btn:hover {
  opacity: 1;
}

.empty-exercises {
  padding: 2rem;
  text-align: center;
  background: rgba(255, 255, 255, 0.02);
  border: 1px dashed rgba(255, 255, 255, 0.1);
  border-radius: 10px;
}

.empty-exercises p {
  color: #666;
  margin: 0;
}

.add-exercise-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.toggle-add-btn {
  padding: 0.75rem 1rem;
  background: rgba(102, 126, 234, 0.1);
  border: 1px dashed rgba(102, 126, 234, 0.3);
  border-radius: 10px;
  color: #667eea;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
}

.toggle-add-btn:hover {
  background: rgba(102, 126, 234, 0.2);
  border-color: #667eea;
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
  background: rgba(102, 126, 234, 0.15);
  border-color: rgba(102, 126, 234, 0.3);
}

.exercise-option .exercise-name {
  color: #fff;
}

.add-icon {
  color: #667eea;
  font-weight: bold;
  font-size: 1.25rem;
}

.no-more {
  padding: 1rem;
  text-align: center;
  color: #666;
  font-size: 0.875rem;
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
  border: none;
  transition: all 0.2s;
}

.cancel-btn {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.cancel-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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