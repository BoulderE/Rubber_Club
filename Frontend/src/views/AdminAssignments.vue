<template>
  <div class="admin-assignments">
    <header>
      <h1>Admin Dashboard</h1>
    </header>

    <nav class="tabs">
      <router-link to="/admin/dashboard" exact-active-class="active">Overview</router-link>
      <router-link to="/admin/users" active-class="active">Users</router-link>
      <router-link to="/admin/assignments" active-class="active">Assignments</router-link>
    </nav>

    <div class="content">
      <h2>Assignments</h2>

      <div v-if="loading" class="loading">Loading...</div>

      <div v-else-if="assignments.length" class="assignments-list">
        <div v-for="a in assignments" :key="a.id" class="assignment-card">
          <div class="assignment-main">
            <div class="assignment-info">
              <h3>{{ a.exercise_name }}</h3>
              <p class="user-name">{{ a.user_name }}</p>
            </div>
            
            <span class="difficulty" :class="a.difficulty">
              {{ a.difficulty === 'beginner' ? 'Beginner' : 'Intermediate' }}
            </span>
            
            <div class="progress-section">
              <span class="progress-text">{{ a.completed_sets }} / {{ a.target_sets }} sets</span>
              <div class="progress-bar">
                <div 
                  class="progress-fill" 
                  :style="{ width: (a.completed_sets / a.target_sets * 100) + '%' }"
                ></div>
              </div>
              <span class="reps-detail">({{ a.target_reps }} reps/set)</span>
            </div>
            
            <div class="due-section">
              <span class="due-label">Due</span>
              <span class="due-date">{{ formatDate(a.due_date) }}</span>
            </div>
            
            <span class="status" :class="a.status">{{ formatStatus(a.status) }}</span>
          </div>
          
          <div class="assignment-actions">
            <button class="btn-secondary" @click="editAssignment(a)">Edit</button>
            <button class="btn-danger" @click="deleteAssignment(a.id)">Delete</button>
          </div>
        </div>
      </div>

      <p v-else class="empty-state">No assignments found.</p>
    </div>

    <!-- Edit Modal -->
    <div v-if="showEdit" class="modal-overlay" @click.self="showEdit = false">
      <div class="modal-content">
        <h3>Edit Assignment</h3>
        <form @submit.prevent="submitEdit">
          <div class="form-group">
            <label>Difficulty</label>
            <select v-model="editForm.difficulty">
              <option value="beginner">Beginner (10 reps per set)</option>
              <option value="intermediate">Intermediate (15 reps per set)</option>
            </select>
          </div>

          <div class="form-group">
            <label>Number of Sets</label>
            <input v-model.number="editForm.target_sets" type="number" min="1" max="10" required />
          </div>

          <div class="assignment-preview">
            <p><strong>Preview:</strong></p>
            <p>{{ editRepsPerSet }} reps × {{ editForm.target_sets }} sets = {{ editTotalReps }} total reps</p>
          </div>

          <div class="form-group">
            <label>Due Date</label>
            <input v-model="editForm.due_date" type="date" />
          </div>

          <div class="form-group">
            <label>Status</label>
            <select v-model="editForm.status">
              <option value="pending">Pending</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
            </select>
          </div>

          <div class="form-group">
            <label>Notes</label>
            <textarea v-model="editForm.admin_notes" rows="3"></textarea>
          </div>

          <div class="modal-actions">
            <button type="button" class="btn-secondary" @click="showEdit = false">Cancel</button>
            <button type="submit" class="btn-primary" :disabled="saving">
              {{ saving ? 'Saving...' : 'Save' }}
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAdminStore } from '@/stores/admin'

const router = useRouter()
const adminStore = useAdminStore()

const loading = ref(true)
const assignments = ref([])
const showEdit = ref(false)
const saving = ref(false)
const editingId = ref(null)

const editForm = ref({
  difficulty: 'beginner',
  target_sets: 3,
  due_date: '',
  status: 'pending',
  admin_notes: ''
})

const editRepsPerSet = computed(() => {
  return editForm.value.difficulty === 'beginner' ? 10 : 15
})

const editTotalReps = computed(() => {
  return editRepsPerSet.value * editForm.value.target_sets
})

onMounted(async () => {
  const res = await adminStore.fetchAssignments()
  if (res.error) {
    router.push('/admin/login')
  } else {
    assignments.value = res || []
  }
  loading.value = false
})

function editAssignment(a) {
  editingId.value = a.id
  editForm.value = {
    difficulty: a.difficulty || 'beginner',
    target_sets: a.target_sets,
    due_date: a.due_date?.split('T')[0] || '',
    status: a.status,
    admin_notes: a.admin_notes || ''
  }
  showEdit.value = true
}

async function submitEdit() {
  saving.value = true
  
  const res = await adminStore.updateAssignment(editingId.value, {
    difficulty: editForm.value.difficulty,
    target_sets: editForm.value.target_sets,
    due_date: editForm.value.due_date || null,
    status: editForm.value.status,
    admin_notes: editForm.value.admin_notes
  })
  
  if (res.success) {
    showEdit.value = false
    const refreshed = await adminStore.fetchAssignments()
    assignments.value = refreshed || []
  } else {
    alert(res.error || 'Failed to update')
  }
  saving.value = false
}

async function deleteAssignment(id) {
  if (!confirm('Delete this assignment?')) return
  const res = await adminStore.deleteAssignment(id)
  if (res.success) {
    assignments.value = assignments.value.filter(a => a.id !== id)
  } else {
    alert(res.error || 'Failed to delete')
  }
}

function formatDate(dateStr) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleDateString()
}

function formatStatus(status) {
  const statusMap = {
    'pending': 'Pending',
    'in_progress': 'In Progress',
    'completed': 'Completed'
  }
  return statusMap[status] || status
}
</script>

<style scoped>
.admin-assignments {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}

header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 24px;
}

.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 32px;
}

.tabs a {
  padding: 10px 20px;
  border-radius: 8px;
  text-decoration: none;
  color: #4b5563;
  font-weight: 500;
  transition: all 0.2s;
}

.tabs a:hover {
  background: #f3f4f6;
}

.tabs a.active {
  background: #3b82f6;
  color: white;
}

.content h2 {
  font-size: 20px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 20px;
}

.loading {
  text-align: center;
  color: #6b7280;
  padding: 40px;
}

.empty-state {
  text-align: center;
  color: #6b7280;
  padding: 40px;
}

/* Assignment Cards */
.assignments-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.assignment-card {
  background: white;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 24px;
}

.assignment-main {
  display: flex;
  align-items: center;
  gap: 32px;
  flex: 1;
}

.assignment-info {
  min-width: 150px;
}

.assignment-info h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px 0;
}

.assignment-info .user-name {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.difficulty {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.difficulty.beginner {
  background: #d1fae5;
  color: #065f46;
}

.difficulty.intermediate {
  background: #fef3c7;
  color: #92400e;
}

.progress-section {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 120px;
}

.progress-text {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #10b981;
  transition: width 0.3s ease;
}

.reps-detail {
  font-size: 12px;
  color: #9ca3af;
}

.due-section {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 80px;
}

.due-label {
  font-size: 12px;
  color: #9ca3af;
}

.due-date {
  font-size: 14px;
  color: #374151;
}

.status {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.status.pending {
  background: #f3f4f6;
  color: #4b5563;
}

.status.in_progress {
  background: #dbeafe;
  color: #1e40af;
}

.status.completed {
  background: #d1fae5;
  color: #065f46;
}

.assignment-actions {
  display: flex;
  gap: 8px;
}

/* Buttons */
.btn-secondary {
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  background: white;
  color: #374151;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

.btn-danger {
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid #fecaca;
  background: #fef2f2;
  color: #dc2626;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-danger:hover {
  background: #fee2e2;
  border-color: #f87171;
}

.btn-primary {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  background: #3b82f6;
  color: white;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary:hover {
  background: #2563eb;
}

.btn-primary:disabled {
  background: #93c5fd;
  cursor: not-allowed;
}

/* Modal */
.modal-overlay {
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

.modal-content {
  background: white;
  border-radius: 16px;
  padding: 32px;
  width: 100%;
  max-width: 480px;
  max-height: 90vh;
  overflow-y: auto;
}

.modal-content h3 {
  font-size: 20px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 24px 0;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  margin-bottom: 6px;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  color: #1f2937;
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.assignment-preview {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
}

.assignment-preview p {
  margin: 4px 0;
  font-size: 14px;
  color: #0369a1;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
}

@media (max-width: 768px) {
  .assignment-card {
    flex-direction: column;
    align-items: stretch;
  }
  
  .assignment-main {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  
  .assignment-actions {
    justify-content: flex-end;
    padding-top: 16px;
    border-top: 1px solid #e5e7eb;
  }
}
</style>