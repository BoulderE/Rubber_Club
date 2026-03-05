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
          <div class="assignment-info">
            <h3>{{ a.exercise_name }}</h3>
            <span class="user-name">{{ a.user_name }}</span>
            <span class="difficulty" :class="a.difficulty">
              {{ a.difficulty === 'beginner' ? 'Beginner' : 'Intermediate' }}
            </span>
          </div>
          
          <div class="assignment-progress">
            <div class="progress-text">{{ a.completed_sets }} / {{ a.target_sets }} sets</div>
            <div class="progress-bar">
              <div 
                class="progress-fill" 
                :style="{ width: (a.completed_sets / a.target_sets * 100) + '%' }"
              ></div>
            </div>
            <span class="reps-detail">({{ a.target_reps }} reps/set)</span>
          </div>
          
          <div class="assignment-meta">
            <span class="due-date">Due: {{ formatDate(a.due_date) }}</span>
            <span class="status" :class="a.status">{{ formatStatus(a.status) }}</span>
          </div>
          
          <div class="assignment-actions">
            <button @click="editAssignment(a)">Edit</button>
            <button @click="deleteAssignment(a.id)" class="delete-btn">Delete</button>
          </div>
        </div>
      </div>

      <p v-else class="empty">No assignments found.</p>
    </div>

    <!-- Edit Modal -->
    <div v-if="showEdit" class="modal" @click.self="showEdit = false">
      <div class="modal-content">
        <h3>Edit Assignment</h3>
        <form @submit.prevent="submitEdit">
          <label>Difficulty</label>
          <select v-model="editForm.difficulty">
            <option value="beginner">Beginner (10 reps per set)</option>
            <option value="intermediate">Intermediate (15 reps per set)</option>
          </select>

          <label>Number of Sets</label>
          <input v-model.number="editForm.target_sets" type="number" min="1" max="10" required />

          <div class="assignment-preview">
            <p><strong>Preview:</strong></p>
            <p>{{ editRepsPerSet }} reps × {{ editForm.target_sets }} sets = {{ editTotalReps }} total reps</p>
          </div>

          <label>Due Date</label>
          <input v-model="editForm.due_date" type="date" />

          <label>Status</label>
          <select v-model="editForm.status">
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
          </select>

          <label>Notes</label>
          <textarea v-model="editForm.admin_notes"></textarea>

          <div class="modal-actions">
            <button type="button" @click="showEdit = false">Cancel</button>
            <button type="submit" :disabled="saving">
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
  min-height: 100vh;
  background: #f5f7fa;
  padding: 1rem;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

h1 { margin: 0; color: #1a1a2e; }

.tabs {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.tabs a {
  padding: 10px 20px;
  background: #fff;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
}

.tabs a.active {
  background: #4a90d9;
  color: #fff;
}

.content h2 {
  margin-bottom: 1rem;
}

.loading {
  text-align: center;
  padding: 2rem;
  color: #666;
}

.empty {
  text-align: center;
  padding: 2rem;
  color: #666;
}

/* Assignment Cards */
.assignments-list {
  display: grid;
  gap: 1rem;
}

.assignment-card {
  background: #fff;
  padding: 1rem;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}

.assignment-info {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 200px;
}

.assignment-info h3 { 
  margin: 0; 
  font-size: 1rem;
}

.user-name { 
  color: #666; 
  font-size: 0.9rem; 
}

.difficulty {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
}

.difficulty.beginner {
  background: #d1fae5;
  color: #065f46;
}

.difficulty.intermediate {
  background: #fef3c7;
  color: #92400e;
}

.assignment-progress {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 120px;
}

.progress-text {
  font-size: 0.9rem;
  color: #333;
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
  font-size: 0.75rem;
  color: #999;
}

.assignment-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: #666;
  font-size: 0.9rem;
}

.status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  width: fit-content;
}

.status.pending {
  background: #e5e7eb;
  color: #374151;
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
  gap: 0.5rem;
}

.assignment-actions button {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  background: #eee;
}

.assignment-actions .delete-btn {
  background: #fee2e2;
  color: #dc2626;
}

.assignment-actions .delete-btn:hover {
  background: #fecaca;
}

/* Modal */
.modal {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: #fff;
  padding: 1.5rem;
  border-radius: 12px;
  width: 100%;
  max-width: 500px;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-content h3 { margin: 0 0 1rem; }

.modal-content form label {
  display: block;
  margin: 0.75rem 0 0.25rem;
  color: #666;
}

.modal-content form input,
.modal-content form select,
.modal-content form textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
  box-sizing: border-box;
}

.modal-content form textarea {
  resize: vertical;
  min-height: 60px;
}

.assignment-preview {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0;
}

.assignment-preview p {
  margin: 4px 0;
  color: #0369a1;
}

.modal-actions {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.modal-actions button {
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.modal-actions button[type="submit"] {
  background: #4a90d9;
  color: #fff;
}

.modal-actions button[type="submit"]:disabled {
  background: #93c5fd;
  cursor: not-allowed;
}
</style>