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

      <table v-else-if="assignments.length">
        <thead>
          <tr>
            <th>User</th>
            <th>Exercise</th>
            <th>Difficulty</th>
            <th>Progress</th>
            <th>Due</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in assignments" :key="a.id">
            <td>{{ a.user_name }}</td>
            <td>{{ a.exercise_name }}</td>
            <td>
              <span class="difficulty" :class="a.difficulty">
                {{ a.difficulty === 'beginner' ? 'Beginner' : 'Intermediate' }}
              </span>
            </td>
            <td>
              <div class="progress-info">
                <span>{{ a.completed_sets }} / {{ a.target_sets }} sets</span>
                <div class="progress-bar">
                  <div 
                    class="progress-fill" 
                    :style="{ width: (a.completed_sets / a.target_sets * 100) + '%' }"
                  ></div>
                </div>
                <span class="reps-detail">
                  ({{ a.target_reps }} reps/set)
                </span>
              </div>
            </td>
            <td>{{ formatDate(a.due_date) }}</td>
            <td>
              <span class="status" :class="a.status">{{ formatStatus(a.status) }}</span>
            </td>
            <td>
              <button @click="editAssignment(a)">Edit</button>
              <button @click="deleteAssignment(a.id)" class="delete-btn">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>

      <p v-else>No assignments found.</p>
    </div>

    <!-- Edit Modal (Updated) -->
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

          <!-- Preview -->
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

// Auto-calculate reps for edit preview
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
  
  // Send to backend - no target_reps, backend calculates from difficulty
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
.progress-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
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
  color: #6b7280;
}

.difficulty {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.difficulty.beginner {
  background: #d1fae5;
  color: #065f46;
}

.difficulty.intermediate {
  background: #fef3c7;
  color: #92400e;
}

.status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
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
</style>