<template>
  <div class="admin-users">
    <header>
      <h1>Admin Dashboard</h1>
    </header>

    <nav class="tabs">
      <router-link to="/admin/dashboard" exact-active-class="active">Overview</router-link>
      <router-link to="/admin/users" active-class="active">Users</router-link>
      <router-link to="/admin/assignments" active-class="active">Assignments</router-link>
    </nav>

    <div class="content">
      <h2>Users</h2>
      
      <div v-if="loading" class="loading">Loading...</div>

      <div v-else class="users-list">
        <div v-for="user in users" :key="user.id" class="user-card">
          <div class="user-info">
            <h3>{{ user.name }}</h3>
            <span class="user-pin">PIN: {{ user.pin }}</span>
            <span class="user-role" :class="user.role">{{ user.role }}</span>
          </div>
          <div class="user-stats">
            <span>{{ user.total_sessions || 0 }} sessions</span>
            <span>{{ user.total_reps || 0 }} reps</span>
          </div>
          <div class="user-actions">
            <button @click="viewHistory(user)">History</button>
            <button @click="openAssignModal(user)" class="assign-btn">Assign</button>
          </div>
        </div>
      </div>
    </div>

    <!-- History Modal -->
    <div v-if="showHistory" class="modal" @click.self="showHistory = false">
      <div class="modal-content">
        <h3>{{ selectedUser?.name }}'s History</h3>
        <div v-if="historyLoading" class="loading">Loading...</div>
        <table v-else-if="history.length">
          <thead>
            <tr><th>Exercise</th><th>Reps</th><th>Date</th></tr>
          </thead>
          <tbody>
            <tr v-for="h in history" :key="h.id">
              <td>{{ h.exercise_name }}</td>
              <td>{{ h.rep_count }}</td>
              <td>{{ formatDate(h.created_at) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else>No history found.</p>
        <button @click="showHistory = false">Close</button>
      </div>
    </div>

    <!-- Assign Modal (Updated) -->
    <div v-if="showAssign" class="modal" @click.self="showAssign = false">
      <div class="modal-content">
        <h3>Assign Exercise to {{ selectedUser?.name }}</h3>
        <form @submit.prevent="submitAssignment">
          <label>Exercise Type</label>
          <select v-model="assignForm.exercise_key" required>
            <option value="bicep_curl">Bicep Curl</option>
            <option value="squat">Squat</option>
            <option value="pushup">Push-up</option>
            <option value="lunge">Lunge</option>
          </select>

          <label>Difficulty</label>
          <select v-model="assignForm.difficulty">
            <option value="beginner">Beginner (10 reps per set)</option>
            <option value="intermediate">Intermediate (15 reps per set)</option>
          </select>

          <label>Number of Sets</label>
          <input v-model.number="assignForm.target_sets" type="number" min="1" max="10" required />

          <!-- Preview (auto-calculated, read-only) -->
          <div class="assignment-preview">
            <p><strong>Preview:</strong></p>
            <p>{{ repsPerSet }} reps × {{ assignForm.target_sets }} sets = {{ totalReps }} total reps</p>
          </div>

          <label>Due Date (optional)</label>
          <input v-model="assignForm.due_date" type="date" />

          <label>Notes (optional)</label>
          <textarea v-model="assignForm.notes"></textarea>

          <div class="modal-actions">
            <button type="button" @click="showAssign = false">Cancel</button>
            <button type="submit" :disabled="assigning">
              {{ assigning ? 'Assigning...' : 'Assign' }}
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
const users = ref([])
const showHistory = ref(false)
const showAssign = ref(false)
const selectedUser = ref(null)
const history = ref([])
const historyLoading = ref(false)
const assigning = ref(false)

const assignForm = ref({
  exercise_key: 'bicep_curl',
  difficulty: 'beginner',
  target_sets: 3,
  due_date: '',
  notes: ''
})

// Auto-calculate reps based on difficulty
const repsPerSet = computed(() => {
  return assignForm.value.difficulty === 'beginner' ? 10 : 15
})

// Calculate total reps for preview
const totalReps = computed(() => {
  return repsPerSet.value * assignForm.value.target_sets
})

onMounted(async () => {
  const res = await adminStore.fetchUsers()
  if (res.error) {
    router.push('/admin/login')
  } else {
    users.value = res.users || []
  }
  loading.value = false
})

async function viewHistory(user) {
  selectedUser.value = user
  showHistory.value = true
  historyLoading.value = true
  const res = await adminStore.fetchUserHistory(user.id)
  history.value = res || []
  historyLoading.value = false
}

function openAssignModal(user) {
  selectedUser.value = user
  assignForm.value = {
    exercise_key: 'bicep_curl',
    difficulty: 'beginner',
    target_sets: 3,
    due_date: '',
    notes: ''
  }
  showAssign.value = true
}

async function submitAssignment() {
  assigning.value = true
  
  const res = await adminStore.assignExercise({
    user_id: selectedUser.value.id,
    exercise_key: assignForm.value.exercise_key,
    difficulty: assignForm.value.difficulty,
    target_sets: assignForm.value.target_sets,
    due_date: assignForm.value.due_date || null,
    notes: assignForm.value.notes
  })
  
  if (res.success) {
    showAssign.value = false
    alert('Exercise assigned successfully!')
  } else {
    alert(res.error || 'Failed to assign')
  }
  assigning.value = false
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString()
}
</script>

<style scoped>
.admin-users {
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

.logout-btn {
  padding: 8px 16px;
  background: #e74c3c;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

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

.users-list {
  display: grid;
  gap: 1rem;
}

.user-card {
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

.user-info h3 { margin: 0; }
.user-pin { color: #666; font-size: 0.9rem; margin-left: 1rem; }
.user-role {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  margin-left: 0.5rem;
}
.user-role.admin { background: #e74c3c; color: #fff; }
.user-role.user { background: #ddd; }

.user-stats {
  display: flex;
  gap: 1rem;
  color: #666;
}

.user-actions {
  display: flex;
  gap: 0.5rem;
}

.user-actions button {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  background: #eee;
}

.user-actions .assign-btn {
  background: #4a90d9;
  color: #fff;
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

.modal-content table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
}

.modal-content th, .modal-content td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

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
</style>