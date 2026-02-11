<template>
  <div class="admin-assignments">
    <header>
      <h1>Admin Dashboard</h1>
      <button @click="logout" class="logout-btn">Logout</button>
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
            <th>Target</th>
            <th>Completed</th>
            <th>Due</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in assignments" :key="a.id">
            <td>{{ a.user_name }}</td>
            <td>{{ a.exercise_type }}</td>
            <td>{{ a.target_reps }}</td>
            <td>{{ a.completed_reps }}</td>
            <td>{{ formatDate(a.due_date) }}</td>
            <td>
              <span class="status" :class="a.status">{{ a.status }}</span>
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

    <!-- Edit Modal -->
    <div v-if="showEdit" class="modal" @click.self="showEdit = false">
      <div class="modal-content">
        <h3>Edit Assignment</h3>
        <form @submit.prevent="submitEdit">
          <label>Target Reps</label>
          <input v-model.number="editForm.target_reps" type="number" min="1" required />

          <label>Due Date</label>
          <input v-model="editForm.due_date" type="date" required />

          <label>Status</label>
          <select v-model="editForm.status">
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
          </select>

          <label>Notes</label>
          <textarea v-model="editForm.notes"></textarea>

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
import { ref, onMounted } from 'vue'
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
  target_reps: 0,
  due_date: '',
  status: 'pending',
  notes: ''
})

onMounted(async () => {
  const res = await adminStore.fetchAssignments()
  if (res.error) {
    router.push('/admin/login')
  } else {
    assignments.value = res.assignments || []
  }
  loading.value = false
})

function logout() {
  adminStore.logout()
  router.push('/admin/login')
}

function editAssignment(a) {
  editingId.value = a.id
  editForm.value = {
    target_reps: a.target_reps,
    due_date: a.due_date?.split('T')[0] || '',
    status: a.status,
    notes: a.notes || ''
  }
  showEdit.value = true
}

async function submitEdit() {
  saving.value = true
  const res = await adminStore.updateAssignment(editingId.value, editForm.value)
  if (res.success) {
    showEdit.value = false
    const refreshed = await adminStore.fetchAssignments()
    assignments.value = refreshed.assignments || []
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

.content h2 { margin-bottom: 1rem; }

.loading {
  text-align: center;
  padding: 2rem;
  color: #666;
}

table {
  width: 100%;
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  border-collapse: collapse;
}

th, td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

th { background: #f8f9fa; color: #666; }

.status {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.85rem;
}
.status.pending { background: #ffeaa7; }
.status.in_progress { background: #74b9ff; color: #fff; }
.status.completed { background: #55efc4; }

td button {
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-right: 4px;
  background: #eee;
}

td .delete-btn {
  background: #e74c3c;
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
  max-width: 400px;
}

.modal-content h3 { margin: 0 0 1rem; }

.modal-content label {
  display: block;
  margin: 0.75rem 0 0.25rem;
  color: #666;
}

.modal-content input,
.modal-content select,
.modal-content textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
  box-sizing: border-box;
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