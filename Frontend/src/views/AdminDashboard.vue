<template>
  <div class="admin-dashboard">
    <header>
      <h1>Admin Dashboard</h1>
    </header>

    <nav class="tabs">
      <router-link to="/admin/dashboard" exact-active-class="active">Overview</router-link>
      <router-link to="/admin/users" active-class="active">Users</router-link>
      <router-link to="/admin/assignments" active-class="active">Assignments</router-link>
    </nav>

    <div v-if="loading" class="loading">Loading...</div>

    <div v-else class="stats-grid">
      <div class="stat-card">
        <h3>Total Users</h3>
        <span class="stat-value">{{ stats?.total_users || 0 }}</span>
      </div>
      <div class="stat-card">
        <h3>Total Sessions</h3>
        <span class="stat-value">{{ stats?.total_sessions || 0 }}</span>
      </div>
      <div class="stat-card">
        <h3>Total Reps</h3>
        <span class="stat-value">{{ stats?.total_reps || 0 }}</span>
      </div>
      <div class="stat-card">
        <h3>Active Today</h3>
        <span class="stat-value">{{ stats?.active_today || 0 }}</span>
      </div>
    </div>

    <div v-if="stats?.recent_sessions?.length" class="recent-section">
      <h2>Recent Sessions</h2>
      <table>
        <thead>
          <tr>
            <th>User</th>
            <th>Exercise</th>
            <th>Reps</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="session in stats.recent_sessions" :key="session.id">
            <td>{{ session.user_name }}</td>
            <td>{{ session.exercise_type }}</td>
            <td>{{ session.reps }}</td>
            <td>{{ formatDate(session.created_at) }}</td>
          </tr>
        </tbody>
      </table>
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
const stats = ref(null)

onMounted(async () => {
  const res = await adminStore.fetchStats()
  if (res.error) {
    router.push('/admin/login')
  } else {
    stats.value = res
  }
  loading.value = false
})

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
  })
}
</script>

<style scoped>
.admin-dashboard {
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

.loading {
  text-align: center;
  padding: 2rem;
  color: #666;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: #fff;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.stat-card h3 {
  margin: 0 0 0.5rem;
  color: #666;
  font-size: 0.9rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: bold;
  color: #1a1a2e;
}

.recent-section h2 {
  margin-bottom: 1rem;
  color: #1a1a2e;
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
</style>