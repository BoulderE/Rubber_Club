<template>
  <div class="admin-container">
    <!-- Header -->
    <div class="admin-header">
      <h1>🛡️ 管理員儀表板</h1>
      <button @click="logout" class="logout-btn">登出</button>
    </div>

    <!-- Stats Overview -->
    <div class="stats-overview">
      <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-number">{{ users.length }}</div>
        <div class="stat-label">總用戶數</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🏋️</div>
        <div class="stat-number">{{ totalWorkouts }}</div>
        <div class="stat-label">總運動次數</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">📅</div>
        <div class="stat-number">{{ activeToday }}</div>
        <div class="stat-label">今日活躍</div>
      </div>
    </div>

    <!-- User Management Section -->
    <div class="section-card">
      <div class="section-header">
        <h2>用戶管理</h2>
        <div class="search-box">
          <input 
            v-model="searchQuery" 
            type="text" 
            placeholder="搜尋用戶..."
            class="search-input"
          />
        </div>
      </div>

      <div class="table-container">
        <table class="user-table">
          <thead>
            <tr>
              <th>用戶</th>
              <th>運動次數</th>
              <th>總次數</th>
              <th>平均流暢度</th>
              <th>最近活動</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in filteredUsers" :key="user.user_id">
              <td>
                <div class="user-info">
                  <div class="user-avatar">{{ getInitial(user.display_name) }}</div>
                  <div class="user-details">
                    <div class="user-name">{{ user.display_name || '未命名用戶' }}</div>
                    <div class="user-id">ID: {{ user.user_id.slice(0, 8) }}...</div>
                  </div>
                </div>
              </td>
              <td>
                <span class="badge">{{ user.total_workouts || 0 }}</span>
              </td>
              <td>{{ user.total_reps || 0 }}</td>
              <td>
                <div class="smoothness-bar">
                  <div 
                    class="smoothness-fill" 
                    :style="{ width: `${user.avg_smoothness || 0}%` }"
                  ></div>
                  <span>{{ user.avg_smoothness || 0 }}%</span>
                </div>
              </td>
              <td>
                <span class="last-active">{{ formatLastActive(user.last_active) }}</span>
              </td>
              <td>
                <button @click="viewUserDetails(user)" class="action-btn view-btn">
                  查看詳情
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="filteredUsers.length === 0" class="no-data">
        沒有找到符合條件的用戶
      </div>
    </div>

    <!-- User Detail Modal -->
    <div v-if="showUserModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal-content">
        <div class="modal-header">
          <div class="modal-user-info">
            <div class="modal-avatar">{{ getInitial(selectedUser?.display_name) }}</div>
            <div>
              <h2>{{ selectedUser?.display_name || '未命名用戶' }}</h2>
              <p class="modal-user-id">ID: {{ selectedUser?.user_id }}</p>
            </div>
          </div>
          <button @click="closeModal" class="close-btn">✕</button>
        </div>

        <!-- Time Tabs -->
        <div class="time-tabs">
          <button 
            v-for="tab in timeTabs" 
            :key="tab.value"
            :class="['tab-btn', { active: activeTab === tab.value }]"
            @click="activeTab = tab.value"
          >
            {{ tab.label }}
          </button>
        </div>

        <!-- Loading State -->
        <div v-if="loadingHistory" class="loading-state">
          <div class="spinner"></div>
          <p>載入中...</p>
        </div>

        <!-- User Stats -->
        <div v-else class="modal-body">
          <div class="user-stats-cards">
            <div class="stat-card-sm">
              <div class="stat-icon-sm">🏋️</div>
              <div class="stat-number-sm">{{ userCurrentStats.total_workouts }}</div>
              <div class="stat-label-sm">運動次數</div>
            </div>
            <div class="stat-card-sm">
              <div class="stat-icon-sm">🔥</div>
              <div class="stat-number-sm">{{ userCurrentStats.total_reps }}</div>
              <div class="stat-label-sm">總次數</div>
            </div>
            <div class="stat-card-sm">
              <div class="stat-icon-sm">✨</div>
              <div class="stat-number-sm">{{ userCurrentStats.avg_smoothness }}%</div>
              <div class="stat-label-sm">平均流暢度</div>
            </div>
            <div class="stat-card-sm">
              <div class="stat-icon-sm">⏱️</div>
              <div class="stat-number-sm">{{ formatTotalDuration(userCurrentStats.total_duration) }}</div>
              <div class="stat-label-sm">運動時長</div>
            </div>
          </div>

          <!-- Activity Rings -->
          <div class="activity-rings-section" v-if="activeTab !== 'all'">
            <h3>活動目標達成</h3>
            <div class="rings-container">
              <svg class="activity-rings" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="80" class="ring-bg" />
                <circle cx="100" cy="100" r="60" class="ring-bg" />
                <circle cx="100" cy="100" r="40" class="ring-bg" />
                <circle 
                  cx="100" cy="100" r="80" 
                  class="ring-progress ring-workout"
                  :style="{ strokeDasharray: `${workoutProgress * 5.02} 502` }"
                />
                <circle 
                  cx="100" cy="100" r="60" 
                  class="ring-progress ring-reps"
                  :style="{ strokeDasharray: `${repsProgress * 3.77} 377` }"
                />
                <circle 
                  cx="100" cy="100" r="40" 
                  class="ring-progress ring-smoothness"
                  :style="{ strokeDasharray: `${smoothnessProgress * 2.51} 251` }"
                />
              </svg>
              <div class="rings-legend">
                <div class="legend-item">
                  <span class="legend-dot workout"></span>
                  <span>運動次數 {{ Math.round(workoutProgress) }}%</span>
                </div>
                <div class="legend-item">
                  <span class="legend-dot reps"></span>
                  <span>總次數 {{ Math.round(repsProgress) }}%</span>
                </div>
                <div class="legend-item">
                  <span class="legend-dot smoothness"></span>
                  <span>流暢度 {{ Math.round(smoothnessProgress) }}%</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Bar Chart -->
          <div class="chart-section">
            <h3>{{ chartTitle }}</h3>
            <div class="bar-chart">
              <div 
                v-for="(bar, index) in chartData" 
                :key="index"
                class="bar-wrapper"
              >
                <div class="bar-value">{{ bar.value }}</div>
                <div 
                  class="bar" 
                  :style="{ height: `${(bar.value / maxChartValue) * 100}%` }"
                ></div>
                <div class="bar-label">{{ bar.label }}</div>
              </div>
            </div>
          </div>

          <!-- Exercise Breakdown -->
          <div class="exercise-breakdown" v-if="exerciseBreakdown.length > 0">
            <h3>運動類型分佈</h3>
            <div class="breakdown-list">
              <div 
                v-for="(item, index) in exerciseBreakdown" 
                :key="index"
                class="breakdown-item"
              >
                <div class="breakdown-info">
                  <span class="breakdown-name">{{ item.name }}</span>
                  <span class="breakdown-count">{{ item.count }} 次</span>
                </div>
                <div class="breakdown-bar-bg">
                  <div 
                    class="breakdown-bar" 
                    :style="{ 
                      width: `${(item.count / maxExerciseCount) * 100}%`,
                      backgroundColor: exerciseColors[index % exerciseColors.length]
                    }"
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Records List -->
          <div class="records-section">
            <h3>{{ recordsTitle }}</h3>
            <div v-if="userFilteredRecords.length === 0" class="no-records">
              {{ noRecordsMessage }}
            </div>
            <div v-else class="records-list">
              <div v-for="(group, date) in userGroupedRecords" :key="date" class="record-group">
                <div class="group-header">
                  <span class="group-date">{{ formatGroupDate(date) }}</span>
                  <span class="group-summary">{{ getGroupSummary(group) }}</span>
                </div>
                <div class="record-item" v-for="record in group" :key="record.id">
                  <div class="record-left">
                    <div class="record-exercise">{{ record.exercise_name }}</div>
                    <div class="record-time">{{ formatTime(record.created_at) }}</div>
                  </div>
                  <div class="record-right">
                    <div class="record-stat">
                      <span class="stat-value">{{ record.rep_count }}</span>
                      <span class="stat-unit">次</span>
                    </div>
                    <div class="record-stat">
                      <span class="stat-value">{{ record.smoothness }}%</span>
                      <span class="stat-unit">流暢度</span>
                    </div>
                    <div class="record-stat">
                      <span class="stat-value">{{ formatDuration(record.duration) }}</span>
                      <span class="stat-unit">時長</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAdminStore } from '@/stores/admin'
import { getUsers, getUserHistory } from '@/api/admin'

const router = useRouter()
const adminStore = useAdminStore()

const users = ref([])
const searchQuery = ref('')
const showUserModal = ref(false)
const selectedUser = ref(null)
const userRecords = ref([])
const loadingHistory = ref(false)
const activeTab = ref('week')

const timeTabs = [
  { label: '日', value: 'day' },
  { label: '週', value: 'week' },
  { label: '月', value: 'month' },
  { label: '全部', value: 'all' }
]

const exerciseColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

const goals = {
  day: { workouts: 1, reps: 50 },
  week: { workouts: 5, reps: 300 },
  month: { workouts: 20, reps: 1200 }
}

// Computed properties for overview stats
const totalWorkouts = computed(() => {
  return users.value.reduce((sum, user) => sum + (user.total_workouts || 0), 0)
})

const activeToday = computed(() => {
  const today = new Date().toDateString()
  return users.value.filter(user => {
    if (!user.last_active) return false
    return new Date(user.last_active).toDateString() === today
  }).length
})

const filteredUsers = computed(() => {
  if (!searchQuery.value) return users.value
  const query = searchQuery.value.toLowerCase()
  return users.value.filter(user => 
    (user.display_name?.toLowerCase().includes(query)) ||
    (user.user_id?.toLowerCase().includes(query))
  )
})

// User detail computed properties
const userFilteredRecords = computed(() => {
  if (!userRecords.value) return []
  
  const now = new Date()
  let startDate, endDate
  
  switch (activeTab.value) {
    case 'day': {
      startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1)
      break
    }
    case 'week': {
      const dayOfWeek = now.getDay()
      startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() - dayOfWeek)
      endDate = now
      break
    }
    case 'month': {
      startDate = new Date(now.getFullYear(), now.getMonth(), 1)
      endDate = now
      break
    }
    case 'all':
    default:
      return userRecords.value
  }
  
  return userRecords.value.filter(record => {
    const date = new Date(record.created_at)
    return date >= startDate && date < endDate
  })
})

const userCurrentStats = computed(() => {
  const filtered = userFilteredRecords.value
  if (filtered.length === 0) {
    return {
      total_workouts: 0,
      total_reps: 0,
      avg_smoothness: 0,
      total_duration: 0
    }
  }
  
  return {
    total_workouts: filtered.length,
    total_reps: filtered.reduce((sum, r) => sum + (r.rep_count || 0), 0),
    avg_smoothness: Math.round(
      filtered.reduce((sum, r) => sum + (r.smoothness || 0), 0) / filtered.length
    ),
    total_duration: filtered.reduce((sum, r) => sum + (r.duration || 0), 0)
  }
})

const workoutProgress = computed(() => {
  if (activeTab.value === 'all') return 0
  const goal = goals[activeTab.value]?.workouts || 1
  return Math.min((userCurrentStats.value.total_workouts / goal) * 100, 100)
})

const repsProgress = computed(() => {
  if (activeTab.value === 'all') return 0
  const goal = goals[activeTab.value]?.reps || 50
  return Math.min((userCurrentStats.value.total_reps / goal) * 100, 100)
})

const smoothnessProgress = computed(() => {
  return userCurrentStats.value.avg_smoothness || 0
})

const chartData = computed(() => {
  const data = []
  const now = new Date()
  
  switch (activeTab.value) {
    case 'day': {
      for (let i = 0; i < 24; i += 4) {
        const count = userFilteredRecords.value.filter(r => {
          const hour = new Date(r.created_at).getHours()
          return hour >= i && hour < i + 4
        }).length
        data.push({ label: `${i}時`, value: count })
      }
      break
    }
    case 'week': {
      const weekDays = ['日', '一', '二', '三', '四', '五', '六']
      for (let i = 0; i < 7; i++) {
        const dayRecords = userFilteredRecords.value.filter(r => {
          return new Date(r.created_at).getDay() === i
        })
        const reps = dayRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
        data.push({ label: weekDays[i], value: reps })
      }
      break
    }
    case 'month': {
      for (let week = 1; week <= 5; week++) {
        const weekRecords = userFilteredRecords.value.filter(r => {
          const date = new Date(r.created_at)
          const weekOfMonth = Math.ceil(date.getDate() / 7)
          return weekOfMonth === week
        })
        const reps = weekRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
        data.push({ label: `第${week}週`, value: reps })
      }
      break
    }
    case 'all':
    default: {
      for (let i = 5; i >= 0; i--) {
        const monthDate = new Date(now.getFullYear(), now.getMonth() - i, 1)
        const monthRecords = userRecords.value.filter(r => {
          const date = new Date(r.created_at)
          return date.getFullYear() === monthDate.getFullYear() && 
                 date.getMonth() === monthDate.getMonth()
        })
        const reps = monthRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
        data.push({ label: `${monthDate.getMonth() + 1}月`, value: reps })
      }
      break
    }
  }
  
  return data
})

const maxChartValue = computed(() => {
  const max = Math.max(...chartData.value.map(d => d.value))
  return max || 1
})

const chartTitle = computed(() => {
  switch (activeTab.value) {
    case 'day': return '今日運動分佈'
    case 'week': return '本週運動統計'
    case 'month': return '本月運動統計'
    case 'all':
    default: return '歷史運動趨勢'
  }
})

const exerciseBreakdown = computed(() => {
  const breakdown = {}
  userFilteredRecords.value.forEach(record => {
    const name = record.exercise_name || '未知'
    if (!breakdown[name]) breakdown[name] = 0
    breakdown[name]++
  })
  
  return Object.entries(breakdown)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5)
})

const maxExerciseCount = computed(() => {
  if (exerciseBreakdown.value.length === 0) return 1
  return exerciseBreakdown.value[0].count
})

const userGroupedRecords = computed(() => {
  const groups = {}
  userFilteredRecords.value.forEach(record => {
    const date = new Date(record.created_at).toDateString()
    if (!groups[date]) groups[date] = []
    groups[date].push(record)
  })
  return groups
})

const recordsTitle = computed(() => {
  switch (activeTab.value) {
    case 'day': return '今日記錄'
    case 'week': return '本週記錄'
    case 'month': return '本月記錄'
    case 'all':
    default: return '所有記錄'
  }
})

const noRecordsMessage = computed(() => {
  switch (activeTab.value) {
    case 'day': return '今天還沒有運動記錄'
    case 'week': return '本週還沒有運動記錄'
    case 'month': return '本月還沒有運動記錄'
    case 'all':
    default: return '還沒有運動記錄'
  }
})

// Methods
const getInitial = (name) => {
  if (!name) return '?'
  return name.charAt(0).toUpperCase()
}

const formatLastActive = (dateStr) => {
  if (!dateStr) return '從未'
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  
  if (diffMins < 1) return '剛剛'
  if (diffMins < 60) return `${diffMins} 分鐘前`
  if (diffHours < 24) return `${diffHours} 小時前`
  if (diffDays < 7) return `${diffDays} 天前`
  return date.toLocaleDateString('zh-TW')
}

const formatGroupDate = (dateString) => {
  const date = new Date(dateString)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  
  if (date.toDateString() === today.toDateString()) return '今天'
  if (date.toDateString() === yesterday.toDateString()) return '昨天'
  
  return date.toLocaleDateString('zh-TW', {
    month: 'long',
    day: 'numeric',
    weekday: 'short'
  })
}

const formatTime = (isoString) => {
  if (!isoString) return ''
  return new Date(isoString).toLocaleTimeString('zh-TW', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const formatDuration = (seconds) => {
  if (!seconds) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const formatTotalDuration = (seconds) => {
  if (!seconds) return '0分'
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  if (hours > 0) return `${hours}時${mins}分`
  return `${mins}分`
}

const getGroupSummary = (group) => {
  const totalReps = group.reduce((sum, r) => sum + (r.rep_count || 0), 0)
  const avgSmoothness = Math.round(
    group.reduce((sum, r) => sum + (r.smoothness || 0), 0) / group.length
  )
  return `${group.length}次運動 · ${totalReps}次 · ${avgSmoothness}%流暢度`
}

const viewUserDetails = async (user) => {
  selectedUser.value = user
  showUserModal.value = true
  loadingHistory.value = true
  activeTab.value = 'week'
  
  try {
    const data = await getUserHistory(adminStore.token, user.user_id)
    userRecords.value = data.records || []
  } catch (error) {
    console.error('Failed to fetch user history:', error)
    userRecords.value = []
  } finally {
    loadingHistory.value = false
  }
}

const closeModal = () => {
  showUserModal.value = false
  selectedUser.value = null
  userRecords.value = []
}

const logout = () => {
  adminStore.logout()
  router.push('/admin/login')
}

onMounted(async () => {
  if (!adminStore.isAuthenticated) {
    router.push('/admin/login')
    return
  }
  
  try {
    const data = await getUsers(adminStore.token)
    users.value = data.users || []
  } catch (error) {
    console.error('Failed to fetch users:', error)
  }
})
</script>

<style scoped>
.admin-container {
  min-height: 100vh;
  background: #f5f5f7;
  padding: 20px;
}

.admin-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.admin-header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #1c1c1e;
}

.logout-btn {
  padding: 10px 20px;
  background: #ff3b30;
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.logout-btn:hover {
  background: #d63029;
}

/* Stats Overview */
.stats-overview {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: white;
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.stat-icon {
  font-size: 32px;
  margin-bottom: 8px;
}

.stat-number {
  font-size: 32px;
  font-weight: 700;
  color: #1c1c1e;
}

.stat-label {
  font-size: 14px;
  color: #8e8e93;
  margin-top: 4px;
}

/* Section Card */
.section-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #1c1c1e;
}

.search-input {
  padding: 10px 16px;
  border: 1px solid #e5e5ea;
  border-radius: 10px;
  font-size: 14px;
  width: 250px;
  transition: border-color 0.2s;
}

.search-input:focus {
  outline: none;
  border-color: #667eea;
}

/* User Table */
.table-container {
  overflow-x: auto;
}

.user-table {
  width: 100%;
  border-collapse: collapse;
}

.user-table th,
.user-table td {
  padding: 16px;
  text-align: left;
  border-bottom: 1px solid #e5e5ea;
}

.user-table th {
  font-size: 12px;
  font-weight: 600;
  color: #8e8e93;
  text-transform: uppercase;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 16px;
}

.user-name {
  font-weight: 600;
  color: #1c1c1e;
}

.user-id {
  font-size: 12px;
  color: #8e8e93;
}

.badge {
  display: inline-block;
  padding: 4px 12px;
  background: #e8f5e9;
  color: #2e7d32;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
}

.smoothness-bar {
  display: flex;
  align-items: center;
  gap: 8px;
}

.smoothness-bar > div {
  width: 80px;
  height: 8px;
  background: #e5e5ea;
  border-radius: 4px;
  overflow: hidden;
}

.smoothness-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 4px;
  transition: width 0.3s;
}

.last-active {
  font-size: 14px;
  color: #8e8e93;
}

.action-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.view-btn {
  background: #667eea;
  color: white;
}

.view-btn:hover {
  background: #5a6fd6;
  transform: translateY(-1px);
}

.no-data {
  text-align: center;
  padding: 40px;
  color: #8e8e93;
}

/* Modal Styles */
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
  padding: 20px;
}

.modal-content {
  background: #f5f5f7;
  border-radius: 20px;
  width: 100%;
  max-width: 700px;
  max-height: 90vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: white;
  border-radius: 20px 20px 0 0;
  border-bottom: 1px solid #e5e5ea;
  position: sticky;
  top: 0;
  z-index: 10;
}

.modal-user-info {
  display: flex;
  align-items: center;
  gap: 16px;
}

.modal-avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 20px;
}

.modal-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #1c1c1e;
  margin: 0;
}

.modal-user-id {
  font-size: 12px;
  color: #8e8e93;
  margin: 4px 0 0 0;
}

.close-btn {
  width: 36px;
  height: 36px;
  border: none;
  background: #e5e5ea;
  border-radius: 50%;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.close-btn:hover {
  background: #d1d1d6;
}

.modal-body {
  padding: 20px 24px;
}

/* Time Tabs in Modal */
.time-tabs {
  display: flex;
  background: #e5e5ea;
  border-radius: 10px;
  padding: 3px;
  margin: 16px 24px;
}

.tab-btn {
  flex: 1;
  padding: 10px 16px;
  border: none;
  background: transparent;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn.active {
  background: white;
  color: #000;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Loading State */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e5e5ea;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-state p {
  margin-top: 16px;
  color: #8e8e93;
}

/* User Stats Cards */
.user-stats-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.stat-card-sm {
  background: white;
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stat-icon-sm {
  font-size: 20px;
  margin-bottom: 6px;
}

.stat-number-sm {
  font-size: 20px;
  font-weight: 700;
  color: #1c1c1e;
}

.stat-label-sm {
  font-size: 11px;
  color: #8e8e93;
  margin-top: 4px;
}

/* Activity Rings */
.activity-rings-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.activity-rings-section h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #1c1c1e;
}

.rings-container {
  display: flex;
  align-items: center;
  gap: 24px;
}

.activity-rings {
  width: 120px;
  height: 120px;
  transform: rotate(-90deg);
}

.ring-bg {
  fill: none;
  stroke: #e5e5ea;
  stroke-width: 12;
}

.ring-progress {
  fill: none;
  stroke-width: 12;
  stroke-linecap: round;
  transition: stroke-dasharray 0.5s ease;
}

.ring-workout { stroke: #ff3b30; }
.ring-reps { stroke: #30d158; }
.ring-smoothness { stroke: #5ac8fa; }

.rings-legend {
  flex: 1;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 13px;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.legend-dot.workout { background: #ff3b30; }
.legend-dot.reps { background: #30d158; }
.legend-dot.smoothness { background: #5ac8fa; }

/* Chart Section */
.chart-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.chart-section h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #1c1c1e;
}

.bar-chart {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  height: 120px;
  padding: 0 10px;
}

.bar-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
}

.bar-value {
  font-size: 10px;
  font-weight: 600;
  color: #8e8e93;
  margin-bottom: 4px;
}

.bar {
  width: 20px;
  background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px 4px 0 0;
  min-height: 4px;
  transition: height 0.3s ease;
}

.bar-label {
  font-size: 10px;
  color: #8e8e93;
  margin-top: 6px;
}

/* Exercise Breakdown */
.exercise-breakdown {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.exercise-breakdown h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #1c1c1e;
}

.breakdown-item {
  margin-bottom: 12px;
}

.breakdown-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.breakdown-name {
  font-size: 13px;
  color: #1c1c1e;
}

.breakdown-count {
  font-size: 13px;
  color: #8e8e93;
}

.breakdown-bar-bg {
  height: 6px;
  background: #e5e5ea;
  border-radius: 3px;
  overflow: hidden;
}

.breakdown-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

/* Records Section */
.records-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.records-section h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #1c1c1e;
}

.no-records {
  text-align: center;
  color: #8e8e93;
  padding: 30px 20px;
  font-size: 14px;
}

.record-group {
  margin-bottom: 16px;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e5e5ea;
}

.group-date {
  font-size: 14px;
  font-weight: 600;
  color: #1c1c1e;
}

.group-summary {
  font-size: 11px;
  color: #8e8e93;
}

.record-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #f5f5f7;
  border-radius: 10px;
  margin-bottom: 8px;
}

.record-left {
  flex: 1;
}

.record-exercise {
  font-weight: 600;
  font-size: 14px;
  color: #1c1c1e;
  margin-bottom: 2px;
}

.record-time {
  font-size: 11px;
  color: #8e8e93;
}

.record-right {
  display: flex;
  gap: 14px;
}

.record-stat {
  text-align: center;
}

.record-stat .stat-value {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: #1c1c1e;
}

.record-stat .stat-unit {
  font-size: 9px;
  color: #8e8e93;
}

/* Responsive */
@media (max-width: 768px) {
  .stats-overview {
    grid-template-columns: 1fr;
  }
  
  .user-stats-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .rings-container {
    flex-direction: column;
    text-align: center;
  }
  
  .rings-legend {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  
  .legend-item {
    margin-bottom: 0;
  }
  
  .user-table th:nth-child(4),
  .user-table td:nth-child(4) {
    display: none;
  }
  
  .record-right {
    gap: 10px;
  }
}

@media (max-width: 480px) {
  .modal-content {
    border-radius: 16px 16px 0 0;
    max-height: 95vh;
    margin-top: auto;
  }
  
  .user-stats-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .search-input {
    width: 150px;
  }
}
</style>