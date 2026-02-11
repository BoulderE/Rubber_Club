<template>
  <div class="history-container">
    <h1>運動歷史記錄</h1>
    
    <!-- 時間範圍切換 -->
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
    
    <!-- 統計卡片 -->
    <div class="stats-cards" v-if="currentStats">
      <div class="stat-card">
        <div class="stat-icon">🏋️</div>
        <div class="stat-number">{{ currentStats.total_workouts }}</div>
        <div class="stat-label">運動次數</div>
        <div class="stat-trend" v-if="currentStats.workout_trend !== undefined">
          <span :class="currentStats.workout_trend >= 0 ? 'trend-up' : 'trend-down'">
            {{ currentStats.workout_trend >= 0 ? '↑' : '↓' }} {{ Math.abs(currentStats.workout_trend) }}%
          </span>
          <span class="trend-label">vs 上期</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🔥</div>
        <div class="stat-number">{{ currentStats.total_reps }}</div>
        <div class="stat-label">總次數</div>
        <div class="stat-trend" v-if="currentStats.reps_trend !== undefined">
          <span :class="currentStats.reps_trend >= 0 ? 'trend-up' : 'trend-down'">
            {{ currentStats.reps_trend >= 0 ? '↑' : '↓' }} {{ Math.abs(currentStats.reps_trend) }}%
          </span>
          <span class="trend-label">vs 上期</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⏱️</div>
        <div class="stat-number">{{ formatTotalDuration(currentStats.total_duration) }}</div>
        <div class="stat-label">運動時長</div>
      </div>
    </div>
    
    <!-- 活動環 (類似 Apple Fitness) -->
    <div class="activity-rings-section" v-if="activeTab !== 'all'">
      <h2>活動目標</h2>
      <div class="rings-container">
        <svg class="activity-rings" viewBox="0 0 200 200">
          <!-- 背景環 -->
          <circle cx="100" cy="100" r="80" class="ring-bg" />
          <circle cx="100" cy="100" r="60" class="ring-bg" />
          <circle cx="100" cy="100" r="40" class="ring-bg" />
          
          <!-- 進度環 -->
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
    
    <!-- 圖表區域 -->
    <div class="charts-section">
      <h2>{{ chartTitle }}</h2>
      
      <!-- 長條圖 -->
      <div class="bar-chart-container">
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
      
      <!-- 運動類型分佈 -->
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
    </div>
    
    <!-- 記錄列表 -->
    <div class="records-list">
      <h2>{{ recordsTitle }}</h2>
      <div v-if="filteredRecords.length === 0" class="no-records">
        {{ noRecordsMessage }}
      </div>
      <div v-else>
        <!-- 按日期分組顯示 -->
        <div v-for="(group, date) in groupedRecords" :key="date" class="record-group">
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
    
    <button @click="$router.push('/')" class="back-btn">返回首頁</button>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
const authStore = useAuthStore()

const records = ref([])
const activeTab = ref('week')

const timeTabs = [
  { label: '日', value: 'day' },
  { label: '週', value: 'week' },
  { label: '月', value: 'month' },
  { label: '全部', value: 'all' }
]

const exerciseColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

// 目標設定 (可以讓用戶自訂)
const goals = {
  day: { workouts: 1, reps: 50 },
  week: { workouts: 5, reps: 300 },
  month: { workouts: 20, reps: 1200 }
}

// 計算當前時間範圍的記錄
const filteredRecords = computed(() => {
  if (!records.value) return []
  
  const now = new Date()
  let startDate
  let endDate
  
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
      return records.value
  }
  
  return records.value.filter(record => {
    const date = new Date(record.created_at)
    return date >= startDate && date < endDate
  })
})

// 計算當前統計
const currentStats = computed(() => {
  const filtered = filteredRecords.value
  if (filtered.length === 0) {
    return {
      total_workouts: 0,
      total_reps: 0,
      avg_smoothness: 0,
      total_duration: 0
    }
  }
  
  const total_workouts = filtered.length
  const total_reps = filtered.reduce((sum, r) => sum + (r.rep_count || 0), 0)
  const avg_smoothness = Math.round(
    filtered.reduce((sum, r) => sum + (r.smoothness || 0), 0) / filtered.length
  )
  const total_duration = filtered.reduce((sum, r) => sum + (r.duration || 0), 0)
  
  // 計算趨勢 (與上一期比較)
  const previousRecords = getPreviousPeriodRecords()
  const prev_workouts = previousRecords.length
  const prev_reps = previousRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
  
  const workout_trend = prev_workouts > 0 
    ? Math.round(((total_workouts - prev_workouts) / prev_workouts) * 100)
    : undefined
  const reps_trend = prev_reps > 0 
    ? Math.round(((total_reps - prev_reps) / prev_reps) * 100)
    : undefined
  
  return {
    total_workouts,
    total_reps,
    avg_smoothness,
    total_duration,
    workout_trend,
    reps_trend
  }
})

// 獲取上一期的記錄
const getPreviousPeriodRecords = () => {
  if (!records.value || activeTab.value === 'all') return []
  
  const now = new Date()
  let startDate, endDate
  
  switch (activeTab.value) {
    case 'day': {
      endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      startDate = new Date(endDate)
      startDate.setDate(startDate.getDate() - 1)
      break
    }
    case 'week': {
      const dayOfWeek = now.getDay()
      endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() - dayOfWeek)
      startDate = new Date(endDate)
      startDate.setDate(startDate.getDate() - 7)
      break
    }
    case 'month': {
      endDate = new Date(now.getFullYear(), now.getMonth(), 1)
      startDate = new Date(now.getFullYear(), now.getMonth() - 1, 1)
      break
    }
    default:
      return []
  }
  
  return records.value.filter(record => {
    const date = new Date(record.created_at)
    return date >= startDate && date < endDate
  })
}

// 活動環進度
const workoutProgress = computed(() => {
  if (activeTab.value === 'all') return 0
  const goal = goals[activeTab.value]?.workouts || 1
  return Math.min((currentStats.value.total_workouts / goal) * 100, 100)
})

const repsProgress = computed(() => {
  if (activeTab.value === 'all') return 0
  const goal = goals[activeTab.value]?.reps || 50
  return Math.min((currentStats.value.total_reps / goal) * 100, 100)
})

const smoothnessProgress = computed(() => {
  return currentStats.value.avg_smoothness || 0
})

// 圖表數據
const chartData = computed(() => {
  const data = []
  const now = new Date()
  
  switch (activeTab.value) {
    case 'day': {
      // 24小時分佈
      for (let i = 0; i < 24; i += 4) {
        const count = filteredRecords.value.filter(r => {
          const hour = new Date(r.created_at).getHours()
          return hour >= i && hour < i + 4
        }).length
        data.push({ label: `${i}時`, value: count })
      }
      break
    }
    case 'week': {
      // 7天分佈
      const weekDays = ['日', '一', '二', '三', '四', '五', '六']
      for (let i = 0; i < 7; i++) {
        const dayRecords = filteredRecords.value.filter(r => {
          return new Date(r.created_at).getDay() === i
        })
        const reps = dayRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
        data.push({ label: weekDays[i], value: reps })
      }
      break
    }
    case 'month': {
      // 按週分佈
      for (let week = 1; week <= 5; week++) {
        const weekRecords = filteredRecords.value.filter(r => {
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
      // 最近6個月
      for (let i = 5; i >= 0; i--) {
        const monthDate = new Date(now.getFullYear(), now.getMonth() - i, 1)
        const monthRecords = records.value.filter(r => {
          const date = new Date(r.created_at)
          return date.getFullYear() === monthDate.getFullYear() && 
                 date.getMonth() === monthDate.getMonth()
        })
        const reps = monthRecords.reduce((sum, r) => sum + (r.rep_count || 0), 0)
        data.push({ 
          label: `${monthDate.getMonth() + 1}月`, 
          value: reps 
        })
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
    case 'day': 
      return '今日運動分佈'
    case 'week': 
      return '本週運動統計'
    case 'month': 
      return '本月運動統計'
    case 'all':
    default:
      return '歷史運動趨勢'
  }
})

// 運動類型分佈
const exerciseBreakdown = computed(() => {
  const breakdown = {}
  filteredRecords.value.forEach(record => {
    const name = record.exercise_name || '未知'
    if (!breakdown[name]) {
      breakdown[name] = 0
    }
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

// 按日期分組記錄
const groupedRecords = computed(() => {
  const groups = {}
  filteredRecords.value.forEach(record => {
    const date = new Date(record.created_at).toDateString()
    if (!groups[date]) {
      groups[date] = []
    }
    groups[date].push(record)
  })
  return groups
})

const recordsTitle = computed(() => {
  switch (activeTab.value) {
    case 'day': 
      return '今日記錄'
    case 'week': 
      return '本週記錄'
    case 'month': 
      return '本月記錄'
    case 'all':
    default:
      return '所有記錄'
  }
})

const noRecordsMessage = computed(() => {
  switch (activeTab.value) {
    case 'day': 
      return '今天還沒有運動記錄，開始運動吧！'
    case 'week': 
      return '本週還沒有運動記錄，開始運動吧！'
    case 'month': 
      return '本月還沒有運動記錄，開始運動吧！'
    case 'all':
    default:
      return '還沒有運動記錄，開始你的第一次運動吧！'
  }
})

onMounted(async () => {
  const userId = authStore.userId
  if (!userId) {
    console.error('未登入')
    return
  }
  
  try {
    // 獲取記錄列表
    const recordsRes = await fetch(`${API_BASE_URL}/api/records/${userId}`)
    const data = await recordsRes.json()
    records.value = data.records || []
  } catch (error) {
    console.error('獲取記錄失敗:', error)
  }
})

// 格式化函數
const formatGroupDate = (dateString) => {
  const date = new Date(dateString)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  
  if (date.toDateString() === today.toDateString()) {
    return '今天'
  } else if (date.toDateString() === yesterday.toDateString()) {
    return '昨天'
  }
  
  return date.toLocaleDateString('zh-TW', {
    month: 'long',
    day: 'numeric',
    weekday: 'short'
  })
}

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleTimeString('zh-TW', {
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
  if (hours > 0) {
    return `${hours}時${mins}分`
  }
  return `${mins}分`
}

const getGroupSummary = (group) => {
  const totalReps = group.reduce((sum, r) => sum + (r.rep_count || 0), 0)
  const avgSmoothness = Math.round(
    group.reduce((sum, r) => sum + (r.smoothness || 0), 0) / group.length
  )
  return `${group.length}次運動 · ${totalReps}次 · ${avgSmoothness}%流暢度`
}
</script>

<style scoped>
.history-container {
  padding: 20px;
  max-width: 600px;
  margin: 0 auto;
  background: #f5f5f7;
  min-height: 100vh;
}

h1 {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 20px;
}

/* 時間標籤切換 */
.time-tabs {
  display: flex;
  background: #e5e5ea;
  border-radius: 10px;
  padding: 3px;
  margin-bottom: 20px;
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

/* 統計卡片 */
.stats-cards {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
}

.stat-card {
  flex: 1;
  background: white;
  border-radius: 16px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.stat-icon {
  font-size: 24px;
  margin-bottom: 8px;
}

.stat-number {
  font-size: 24px;
  font-weight: 700;
  color: #1c1c1e;
}

.stat-label {
  font-size: 11px;
  color: #8e8e93;
  margin-top: 4px;
}

.stat-trend {
  margin-top: 8px;
  font-size: 11px;
}

.trend-up {
  color: #34c759;
  font-weight: 600;
}

.trend-down {
  color: #ff3b30;
  font-weight: 600;
}

.trend-label {
  color: #8e8e93;
  margin-left: 4px;
}

/* 活動環 */
.activity-rings-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.activity-rings-section h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
}

.rings-container {
  display: flex;
  align-items: center;
  gap: 24px;
}

.activity-rings {
  width: 140px;
  height: 140px;
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

.ring-workout {
  stroke: #ff3b30;
}

.ring-reps {
  stroke: #30d158;
}

.ring-smoothness {
  stroke: #5ac8fa;
}

.rings-legend {
  flex: 1;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-size: 14px;
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-dot.workout {
  background: #ff3b30;
}

.legend-dot.reps {
  background: #30d158;
}

.legend-dot.smoothness {
  background: #5ac8fa;
}

/* 圖表區域 */
.charts-section {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.charts-section h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
}

.bar-chart-container {
  margin-bottom: 24px;
}

.bar-chart {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  height: 150px;
  padding: 0 10px;
}

.bar-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
}

.bar-value {
  font-size: 11px;
  font-weight: 600;
  color: #8e8e93;
  margin-bottom: 4px;
}

.bar {
  width: 24px;
  background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
  border-radius: 6px 6px 0 0;
  min-height: 4px;
  transition: height 0.3s ease;
}

.bar-label {
  font-size: 11px;
  color: #8e8e93;
  margin-top: 8px;
}

/* 運動類型分佈 */
.exercise-breakdown h3 {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
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
  font-size: 14px;
  color: #1c1c1e;
}

.breakdown-count {
  font-size: 14px;
  color: #8e8e93;
}

.breakdown-bar-bg {
  height: 8px;
  background: #e5e5ea;
  border-radius: 4px;
  overflow: hidden;
}

.breakdown-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

/* 記錄列表 */
.records-list {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.records-list h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
}

.no-records {
  text-align: center;
  color: #8e8e93;
  padding: 40px 20px;
  font-size: 15px;
}

.record-group {
  margin-bottom: 20px;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e5e5ea;
}

.group-date {
  font-size: 15px;
  font-weight: 600;
  color: #1c1c1e;
}

.group-summary {
  font-size: 12px;
  color: #8e8e93;
}

.record-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #f5f5f7;
  border-radius: 12px;
  margin-bottom: 8px;
}

.record-left {
  flex: 1;
}

.record-exercise {
  font-weight: 600;
  font-size: 15px;
  color: #1c1c1e;
  margin-bottom: 4px;
}

.record-time {
  font-size: 12px;
  color: #8e8e93;
}

.record-right {
  display: flex;
  gap: 16px;
}

.record-stat {
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 15px;
  font-weight: 600;
  color: #1c1c1e;
}

.stat-unit {
  font-size: 10px;
  color: #8e8e93;
}

/* 返回按鈕 */
.back-btn {
  width: 100%;
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 14px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.back-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
}

.back-btn:active {
  transform: translateY(0);
}

/* 響應式設計 */
@media (max-width: 480px) {
  .stats-cards {
    flex-direction: column;
  }
  
  .record-right {
    gap: 10px;
  }
  
  .record-stat {
    min-width: 45px;
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
}
</style>