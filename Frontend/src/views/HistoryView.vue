<template>
  <div class="history-container">
    <h1>運動歷史記錄</h1>
    
    <!-- 統計卡片 -->
    <div class="stats-cards" v-if="stats">
      <div class="stat-card">
        <div class="stat-number">{{ stats.total_workouts }}</div>
        <div class="stat-label">總運動次數</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{{ stats.total_reps }}</div>
        <div class="stat-label">總次數</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{{ stats.avg_smoothness }}%</div>
        <div class="stat-label">平均流暢度</div>
      </div>
    </div>
    
    <!-- 記錄列表 -->
    <div class="records-list">
      <h2>最近記錄</h2>
      <div v-if="records.length === 0" class="no-records">
        還沒有運動記錄，開始你的第一次運動吧！
      </div>
      <div v-else>
        <div class="record-item" v-for="record in records" :key="record.id">
          <div class="record-exercise">{{ record.exercise_name }}</div>
          <div class="record-details">
            <span>{{ record.rep_count }} 次</span>
            <span>{{ record.smoothness }}% 流暢度</span>
            <span>{{ formatDuration(record.duration) }}</span>
          </div>
          <div class="record-date">{{ formatDate(record.created_at) }}</div>
        </div>
      </div>
    </div>
    
    <button @click="$router.push('/')" class="back-btn">返回首頁</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
const authStore = useAuthStore()

const records = ref([])
const stats = ref(null)

onMounted(async () => {
    const userId = authStore.userId
  if (!userId) {
    console.error('未登入')
    return
  }
  
  try {
    // 獲取統計
    const statsRes = await fetch(`${API_BASE_URL}/api/records/${userId}/stats`)
    stats.value = await statsRes.json()
    
    // 獲取記錄列表
    const recordsRes = await fetch(`${API_BASE_URL}/api/records/${userId}`)
    const data = await recordsRes.json()
    records.value = data.records
  } catch (error) {
    console.error('獲取記錄失敗:', error)
  }
})

const formatDate = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleDateString('zh-TW', {
    timeZone: 'Asia/Hong Kong',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
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
</script>

<style scoped>
.history-container {
  padding: 20px;
  max-width: 600px;
  margin: 0 auto;
}

.stats-cards {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
}

.stat-card {
  flex: 1;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  color: white;
}

.stat-number {
  font-size: 28px;
  font-weight: bold;
}

.stat-label {
  font-size: 12px;
  opacity: 0.9;
}

.records-list h2 {
  margin-bottom: 15px;
}

.no-records {
  text-align: center;
  color: #666;
  padding: 40px;
}

.record-item {
  background: #f8f9fa;
  border-radius: 10px;
  padding: 15px;
  margin-bottom: 10px;
}

.record-exercise {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 8px;
}

.record-details {
  display: flex;
  gap: 15px;
  font-size: 14px;
  color: #666;
}

.record-date {
  font-size: 12px;
  color: #999;
  margin-top: 8px;
}

.back-btn {
  width: 100%;
  padding: 15px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 16px;
  margin-top: 20px;
  cursor: pointer;
}
</style>