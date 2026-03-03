<template>
  <div class="playlist-card" @click="$emit('click')">
    <div class="card-content">
      <div class="card-icon">
        {{ playlist.is_routine ? '⭐' : '📋' }}
      </div>
      <div class="card-info">
        <h3 class="card-title">{{ playlist.playlist_name }}</h3>
        <p class="card-meta">
          {{ playlist.exercise_count }} 個動作
          <span v-if="playlist.progress > 0" class="progress-text">
            · {{ playlist.progress }}% 完成
          </span>
        </p>
      </div>
    </div>
    
    <div class="card-actions" @click.stop>
      <button class="action-btn start-btn" @click="$emit('start')" title="開始">
        ▶
      </button>
      <button class="action-btn edit-btn" @click="$emit('edit')" title="編輯">
        ✏️
      </button>
      <button class="action-btn delete-btn" @click="$emit('delete')" title="刪除">
        🗑️
      </button>
    </div>

    <!-- Progress Bar -->
    <div v-if="playlist.progress > 0" class="progress-bar">
      <div class="progress-fill" :style="{ width: playlist.progress + '%' }"></div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  playlist: {
    type: Object,
    required: true
  }
})

defineEmits(['click', 'start', 'edit', 'delete'])
</script>

<style scoped>
.playlist-card {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}

.playlist-card:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(102, 126, 234, 0.3);
  transform: translateY(-2px);
}

.card-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.card-icon {
  font-size: 2rem;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
}

.card-info {
  flex: 1;
  min-width: 0;
}

.card-title {
  color: #fff;
  font-size: 1.1rem;
  margin: 0 0 0.25rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card-meta {
  color: #a0a0a0;
  font-size: 0.875rem;
  margin: 0;
}

.progress-text {
  color: #667eea;
}

.card-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.action-btn {
  flex: 1;
  padding: 0.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s;
}

.start-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
}

.start-btn:hover {
  transform: scale(1.05);
}

.edit-btn {
  background: rgba(255, 255, 255, 0.1);
}

.edit-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.delete-btn {
  background: rgba(255, 255, 255, 0.1);
}

.delete-btn:hover {
  background: rgba(239, 68, 68, 0.2);
}

.progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: rgba(255, 255, 255, 0.1);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  transition: width 0.3s;
}
</style>