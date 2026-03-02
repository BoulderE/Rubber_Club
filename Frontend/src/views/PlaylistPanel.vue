<template>
  <div class="playlist-panel-wrapper">
    <!-- Backdrop -->
    <transition name="fade">
      <div 
        v-if="visible" 
        class="panel-backdrop" 
        @click="$emit('close')"
      ></div>
    </transition>

    <!-- Panel -->
    <transition name="slide-down">
      <div v-if="visible" class="playlist-panel">
        <div class="panel-header">
          <h2>📋 我的訓練清單</h2>
          <button class="create-new-btn" @click="handleCreatePlaylist">
            <span>+</span> 建立新清單
          </button>
        </div>

        <!-- Loading State -->
        <div v-if="loading" class="playlist-loading">
          <div class="loading-spinner"></div>
          <span>載入中...</span>
        </div>

        <!-- Playlists List -->
        <div v-else-if="hasPlaylists" class="playlists-grid">
          <div 
            v-for="playlist in playlists" 
            :key="playlist.id"
            class="playlist-item-card"
            @click="handleStartPlaylist(playlist)"
          >
            <div class="playlist-item-info">
              <h3 class="playlist-item-name">{{ playlist.name }}</h3>
              <p class="playlist-item-meta">
                {{ playlist.exercises?.length || 0 }} 個動作
                <span v-if="playlist.is_routine" class="routine-badge">常用</span>
              </p>
            </div>
            <div class="playlist-item-action">
              <span class="play-icon">▶</span>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-else class="playlist-empty">
          <div class="empty-icon">📝</div>
          <p>尚未建立訓練清單</p>
          <button class="create-first-btn" @click="handleCreatePlaylist">
            建立第一個清單
          </button>
        </div>

        <!-- Footer -->
        <div v-if="hasPlaylists" class="panel-footer">
          <router-link to="/playlists" class="view-all-link" @click="$emit('close')">
            查看全部清單 →
          </router-link>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useExerciseStore } from '@/stores/exercise';
import { fetchMyPlaylists } from '@/api/tasks';

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(['close']);

const router = useRouter();
const exerciseStore = useExerciseStore();

const loading = ref(false);
const playlists = ref([]);
const loaded = ref(false);

const hasPlaylists = computed(() => playlists.value.length > 0);

// Load playlists when panel opens
watch(() => props.visible, async (isVisible) => {
  if (isVisible && !loaded.value) {
    await loadPlaylists();
  }
});

async function loadPlaylists() {
  loading.value = true;
  try {
    const res = await fetchMyPlaylists();
    playlists.value = res || [];
    loaded.value = true;
  } catch (err) {
    console.error('Failed to load playlists:', err);
    playlists.value = [];
  } finally {
    loading.value = false;
  }
}

function handleCreatePlaylist() {
  emit('close');
  router.push('/playlists/create'); // Adjust this path to match your router
}

function handleStartPlaylist(playlist) {
  if (!playlist.exercises || playlist.exercises.length === 0) {
    alert('此清單沒有動作');
    return;
  }

  emit('close');

  const exerciseIds = playlist.exercises.map(e => e.exercise_type || e.exercise_key || e.id);
  const firstExerciseId = exerciseStore.startPlaylist(exerciseIds);
  
  router.push({
    name: 'exercise',
    params: { type: firstExerciseId },
    query: {
      style: 'beginner',
      autoPlayVoice: 'true',
      playlistId: playlist.id
    }
  });
}

// Expose refresh method for parent component
function refresh() {
  loaded.value = false;
  loadPlaylists();
}

defineExpose({ refresh });
</script>

<style scoped>
.playlist-panel-wrapper {
  position: relative;
}

.panel-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: 50;
}

.playlist-panel {
  position: relative;
  z-index: 100;
  background: white;
  border-radius: 20px;
  margin: -20px 0 40px 0;
  padding: 28px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f0f0f0;
}

.panel-header h2 {
  font-size: 1.4rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
}

.create-new-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 20px;
  border-radius: 12px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.create-new-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.create-new-btn span {
  font-size: 1.2rem;
  font-weight: 700;
}

.playlist-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 60px;
  color: #666;
}

.loading-spinner {
  width: 28px;
  height: 28px;
  border: 3px solid #e0e0e0;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.playlists-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.playlist-item-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, #f8f9ff 0%, #f0f3ff 100%);
  border: 2px solid #e8ebff;
  border-radius: 14px;
  padding: 20px 24px;
  cursor: pointer;
  transition: all 0.25s ease;
}

.playlist-item-card:hover {
  border-color: #667eea;
  background: linear-gradient(135deg, #f0f3ff 0%, #e8ebff 100%);
  transform: translateX(6px);
  box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
}

.playlist-item-info {
  flex: 1;
}

.playlist-item-name {
  font-size: 1.15rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 6px 0;
}

.playlist-item-meta {
  font-size: 0.9rem;
  color: #6b7280;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.routine-badge {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 0.75rem;
  padding: 3px 12px;
  border-radius: 12px;
  font-weight: 600;
}

.playlist-item-action {
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: transform 0.3s ease;
}

.playlist-item-card:hover .playlist-item-action {
  transform: scale(1.1);
}

.play-icon {
  color: white;
  font-size: 16px;
  margin-left: 3px;
}

.playlist-empty {
  text-align: center;
  padding: 50px 20px;
  color: #6b7280;
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: 16px;
}

.playlist-empty p {
  font-size: 1.1rem;
  margin-bottom: 24px;
  color: #9ca3af;
}

.create-first-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 14px 32px;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.create-first-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

.panel-footer {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
  text-align: center;
}

.view-all-link {
  color: #667eea;
  text-decoration: none;
  font-weight: 600;
  font-size: 1rem;
  transition: color 0.2s ease;
}

.view-all-link:hover {
  color: #764ba2;
}

/* Transitions */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}

.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 768px) {
  .playlist-panel {
    margin: -10px -10px 30px -10px;
    border-radius: 0 0 20px 20px;
    padding: 20px;
  }

  .panel-header {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }

  .create-new-btn {
    justify-content: center;
  }

  .playlists-grid {
    grid-template-columns: 1fr;
  }
}
</style>