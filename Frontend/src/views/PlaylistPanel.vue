<template>
  <div class="playlist-view">
    <header class="view-header">
      <button class="back-btn" @click="$router.back()">← 返回</button>
      <h1>我的訓練清單</h1>
      <button class="create-btn" @click="showCreateModal = true">
        + 新增清單
      </button>
    </header>

    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <p>載入中...</p>
    </div>

    <div v-else-if="playlists.length === 0" class="empty-state">
      <div class="empty-icon">📋</div>
      <h2>尚未建立訓練清單</h2>
      <p>建立您的第一個清單，輕鬆安排運動計畫</p>
      <button class="primary-btn" @click="showCreateModal = true">
        建立清單
      </button>
    </div>

    <div v-else class="playlists-container">
      <section v-if="routines.length > 0" class="playlist-section">
        <h2 class="section-title">⭐ 常用訓練</h2>
        <div class="playlist-grid">
          <PlaylistCard
            v-for="playlist in routines"
            :key="playlist.playlist_id"
            :playlist="playlist"
            @click="openPlaylistDetail(playlist)"
            @start="startPlaylist(playlist)"
            @edit="openEditModal(playlist)"
            @delete="confirmDelete(playlist)"
          />
        </div>
      </section>

      <section v-if="regularPlaylists.length > 0" class="playlist-section">
        <h2 class="section-title">📋 所有清單</h2>
        <div class="playlist-grid">
          <PlaylistCard
            v-for="playlist in regularPlaylists"
            :key="playlist.playlist_id"
            :playlist="playlist"
            @click="openPlaylistDetail(playlist)"
            @start="startPlaylist(playlist)"
            @edit="openEditModal(playlist)"
            @delete="confirmDelete(playlist)"
          />
        </div>
      </section>
    </div>

    <PlaylistCreateModal
      v-if="showCreateModal"
      :available-exercises="availableExercises"
      @close="showCreateModal = false"
      @created="onPlaylistCreated"
    />

    <PlaylistEditModal
      v-if="showEditModal"
      :playlist="selectedPlaylist"
      :playlist-exercises="selectedPlaylistExercises"
      :available-exercises="availableExercises"
      @close="closeEditModal"
      @updated="onPlaylistUpdated"
    />

    <PlaylistDetailModal
      v-if="showDetailModal"
      :playlist="selectedPlaylist"
      :exercises="selectedPlaylistExercises"
      @close="showDetailModal = false"
      @start="startPlaylist(selectedPlaylist)"
      @edit="openEditFromDetail"
    />

    <div v-if="showDeleteConfirm" class="delete-confirm-overlay">
      <div class="delete-confirm-modal">
        <h3>確認刪除</h3>
        <p>確定要刪除清單「{{ playlistToDelete?.playlist_name }}」嗎？此操作無法復原。</p>
        <div class="confirm-actions">
          <button class="cancel-btn" @click="showDeleteConfirm = false">取消</button>
          <button class="delete-btn" @click="executeDelete">刪除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useExerciseStore } from '@/stores/exercise'
import { fetchMyPlaylists, fetchPlaylistDetail, createPlaylist, deletePlaylist, startRoutine} from '@/api/tasks'
import PlaylistCard from '@/components/playlist/PlaylistCard.vue'
import PlaylistCreateModal from '@/components/playlist/PlaylistCreateModal.vue'
import PlaylistEditModal from '@/components/playlist/PlaylistEditModal.vue'
import PlaylistDetailModal from '@/components/playlist/PlaylistDetailModal.vue'

const router = useRouter()
const exerciseStore = useExerciseStore()

const loading = ref(true)
const playlists = ref([])
const showCreateModal = ref(false)
const showEditModal = ref(false)
const showDetailModal = ref(false)
const showDeleteConfirm = ref(false)
const selectedPlaylist = ref(null)
const selectedPlaylistExercises = ref([])
const playlistToDelete = ref(null)

const availableExercises = ref([
  { exercise_key: 'shoulder_flexion', exercise_name: '肩部屈曲' },
  { exercise_key: 'shoulder_abduction', exercise_name: '肩部外展' },
  { exercise_key: 'elbow_flexion', exercise_name: '肘部屈曲' },
  { exercise_key: 'wrist_extension', exercise_name: '腕部伸展' },
  { exercise_key: 'finger_spread', exercise_name: '手指張開' },
  { exercise_key: 'hip_flexion', exercise_name: '髖部屈曲' },
  { exercise_key: 'knee_extension', exercise_name: '膝部伸展' },
  { exercise_key: 'ankle_dorsiflexion', exercise_name: '踝部背屈' }
])

// Computed
const routines = computed(() => 
  playlists.value.filter(p => p.is_routine)
)

const regularPlaylists = computed(() => 
  playlists.value.filter(p => !p.is_routine)
)

// Lifecycle
onMounted(async () => {
  await loadPlaylists()
})

// Methods
async function loadPlaylists() {
  loading.value = true
  try {
    const data = await fetchMyPlaylists()
    playlists.value = data || []
  } catch (err) {
    console.error('Failed to load playlists:', err)
  } finally {
    loading.value = false
  }
}

async function openPlaylistDetail(playlist) {
  try {
    const detail = await fetchPlaylistDetail(playlist.playlist_id)
    selectedPlaylist.value = playlist
    selectedPlaylistExercises.value = detail.exercises || []
    showDetailModal.value = true
  } catch (err) {
    console.error('Failed to load playlist detail:', err)
  }
}

async function openEditModal(playlist) {
  try {
    const detail = await fetchPlaylistDetail(playlist.playlist_id)
    selectedPlaylist.value = playlist
    selectedPlaylistExercises.value = detail.exercises || []
    showDetailModal.value = false
    showEditModal.value = true
  } catch (err) {
    console.error('Failed to load playlist for editing:', err)
  }
}

function openEditFromDetail() {
  showDetailModal.value = false
  showEditModal.value = true
}

function closeEditModal() {
  showEditModal.value = false
  selectedPlaylist.value = null
  selectedPlaylistExercises.value = []
}

function confirmDelete(playlist) {
  playlistToDelete.value = playlist
  showDeleteConfirm.value = true
}

async function executeDelete() {
  if (!playlistToDelete.value) return
  
  try {
    await deletePlaylist(playlistToDelete.value.playlist_id)
    await loadPlaylists()
    showDeleteConfirm.value = false
    playlistToDelete.value = null
  } catch (err) {
    console.error('Failed to delete playlist:', err)
    alert('刪除失敗，請稍後再試')
  }
}

async function startPlaylist(playlist) {
  try {
    if (playlist.is_routine) {
      const result = await startRoutine(playlist.playlist_id)
      const detail = await fetchPlaylistDetail(result.new_playlist_id)
      navigateToExercise(detail.exercises)
    } else {
      const detail = await fetchPlaylistDetail(playlist.playlist_id)
      navigateToExercise(detail.exercises)
    }
  } catch (err) {
    console.error('Failed to start playlist:', err)
  }
}

function navigateToExercise(exercises) {
  if (!exercises || exercises.length === 0) {
    alert('此清單沒有運動項目')
    return
  }
  
  const exerciseIds = exercises.map(e => e.exercise_key)
  const firstExerciseId = exerciseStore.startPlaylist(exerciseIds)
  
  router.push({
    name: 'exercise',
    params: { type: firstExerciseId },
    query: {
      style: 'beginner',
      autoPlayVoice: 'true'
    }
  })
}

function onPlaylistCreated() {
  showCreateModal.value = false
  loadPlaylists()
}

function onPlaylistUpdated() {
  closeEditModal()
  loadPlaylists()
}
</script>

<style scoped>
.playlist-view {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  padding: 1rem;
  padding-bottom: 5rem;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 0;
  margin-bottom: 1.5rem;
}

.view-header h1 {
  color: #fff;
  font-size: 1.5rem;
  margin: 0;
}

.back-btn {
  background: rgba(255, 255, 255, 0.1);
  border: none;
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.back-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.create-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: transform 0.2s, box-shadow 0.2s;
}

.create-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  color: #a0a0a0;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.1);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.empty-state h2 {
  color: #fff;
  margin: 0 0 0.5rem;
}

.empty-state p {
  color: #a0a0a0;
  margin: 0 0 1.5rem;
}

.primary-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: #fff;
  padding: 0.75rem 2rem;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 600;
}

.playlists-container {
  max-width: 800px;
  margin: 0 auto;
}

.playlist-section {
  margin-bottom: 2rem;
}

.section-title {
  color: #fff;
  font-size: 1.1rem;
  margin: 0 0 1rem;
  padding-left: 0.5rem;
}

.playlist-grid {
  display: grid;
  gap: 1rem;
}

@media (min-width: 640px) {
  .playlist-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>