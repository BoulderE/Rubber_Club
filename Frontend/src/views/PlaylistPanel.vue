<template>
  <div style="background: red; padding: 50px; color: white;">
    <h1>TEST - Does this show?</h1>
    <p>Loading: {{ loading }}</p>
    <p>Playlists count: {{ playlists.length }}</p>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useExerciseStore } from '@/stores/exercise'
import { fetchMyPlaylists, fetchPlaylistDetail, createPlaylist, deletePlaylist, startRoutine} from '@/api/tasks'
import PlaylistCard from '@/components/PlaylistCard.vue'
import PlaylistCreateModal from '@/components/PlaylistCreate.vue'
import PlaylistEditModal from '@/components/PlaylistEdit.vue'
import PlaylistDetailModal from '@/components/PlaylistDetail.vue'

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

const routines = computed(() => 
  playlists.value.filter(p => p.is_routine)
)

const regularPlaylists = computed(() => 
  playlists.value.filter(p => !p.is_routine)
)

onMounted(async () => {
  await loadPlaylists()
})

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