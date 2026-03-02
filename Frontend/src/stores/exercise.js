import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useExerciseStore = defineStore('exercise', () => {
  const exerciseTypes = ref([
    {
      id: 'lateral_raise',
      name: '側平舉',
      description: '锻炼肩部中束',
      icon: '💪',
      tips: ['保持身體直立，核心收緊','手臂舉至肩高','控制下放'],
      imageUrl: '/images/lateral_raise_image_1.png',
      orientation: 'landscape'
    },
    {
      id: 'bicep_curl',
      name: '二頭肌彎舉',
      description: '锻炼胸部肌肉',
      icon: '🏋️',
      tips: ['握住拉力器','肩胛後收','控制速度','拉時呼氣'],
      imageUrl: '/images/bicep_curl_image_1.png',
      orientation: 'portrait'
    },
    {
      id: 'front_raise',
      name: '前平舉',
      description: '锻炼肩部前束',
      icon: '💪',
      tips: ['從大腿前抬至與肩同高','核心收緊避免後仰'],
      imageUrl: '/images/front_raise_image_1.png',
      orientation: 'portrait'
    },
    {
      id: 'overhead_press',
      name: '過頭推舉',
      description: '綜合鍛鍊肩部和手臂力量',
      icon: '🏋️',
      tips: ['起於肩膀高','推至手腕過頭頂','避免過度後仰'],
      imageUrl: '/images/overhead_press_image_1.png',
      orientation: 'portrait'
    },
    {
      id: 'chest_pull',
      name: '胸部側拉',
      description: '強化上背與肩後肌群的穩健訓練，協助長者改善肩胛後收與挺胸姿勢。',
      icon: '🦵',
      tips: ['膝蓋對齊腳尖','髖向後坐','站起伸直髖膝'],
      imageUrl: '/images/chest_pull_image_1.png',
      orientation: 'landscape'
    },
    {
      id: 'diagonal_lift',
      name: '對角線推舉',
      description: '改善日常生活中斜向抬舉物品的能力。',
      icon: '🎯',
      tips: [
        '單手斜推至對側上方',
        '核心收緊，避免聳肩',
      ],
      imageUrl: '/images/diagonal_lift_image_1.png',
      orientation: 'portrait'
    },
  ])

  const selectedExercise = ref('lateral_raise')
  const exerciseHistory = ref([])  
  const startTime = ref(null) 
  const endTime = ref(null) 

  // ==================== Playlist Mode ====================
  const playlist = ref([])
  const currentPlaylistIndex = ref(-1)
  const isPlaylistMode = computed(() => playlist.value.length > 0 && currentPlaylistIndex.value >= 0)

  const currentPlaylistExercise = computed(() => {
    if (!isPlaylistMode.value) return null
    return playlist.value[currentPlaylistIndex.value]
  })

  const hasNextInPlaylist = computed(() => {
    return isPlaylistMode.value && currentPlaylistIndex.value < playlist.value.length - 1
  })

  const playlistProgress = computed(() => ({
    current: currentPlaylistIndex.value + 1,
    total: playlist.value.length
  }))

  const startPlaylist = (exerciseIds) => {
    playlist.value = [...exerciseIds]
    currentPlaylistIndex.value = 0
    return playlist.value[0]
  }

  const nextInPlaylist = () => {
    if (hasNextInPlaylist.value) {
      currentPlaylistIndex.value++
      return playlist.value[currentPlaylistIndex.value]
    }
    return null
  }

  const clearPlaylist = () => {
    playlist.value = []
    currentPlaylistIndex.value = -1
  }

  const getPlaylistExercises = computed(() => {
    return playlist.value.map(id => exerciseTypes.value.find(ex => ex.id === id)).filter(Boolean)
  })

  // ==================== Active Task ====================
  const activeTask = ref(null)

  const isTaskMode = computed(() => activeTask.value !== null)

  const taskProgress = computed(() => {
    if (!activeTask.value) return null
    return {
      completed_sets: activeTask.value.completed_sets,
      target_sets: activeTask.value.target_sets,
      target_reps: activeTask.value.target_reps,
      remaining_sets: activeTask.value.target_sets - activeTask.value.completed_sets
    }
  })

  const setActiveTask = (task) => {
    activeTask.value = task
  }

  const clearActiveTask = () => {
    activeTask.value = null
  }

  const getActiveTask = () => {
    return activeTask.value
  }

  const incrementTaskProgress = () => {
    if (activeTask.value) {
      activeTask.value.completed_sets += 1
    }
  }

  const isTaskComplete = computed(() => {
    if (!activeTask.value) return false
    return activeTask.value.completed_sets >= activeTask.value.target_sets
  })

  // user playlists and routines
  const playlists = ref([])           
  const routines = ref([])            
  const currentPlaylistDetail = ref(null)  
  const playlistLoading = ref(false)

  const setPlaylists = (data) => {
    playlists.value = data.filter(p => !p.is_routine)
    routines.value = data.filter(p => p.is_routine)
  }

  const setCurrentPlaylist = (playlist) => {
    currentPlaylistDetail.value = playlist
  }

  const clearCurrentPlaylist = () => {
    currentPlaylistDetail.value = null
  }

  const setPlaylistLoading = (val) => {
    playlistLoading.value = val
  }

  const getPlaylistById = (playlistId) => {
    return playlists.value.find(p => p.playlist_id === playlistId) ||
           routines.value.find(p => p.playlist_id === playlistId)
  }

  const selectExercise = (exerciseId) => {
    selectedExercise.value = exerciseId
  }

  const addToHistory = (record) => {
    exerciseHistory.value.unshift({
      ...record,
      timestamp: new Date().toISOString()
    })
  }

  const getExerciseById = (id) => {
    return exerciseTypes.value.find(ex => ex.id === id)
  }

  const startExercise = () => { 
    startTime.value = Date.now();
    endTime.value = null; 
  }

  const endExercise = () => {   
    endTime.value = Date.now();
  }

  return {
    // Exercise types
    exerciseTypes,
    selectedExercise,
    exerciseHistory,
    selectExercise,
    addToHistory,
    getExerciseById,
    
    // Timing
    startTime,
    endTime,
    startExercise,
    endExercise,

    // Playlist mode (for active exercise session)
    playlist,
    currentPlaylistIndex,
    isPlaylistMode,
    currentPlaylistExercise,
    hasNextInPlaylist,
    playlistProgress,
    getPlaylistExercises,
    startPlaylist,
    nextInPlaylist,
    clearPlaylist,

    // Active task mode
    activeTask,
    isTaskMode,
    taskProgress,
    setActiveTask,
    clearActiveTask,
    getActiveTask,
    incrementTaskProgress,
    isTaskComplete,

    // User playlists (NEW - for saved playlists from backend)
    playlists,
    routines,
    currentPlaylistDetail,
    playlistLoading,
    setPlaylists,
    setCurrentPlaylist,
    clearCurrentPlaylist,
    setPlaylistLoading,
    getPlaylistById
  }
})