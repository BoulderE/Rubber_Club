import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useExerciseStore } from './exercise'
import { getApiBase } from '@/api/base';
const API_URL = getApiBase();

let requestSequence = 0

export const useMediapipeStore = defineStore('mediapipe', () => {

  const status = ref('ready')
  const count = ref(0)
  const energy = ref(0)
  const currentExercise = ref('')
  const isAnalyzing = ref(false)
  const analysisResults = ref(null)
  const isPaused = ref(false)
  const currentStyle = ref(null)
  const repetitionLimit = ref(15)
  const isActive = computed(() => status.value === 'active')

  const accurateCount = ref(0)
  const accuracy = ref(0)
  const lastCount = ref(0)

  const smoothness = ref(100)   
  const repDurations = ref([])

  // ===== new: LSTM scores =====
  const lstmScore = ref(0)           // 最新一幀的 LSTM 分數
  const lstmScores = ref([])         // 整組運動期間所有 LSTM 分數
  const lstmScoreAvg = computed(() => {
    if (lstmScores.value.length === 0) return 0
    const sum = lstmScores.value.reduce((a, b) => a + b, 0)
    return Math.round(sum / lstmScores.value.length)
  })

  const lastGesture = ref(null)           
  const gestureMessage = ref('')          
  const waitingForGesture = ref(false)    

  const completedThisFrame = ref(false)

  const lastPauseChangeTime = ref(0)
  const PAUSE_DEBOUNCE_MS = 1500

  async function controlBackend(payload) {
    try {
      const response = await fetch(`${API_URL}/mediapipe/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Failed to send control action: ${payload.action}`);
      }
      console.log(`Backend action '${payload.action}' successful.`);
      return true;
    } catch (error) {
      console.error(`Error with backend action '${payload.action}':`, error);
      return false;
    }
  }

  async function analyzeFrame(imageData) {
    if (isAnalyzing.value) return null

    const thisRequestId = ++requestSequence

    try {
      isAnalyzing.value = true

      const base64Data = imageData.split(',')[1]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })

      const formData = new FormData()
      formData.append('file', blob, 'frame.jpg')

      const response = await fetch(`${API_URL}/mediapipe/analyze-stream`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      if (thisRequestId !== requestSequence) {
        console.log(`[analyzeFrame] 丢弃过期响应 #${thisRequestId}，当前最新 #${requestSequence}`)
        return null
      }

      console.log('[analyzeFrame] backend raw result:', result)
      console.log('[analyzeFrame] keys:', Object.keys(result))

      if (result.gesture_detected) {
        lastGesture.value = result.gesture_detected
      }

      return result
    } catch (error) {
      console.error('Frame analysis error:', error)
      return null
    } finally {
      isAnalyzing.value = false
    }
  }

  async function startExercise(exerciseType, style) {
    requestSequence = 0

    const success = await controlBackend({
      action: 'start',
      exercise: exerciseType,
      style: style
    });

    if (success) {
      accurateCount.value = 0
      accuracy.value = 0
      lastCount.value = 0
      currentExercise.value = exerciseType
      currentStyle.value = style

      repetitionLimit.value = (style === 'beginner') ? 10 : 15
    
      status.value = 'paused' 
      count.value = 0
      energy.value = 0
      isPaused.value = true    
      analysisResults.value = null
      smoothness.value = 100          
      repDurations.value = []

      // ===== new: reset LSTM =====
      lstmScore.value = 0
      lstmScores.value = []

      lastPauseChangeTime.value = 0
      console.log(`Exercise '${exerciseType}' started with style '${style}'. max reps: ${repetitionLimit.value}`);
    } else {
      console.error("Failed to start exercise on the backend.");
    }
  }

  function stopExercise() {
    requestSequence = 0
    status.value = 'ready'
    isAnalyzing.value = false
  }

  function updateAnalysisData(analysisData) {
    console.log('[updateAnalysisData] input:', analysisData)

    const newCount = analysisData.count ?? count.value
    const oldCount = count.value
    const grew = Number(newCount) > Number(oldCount)
    count.value = newCount
    energy.value = Math.round(analysisData.energy ?? energy.value)
    completedThisFrame.value = !!analysisData.completed || grew

    // 暂停状态防抖动
    const backendPaused = !!analysisData.paused
    const now = Date.now()
    
    if (backendPaused !== isPaused.value) {
      if (now - lastPauseChangeTime.value > PAUSE_DEBOUNCE_MS) {
        console.log(`[updateAnalysisData] 暂停状态切换: ${isPaused.value} -> ${backendPaused}`)
        isPaused.value = backendPaused
        lastPauseChangeTime.value = now
      } else {
        console.log(`[updateAnalysisData] 忽略抖动: 尝试切换到 ${backendPaused}，距上次切换仅 ${now - lastPauseChangeTime.value}ms`)
      }
    }

    if (typeof analysisData.smoothness !== 'undefined') {           
      smoothness.value = Number(analysisData.smoothness) || 0      
    }                                                               
    if (Array.isArray(analysisData.rep_durations)) {               
      repDurations.value = analysisData.rep_durations.slice()       
    }

    // ===== new: fetch LSTM scores =====
    if (typeof analysisData.lstm_score !== 'undefined' && analysisData.lstm_score !== null) {
      const score = Number(analysisData.lstm_score)
      lstmScore.value = score
      if (!isPaused.value && score > 0) {
        lstmScores.value.push(score)
      }
      console.log('[updateAnalysisData] lstm_score=', score, 'avg=', lstmScoreAvg.value)
    }
    
    if (isPaused.value) {
      waitingForGesture.value = true
      gestureMessage.value = analysisData.feedback || '請做 👍 手勢開始運動'
    } else {
      waitingForGesture.value = false
      gestureMessage.value = ''
    }

    if (isPaused.value && !backendPaused) {
      console.log('[updateAnalysisData] 后端要求恢复，但被防抖忽略')
    } else if (!isPaused.value && backendPaused) {
      console.log('[updateAnalysisData] 后端要求暂停，但被防抖忽略')
    }

    analysisResults.value = analysisData

    if (isPaused.value) status.value = 'paused'
    else status.value = 'active'
    
    if (typeof repetitionLimit.value !== 'undefined' && repetitionLimit.value !== null && count.value >= repetitionLimit.value) {
      status.value = 'completed'
    }
    
    if (grew) {
      const nonStandard = analysisData.category === 'non_standard'
      if (!nonStandard) {
        accurateCount.value = (accurateCount.value || 0) + 1
      }
    }
    lastCount.value = Number(count.value)

    const total = Number(count.value) || 0
    if (total > 0) {
      accuracy.value = Math.round((accurateCount.value / total) * 100)
    } else {
      accuracy.value = 0
    }

    console.log(
      '[updateAnalysisData] after calc ->',
      'count=', count.value,
      'accurateCount=', accurateCount.value,
      'accuracy=', accuracy.value,
      'lstmScore=', lstmScore.value,
      'lstmScoreAvg=', lstmScoreAvg.value,
      'paused=', isPaused.value
    )
  }

  async function reset(skipBackend=false) {
    requestSequence = 0

    if (!skipBackend) {
      const success = await controlBackend({ action: 'reset' });
      if (!success) {
        console.warn('[reset] 后端 reset 失败，但仍重置本地状态')
      }
    }

      accurateCount.value = 0
      accuracy.value = 0
      lastCount.value = 0
      count.value = 0
      energy.value = 0
      analysisResults.value = null
      isAnalyzing.value = false
      currentStyle.value = null
      smoothness.value = 100
      repDurations.value = []

      // ===== new: reset LSTM =====
      lstmScore.value = 0
      lstmScores.value = []

      lastGesture.value = null
      gestureMessage.value = '請做 👍 手勢開始運動'
      waitingForGesture.value = true
      isPaused.value = true
      status.value = 'paused'
      lastPauseChangeTime.value = 0

      completedThisFrame.value = false
  }
  
  return {
    status,
    count,
    energy,
    currentExercise,
    isAnalyzing,
    analysisResults,
    repetitionLimit,
    currentStyle,
    isActive,
    isPaused,

    accurateCount,
    accuracy,

    smoothness,   
    repDurations,

    // ===== new: export LSTM =====
    lstmScore,
    lstmScores,
    lstmScoreAvg,

    lastGesture,
    gestureMessage,
    waitingForGesture,

    completedThisFrame,
    
    analyzeFrame,
    startExercise,
    stopExercise,
    reset,
    updateAnalysisData
  }
})