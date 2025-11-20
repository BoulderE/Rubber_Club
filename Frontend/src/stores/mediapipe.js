import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useExerciseStore } from './exercise'
import { getApiBase } from '@/api/base';
const API_URL = getApiBase();

export const useMediapipeStore = defineStore('mediapipe', () => {
  // 状态
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

  // 【新增】手势相关状态
  const lastGesture = ref(null)           // 最后检测到的手势
  const gestureMessage = ref('')          // 手势提示信息
  const waitingForGesture = ref(false)    // 是否在等待手势

  const completedThisFrame = ref(false)
  async function controlBackend(payload) {
    try {
      const response = await fetch(`${API_URL}/mediapipe/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // 直接将整个 payload 对象序列化
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
    console.log('[analyzeFrame] backend raw result:', result)
    console.log('[analyzeFrame] keys:', Object.keys(result))

    // 手势
    if (result.gesture_detected) {
      lastGesture.value = result.gesture_detected
    }

  //   analysisResults.value = result

  //   const exerciseStore = useExerciseStore()
  //   const currentExerciseInfo = exerciseStore.getExerciseById(currentExercise.value)
  //   if (!currentExerciseInfo) {
  //     console.error('无法在 exerciseStore 中找到当前运动的配置！')
  //     return result
  //   }

  //   // 优先 type，其次 name
  //   const keyByType = currentExercise.value
  //   const keyByName = currentExerciseInfo.name
  //   const exerciseData = result[keyByType] || result[keyByName]

  //   console.log(
  //     '[analyzeFrame] select key ->',
  //     exerciseData ? (result[keyByType] ? 'type' : 'name') : 'none',
  //     'keyByType=',
  //     keyByType,
  //     'keyByName=',
  //     keyByName
  //   )

  //   // 1) 原路更新（保持你原有逻辑）
  //   if (exerciseData && typeof exerciseData === 'object') {
  //     updateAnalysisData(exerciseData)
  //   } else {
  //     console.warn('后端未返回匹配当前运动的键：', keyByType, '或', keyByName)
  //   }

  // // 2) 兜底解析 count 和 completed，确保无论放在哪层都能拿到
  // const resolvedCount =
  //   (exerciseData && typeof exerciseData.count === 'number' ? exerciseData.count : undefined) ??
  //   (typeof result?.count === 'number' ? result.count : undefined) ??
  //   (typeof result?.current?.count === 'number' ? result.current.count : undefined) ??
  //   (keyByType && typeof result?.[keyByType]?.count === 'number' ? result[keyByType].count : undefined) ??
  //   (keyByName && typeof result?.[keyByName]?.count === 'number' ? result[keyByName].count : undefined)

  // if (typeof resolvedCount === 'number' && !Number.isNaN(resolvedCount)) {
  //   count.value = resolvedCount
  // }

  // const resolvedCompleted =
  //   (exerciseData ? exerciseData.completed : undefined) ??
  //   result?.completed ??
  //   result?.current?.completed ??
  //   (keyByType ? result?.[keyByType]?.completed : undefined) ??
  //   (keyByName ? result?.[keyByName]?.completed : undefined)

  // completedThisFrame.value = !!resolvedCompleted

  // console.log('[analyzeFrame] resolvedCount =', resolvedCount, 'resolvedCompleted =', completedThisFrame.value)

  // // 完成状态判定
  // if (count.value >= repetitionLimit.value) {
  //   status.value = 'completed'
  // }

  return result
  } catch (error) {
  console.error('Frame analysis error:', error)
  return null
  } finally {
  isAnalyzing.value = false
  }
  }

  async function startExercise(exerciseType, style) {
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
      //new
      repetitionLimit.value = (style === 'beginner') ? 10 : 15
    
      status.value = 'paused' 
      count.value = 0
      energy.value = 0
      isPaused.value = true    
      analysisResults.value = null
      smoothness.value = 100          
      repDurations.value = []         
      console.log(`Exercise '${exerciseType}' started with style '${style}'. max reps: ${repetitionLimit.value}`);
    } else {
      console.error("Failed to start exercise on the backend.");
    }
  }

  
  // 停止运动
  function stopExercise() {
    status.value = 'ready'
    isAnalyzing.value = false
  }

  function updateAnalysisData(analysisData) {
    console.log('[updateAnalysisData] input:', analysisData)

    //new
    const newCount = analysisData.count ?? count.value
    const oldCount = count.value
    const grew = Number(newCount) > Number(oldCount)
    // 原有赋值（根据后端键名）
    count.value = newCount
    energy.value = Math.round(analysisData.energy ?? energy.value)
    completedThisFrame.value = !!analysisData.completed || grew
    const wasPaused = isPaused.value
    isPaused.value = !!analysisData.paused

    if (typeof analysisData.smoothness !== 'undefined') {           
      smoothness.value = Number(analysisData.smoothness) || 0      
    }                                                               
    if (Array.isArray(analysisData.rep_durations)) {               
      repDurations.value = analysisData.rep_durations.slice()       
    }
    
    if (isPaused.value) {
      waitingForGesture.value = true
      gestureMessage.value = analysisData.feedback || '請做 👍 手勢開始運動'
    } else {
      waitingForGesture.value = false
      gestureMessage.value = ''
    }

    // 【新增】检测暂停状态变化
    if (wasPaused && !isPaused.value) {
      console.log('[updateAnalysisData] 运动已通过手势恢复')
    } else if (!wasPaused && isPaused.value) {
      console.log('[updateAnalysisData] 运动已通过手势暂停')
    }

    analysisResults.value = analysisData

    console.log(
      '[updateAnalysisData] after update ->',
      'count:', count.value,
      'energy:', energy.value,
      'paused:', isPaused.value,
      'status:', status.value,
      'smoothness:', smoothness.value
    )

    analysisResults.value = analysisData

    // 状态机
    if (isPaused.value) status.value = 'paused'
    else status.value = 'active'
    if (typeof repetitionLimit.value !== 'undefined' && repetitionLimit.value !== null && count.value >= repetitionLimit.value) {
      status.value = 'completed'
    }

    // 只在 count 增长时统计一次，避免 hold=2 帧导致重复累计
    
    if (grew) {
      const nonStandard = analysisData.category === 'non_standard'
      if (!nonStandard) {
        accurateCount.value = (accurateCount.value || 0) + 1
      }
    }
    lastCount.value = Number(count.value)

    // 计算准确率（0-100 的整数百分比）
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
      'category=', analysisData.category,
      'grew=', grew
    )
  }


  async function reset() {
    // 【修改】使用新的 payload 格式
    const success = await controlBackend({ action: 'reset' });
    if (success) {
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

      //new
      lastGesture.value = null
      gestureMessage.value = '請做 👍 手勢開始運動'
      waitingForGesture.value = true
      isPaused.value = true
      status.value = 'paused'

      completedThisFrame.value = false
    }
  }
  
  return {
    // 状态
    status,
    count,
    energy,
    currentExercise,
    isAnalyzing,
    analysisResults,
    repetitionLimit,
    currentStyle,
    // 计算属性
    isActive,
    isPaused,

    accurateCount,
    accuracy,

    // new
    smoothness,   
    repDurations,

    lastGesture,
    gestureMessage,
    waitingForGesture,

    completedThisFrame,
    
    // 方法
    analyzeFrame,
    startExercise,
    stopExercise,
    reset,
    // pauseWorkout,
    // resumeWorkout,
    updateAnalysisData
  }
})