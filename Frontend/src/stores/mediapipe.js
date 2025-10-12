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
  // 运动限制
  const repetitionLimit = 15
  
  // 计算属性
  const isActive = computed(() => status.value === 'active')

  const accurateCount = ref(0)
  const accuracy = ref(0)
  const lastCount = ref(0)

  //new
  const smoothness = ref(100)   
  const repDurations = ref([])

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
      console.log('[analyzeFrame] backend raw result:', result);
      console.log('[analyzeFrame] keys:', Object.keys(result));
      
      analysisResults.value = result
      
      const exerciseStore = useExerciseStore();

      const currentExerciseInfo = exerciseStore.getExerciseById(currentExercise.value);

      if (!currentExerciseInfo) {
        console.error("无法在 exerciseStore 中找到当前运动的配置！");
        return result;
      }

      const dynamicKey = currentExerciseInfo.name;

      if (result[dynamicKey]) {
        const exerciseData = result[dynamicKey]; 
        console.log('[analyzeFrame] currentExercise =', currentExercise.value);
        console.log('[analyzeFrame] exerciseStore.name (dynamicKey) =', dynamicKey);
        console.log('[analyzeFrame] result[dynamicKey] exists?', !!result[dynamicKey]);
        
        count.value = exerciseData.count;
        energy.value = Math.round(exerciseData.energy);
        isPaused.value = exerciseData.paused;

        if (typeof exerciseData.smoothness !== 'undefined') {           
          smoothness.value = Number(exerciseData.smoothness) || 0        
        }                                                               
        if (Array.isArray(exerciseData.rep_durations)) {                 
          repDurations.value = exerciseData.rep_durations.slice()        
        }  
        
        if (exerciseData.paused && status.value === 'active') {
            status.value = 'paused';
        } else if (!exerciseData.paused && status.value === 'paused') {
            status.value = 'active';
        }
      }

      if (count.value >= repetitionLimit) {
        status.value = 'completed'
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
      status.value = 'active'
      count.value = 0
      energy.value = 0
      isPaused.value = false
      analysisResults.value = null
      smoothness.value = 100          
      repDurations.value = []         
      console.log(`Exercise '${exerciseType}' started with style '${style}'.`);
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

  // 原有赋值（根据后端键名）
  count.value = analysisData.count ?? count.value
  energy.value = Math.round(analysisData.energy ?? energy.value)
  isPaused.value = !!analysisData.paused

  if (typeof analysisData.smoothness !== 'undefined') {           
    smoothness.value = Number(analysisData.smoothness) || 0      
  }                                                               
  if (Array.isArray(analysisData.rep_durations)) {               
    repDurations.value = analysisData.rep_durations.slice()       
  }   

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
  if (typeof repetitionLimit !== 'undefined' && repetitionLimit !== null && count.value >= repetitionLimit) {
    status.value = 'completed'
  }

  // 只在 count 增长时统计一次，避免 hold=2 帧导致重复累计
  const grew = Number(count.value) > Number(lastCount.value)
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
    }
  }
  
  async function pauseWorkout() {
    const success = await controlBackend({ action: 'pause' });
    if (success) {
      console.log('[startExercise] set active. currentExercise:', currentExercise.value, 'style:', currentStyle.value);
      status.value = 'paused';
      isPaused.value = true;
    }
  }

  async function resumeWorkout() {
    const success = await controlBackend({ action: 'resume' });
    if (success) {
      status.value = 'active';
      isPaused.value = false;
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
    
    // 方法
    analyzeFrame,
    startExercise,
    stopExercise,
    reset,
    pauseWorkout,
    resumeWorkout,
    updateAnalysisData
  }
})