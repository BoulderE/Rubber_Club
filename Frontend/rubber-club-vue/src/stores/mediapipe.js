import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useExerciseStore } from './exercise'
const API_URL = 'http://localhost:5001'

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
      
      // 更新分析结果
      analysisResults.value = result
      
      const exerciseStore = useExerciseStore();

      const currentExerciseInfo = exerciseStore.getExerciseById(currentExercise.value);

      if (!currentExerciseInfo) {
        // 如果因为某些原因找不到，就直接退出，防止程序崩溃
        console.error("无法在 exerciseStore 中找到当前运动的配置！");
        return result;
      }

      const dynamicKey = currentExerciseInfo.name;

      if (result[dynamicKey]) {
        const exerciseData = result[dynamicKey]; // 正确获取数据！
        console.log('[analyzeFrame] currentExercise =', currentExercise.value);
        console.log('[analyzeFrame] exerciseStore.name (dynamicKey) =', dynamicKey);
        console.log('[analyzeFrame] result[dynamicKey] exists?', !!result[dynamicKey]);
        
        // 后续逻辑完全不变，因为 exerciseData 现在有值了！
        count.value = exerciseData.count;
        energy.value = Math.round(exerciseData.energy);
        isPaused.value = exerciseData.paused;
        
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
    // 1. 向后端发送包含所有参数的 'start' 指令
    const success = await controlBackend({
      action: 'start',
      exercise: exerciseType,
      style: style
    });

    // 2. 如果后端确认成功，则更新前端的状态
    if (success) {
      currentExercise.value = exerciseType
      currentStyle.value = style // 保存难度模式
      status.value = 'active'
      count.value = 0
      energy.value = 0
      isPaused.value = false
      analysisResults.value = null
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
  console.log('[updateAnalysisData] input:', analysisData);
  // 根据你的后端返回键名调整
  count.value = analysisData.count ?? count.value
  energy.value = Math.round(analysisData.energy ?? energy.value)
  isPaused.value = !!analysisData.paused

  console.log('[updateAnalysisData] after update -> count:', count.value, 'energy:', energy.value, 'paused:', isPaused.value, 'status:', status.value);

  analysisResults.value = analysisData

  // if (isPaused.value && status.value === 'active') status.value = 'paused'
  // if (!isPaused.value && status.value === 'paused') status.value = 'active'
  if (isPaused.value) status.value = 'paused'
    else status.value = 'active'
  if (count.value >= repetitionLimit) status.value = 'completed'
}


  async function reset() {
    // 【修改】使用新的 payload 格式
    const success = await controlBackend({ action: 'reset' });
    if (success) {
      count.value = 0
      energy.value = 0
      analysisResults.value = null
      isAnalyzing.value = false
      currentStyle.value = null
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