import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

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
  // 运动限制
  const repetitionLimit = 15
  
  // 计算属性
  const isActive = computed(() => status.value === 'active')

  async function controlBackend(action) {
    try {
      const response = await fetch(`${API_URL}/mediapipe/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action }), // 'reset', 'pause', 'resume'
      });
      if (!response.ok) {
        throw new Error(`Failed to send control action: ${action}`);
      }
      console.log(`Backend action '${action}' successful.`);
      return true;
    } catch (error) {
      console.error(`Error with backend action '${action}':`, error);
      return false;
    }
  }
  
  // 分析帧的方法
  async function analyzeFrame(imageData) {
    if (!isActive.value || isAnalyzing.value) return null
    
    try {
      isAnalyzing.value = true
      
      // 将 base64 图像转换为 Blob
    const base64Data = imageData.split(',')[1]
    const byteCharacters = atob(base64Data)
    const byteNumbers = new Array(byteCharacters.length)
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    const byteArray = new Uint8Array(byteNumbers)
    const blob = new Blob([byteArray], { type: 'image/jpeg' })
    
    // 创建 FormData（修改点1：改为 FormData）
    const formData = new FormData()
    formData.append('file', blob, 'frame.jpg')
    
    // 发送到后端的 analyze-stream 端点（修改点2：改为正确的端点）
    const response = await fetch(`${API_URL}/mediapipe/analyze-stream`, {
      method: 'POST',
      body: formData
    })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const result = await response.json()
      
      // 更新分析结果
      analysisResults.value = result
      
      // 根据当前运动类型更新数据
      if (result[currentExercise.value]) {
        const exerciseData = result[currentExercise.value];
        count.value = exerciseData.count;
        energy.value = Math.round(exerciseData.energy);
        isPaused.value = exerciseData.paused; // 同步暂停状态
        
        // 根据暂停状态更新主状态
        if (exerciseData.paused && status.value === 'active') {
            status.value = 'paused';
        } else if (!exerciseData.paused && status.value === 'paused') {
            status.value = 'active';
        }
      }
      
      // 检查是否达到限制
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
  
  // 开始运动
  async function startExercise(exerciseType) {
    const success = await controlBackend('reset');
    if (success) {
      currentExercise.value = exerciseType
      status.value = 'active'
      count.value = 0
      energy.value = 0
      isPaused.value = false
      analysisResults.value = null
    }
  }
  
  // 停止运动
  function stopExercise() {
    status.value = 'ready'
    isAnalyzing.value = false
  }
  
  // 重置
  async function reset() {
    const success = await controlBackend('reset');
    if (success) {
      count.value = 0
      energy.value = 0
      status.value = 'ready'
      isPaused.value = false
      analysisResults.value = null
      isAnalyzing.value = false
    }
  }
  
  async function pauseWorkout() {
    const success = await controlBackend('pause');
    if (success) {
      status.value = 'paused';
      isPaused.value = true;
    }
  }

  async function resumeWorkout() {
    const success = await controlBackend('resume');
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
    
    // 计算属性
    isActive,
    isPaused,

    // 方法
    analyzeFrame,
    startExercise,
    stopExercise,
    reset,
    pauseWorkout,
    resumeWorkout
  }
})