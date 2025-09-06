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
  
  // 运动限制
  const repetitionLimit = 20
  
  // 计算属性
  const isActive = computed(() => status.value === 'active')
  
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
      
      // 处理运动计数
      if (result.lateral_raise?.count) {
        count.value = result.lateral_raise.count
      } else if (result.chest_pull?.count) {
        count.value = result.chest_pull.count
      }
      
      // 根据当前运动类型更新数据
      if (currentExercise.value === 'lateral_raise' && result.lateral_raise) {
        count.value = result.lateral_raise.count
        energy.value = Math.round(result.lateral_raise.energy)
      } else if (currentExercise.value === 'chest_pull' && result.chest_pull) {
        count.value = result.chest_pull.count
        energy.value = Math.round(result.chest_pull.energy)
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
  function startExercise(exerciseType) {
    currentExercise.value = exerciseType
    status.value = 'active'
    count.value = 0
    energy.value = 0
    analysisResults.value = null
  }
  
  // 停止运动
  function stopExercise() {
    status.value = 'ready'
    isAnalyzing.value = false
  }
  
  // 重置
  function reset() {
    count.value = 0
    energy.value = 0
    status.value = 'ready'
    analysisResults.value = null
    isAnalyzing.value = false
  }
  
  // 更新运动数据（用于手动更新）
  function updateExerciseData(data) {
    if (data.count !== undefined) count.value = data.count
    if (data.energy !== undefined) energy.value = data.energy
    if (data.status !== undefined) status.value = data.status
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
    
    // 方法
    analyzeFrame,
    startExercise,
    stopExercise,
    reset,
    updateExerciseData
  }
})