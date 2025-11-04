import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useExerciseStore = defineStore('exercise', () => {
  // 运动类型列表
  const exerciseTypes = ref([
    {
      id: 'lateral_raise',
      name: '二頭肌彎舉',
      description: '锻炼肩部中束',
      icon: '💪',
      tips: ['保持身體直立，核心收緊','手臂舉至肩高','控制下放','肘部微屈'],
      imageUrl: '/images/lateral_raise_image_1.png',
      orientation: 'landscape'
    },
    {
      id: 'bicep_curl',
      name: '側平舉',
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
      description: '進階肩部與核心穩定訓練,強化單側肩部力量與身體協調性,改善日常生活中斜向抬舉物品的能力。',
      icon: '🎯',
      tips: [
        '單手持啞鈴,從肩膀斜向推至對側上方',
        '保持核心穩定,避免身體過度旋轉',
        '非訓練側肩膀保持穩定,不可聳肩',
        '控制速度,感受肩部與核心發力',
        '兩側交替訓練,保持平衡'
      ],
      imageUrl: '/images/diagonal_lift_image_1.png',
      orientation: 'landscape' // 横向拍摄
    },
  ])

  const selectedExercise = ref('lateral_raise')
  const exerciseHistory = ref([])  
  const startTime = ref(null) 
  const endTime = ref(null) 

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

  const startExercise = () => { // <-- 新增：开始计时的函数
    startTime.value = Date.now();
    endTime.value = null; // 重置结束时间，以防万一
  }

  const endExercise = () => {   // <-- 新增：结束计时的函数
    endTime.value = Date.now();
  }


  return {
    exerciseTypes,
    selectedExercise,
    exerciseHistory,
    selectExercise,
    addToHistory,
    getExerciseById,
    startTime,
    endTime,
    startExercise,
    endExercise
  }
})