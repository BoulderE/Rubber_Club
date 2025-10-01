import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useExerciseStore = defineStore('exercise', () => {
  // 运动类型列表
  const exerciseTypes = ref([
    {
      id: 'lateral_raise',
      name: '侧平举',
      description: '锻炼肩部中束',
      icon: '💪',
      tips: ['保持身体直立，核心收紧','手臂举至肩高','控制下放','肘部微屈'],
      imageUrl: '/images/Lateral-raise.png'
    },
    {
      id: 'chest_pull',
      name: '拉胸',
      description: '锻炼胸部肌肉',
      icon: '🏋️',
      tips: ['握住拉力器','肩胛后收','控制速度','拉时呼气'],
      imageUrl: '/images/Chest-pull.png'
    },
    {
      id: 'front_raise',
      name: '前平举',
      description: '锻炼肩部前束',
      icon: '💪',
      tips: ['从大腿前抬至与肩同高','核心收紧避免后仰'],
      imageUrl: '/images/Front-raise.png'
    },
    {
      id: 'overhead_press',
      name: '过顶举',
      description: '综合锻炼肩部和手臂力量',
      icon: '🏋️',
      tips: ['起于肩高','推至手腕过头顶','避免过度后仰'],
      imageUrl: '/images/Overhead-press.png'
    },
    {
      id: 'squat',
      name: '深蹲',
      description: '锻炼腿部和臀部力量',
      icon: '🦵',
      tips: ['膝盖对齐脚尖','髋向后坐','站起伸直髋膝'],
      imageUrl: '/images/squat.png'
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