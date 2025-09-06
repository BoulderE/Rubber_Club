import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useExerciseStore = defineStore('exercise', () => {
  // 运动类型列表
  const exerciseTypes = ref([
    {
      id: 'lateral_raise',
      name: '侧平举',
      description: '锻炼肩部中束',
      icon: '💪'
    },
    {
      id: 'chest_pull',
      name: '拉胸',
      description: '锻炼胸部肌肉',
      icon: '🏋️'
    }
  ])

  const selectedExercise = ref('lateral_raise')

  const exerciseHistory = ref([])

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

  return {
    exerciseTypes,
    selectedExercise,
    exerciseHistory,
    selectExercise,
    addToHistory,
    getExerciseById
  }
})