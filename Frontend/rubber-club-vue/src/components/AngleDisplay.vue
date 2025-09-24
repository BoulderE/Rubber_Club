<template>
  <div class="angle-display">
    <h3>Joint Angle</h3>
    
    <div v-for="(angle, joint) in displayAngles" :key="joint" class="angle-item">
      <div class="angle-header">
        <span class="joint-name">{{ jointNames[joint] }}</span>
        <span class="angle-value">{{ angle }}°</span>
      </div>
      <div class="angle-bar">
        <div 
          class="angle-fill" 
          :style="{ 
            width: getAnglePercentage(joint, angle) + '%',
            backgroundColor: getAngleColor(joint, angle)
          }"
        ></div>
      </div>
      <div class="angle-range">
        <span>{{ angleRanges[joint]?.min }}°</span>
        <span>{{ angleRanges[joint]?.max }}°</span>
      </div>
    </div>
    
    <div class="angle-tips">
      <h4>提示</h4>
      <p v-if="exerciseType === 'lateral_raise'">
        • 肩部角度应在 0-90° 之间<br>
        • 肘部保持微屈（150-170°）
      </p>
      <p v-else-if="exerciseType === 'chest_pull'">
        • 肩部角度应在 30-90° 之间<br>
        • 肘部角度应在 90-150° 之间
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  angles: {
    type: Object,
    required: true
  },
  exerciseType: {
    type: String,
    required: true
  }
})

const jointNames = {
  shoulder: '肩部',
  elbow: '肘部',
  hip: '髋部',
  knee: '膝部'
}

const angleRanges = computed(() => {
  if (props.exerciseType === 'lateral_raise') {
    return {
      shoulder: { min: 0, max: 90, optimal: [70, 90] },
      elbow: { min: 90, max: 180, optimal: [150, 170] }
    }
  } else if (props.exerciseType === 'chest_pull') {
    return {
      shoulder: { min: 0, max: 120, optimal: [30, 90] },
      elbow: { min: 45, max: 180, optimal: [90, 150] }
    }
  }
  return {}
})

const displayAngles = computed(() => {
  const filtered = {}
  for (const [joint, angle] of Object.entries(props.angles)) {
    if (angleRanges.value[joint] && typeof angle === 'number') {
      filtered[joint] = Math.round(angle)
    }
  }
  return filtered
})

function getAnglePercentage(joint, angle) {
  const range = angleRanges.value[joint]
  if (!range) return 0
  return ((angle - range.min) / (range.max - range.min)) * 100
}

function getAngleColor(joint, angle) {
  const range = angleRanges.value[joint]
  if (!range || !range.optimal) return '#667eea'
  
  if (angle >= range.optimal[0] && angle <= range.optimal[1]) {
    return '#2ed573' // 绿色 - 最佳范围
  } else if (angle < range.min || angle > range.max) {
    return '#ff4757' // 红色 - 超出范围
  }
  return '#ffa502' // 橙色 - 可接受范围
}
</script>

<style scoped>
.angle-display {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.angle-display h3 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #333;
}

.angle-item {
  margin-bottom: 20px;
}

.angle-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.joint-name {
  color: #666;
  font-size: 14px;
}

.angle-value {
  font-weight: 600;
  color: #333;
  font-size: 18px;
}

.angle-bar {
  height: 8px;
  background: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 4px;
}

.angle-fill {
  height: 100%;
  transition: all 0.3s ease;
}

.angle-range {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #999;
}

.angle-tips {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #f0f0f0;
}

.angle-tips h4 {
  margin-top: 0;
  margin-bottom: 8px;
  color: #333;
  font-size: 14px;
}

.angle-tips p {
  margin: 0;
  font-size: 13px;
  color: #666;
  line-height: 1.6;
}
</style>