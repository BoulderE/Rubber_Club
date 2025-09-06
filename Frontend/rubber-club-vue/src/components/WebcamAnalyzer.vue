<template>
  <div class="webcam-analyzer">
    <div class="video-container">
      <video 
        ref="videoElement" 
        class="video-feed"
        :class="{ 'mirror': mirror }"
        autoplay
        playsinline
      ></video>
      
      <canvas 
        ref="canvasElement" 
        class="pose-overlay"
        :class="{ 'mirror': mirror }"
      ></canvas>
      
      <div v-if="!isStreaming" class="video-placeholder">
        <div class="placeholder-content">
          <p>📹 摄像头未启动</p>
          <button @click="startCamera" class="start-button">启动摄像头</button>
        </div>
      </div>
      
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>
    
    <div class="controls">
      <button @click="toggleCamera" class="control-button">
        {{ isStreaming ? '停止' : '开始' }}
      </button>
      <button @click="toggleMirror" class="control-button">
        {{ mirror ? '取消镜像' : '镜像' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useMediapipeStore } from '@/stores/mediapipe'

const props = defineProps({
  analyzeInterval: {
    type: Number,
    default: 100
  }
})

const emit = defineEmits(['frame-analyzed', 'error'])

const mediapipeStore = useMediapipeStore()

const videoElement = ref(null)
const canvasElement = ref(null)
const isStreaming = ref(false)
const mirror = ref(true)
const error = ref('')
const stream = ref(null)
const analyzeTimer = ref(null)

async function startCamera() {
  try {
    error.value = ''
    
    // 获取摄像头权限
    stream.value = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    })
    
    if (videoElement.value) {
      videoElement.value.srcObject = stream.value
      isStreaming.value = true
      
      // 等待视频元数据加载
      videoElement.value.onloadedmetadata = () => {
        // 设置画布大小
        if (canvasElement.value) {
          canvasElement.value.width = videoElement.value.videoWidth
          canvasElement.value.height = videoElement.value.videoHeight
        }
        
        // 开始分析
        startAnalysis()
      }
    }
  } catch (err) {
    error.value = '无法访问摄像头：' + err.message
    emit('error', err)
  }
}

function stopCamera() {
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop())
    stream.value = null
  }
  
  if (videoElement.value) {
    videoElement.value.srcObject = null
  }
  
  isStreaming.value = false
  stopAnalysis()
}

function toggleCamera() {
  if (isStreaming.value) {
    stopCamera()
  } else {
    startCamera()
  }
}

function toggleMirror() {
  mirror.value = !mirror.value
}

function startAnalysis() {
  if (analyzeTimer.value) {
    clearInterval(analyzeTimer.value)
  }
  
  analyzeTimer.value = setInterval(async () => {
    if (videoElement.value && isStreaming.value) {
      await analyzeFrame()
    }
  }, props.analyzeInterval)
}

function stopAnalysis() {
  if (analyzeTimer.value) {
    clearInterval(analyzeTimer.value)
    analyzeTimer.value = null
  }
}

async function analyzeFrame() {
  if (!videoElement.value || !canvasElement.value) return
  
  try {
    const ctx = canvasElement.value.getContext('2d')
    
    // 如果是镜像模式，翻转画布
    if (mirror.value) {
      ctx.save()
      ctx.scale(-1, 1)
      ctx.drawImage(
        videoElement.value, 
        -canvasElement.value.width, 
        0, 
        canvasElement.value.width, 
        canvasElement.value.height
      )
      ctx.restore()
    } else {
      ctx.drawImage(
        videoElement.value, 
        0, 
        0, 
        canvasElement.value.width, 
        canvasElement.value.height
      )
    }
    
    // 获取图像数据
    const imageData = canvasElement.value.toDataURL('image/jpeg', 0.8)
    
    // 调用 store 中的分析方法
    const result = await mediapipeStore.analyzeFrame(imageData)
    
    if (result) {
      emit('frame-analyzed', result)
      
      // 在画布上绘制骨架（如果有姿势数据）
      if (result.pose_landmarks) {
        drawPose(ctx, result.pose_landmarks)
      }
    }
  } catch (err) {
    console.error('Frame analysis error:', err)
    emit('error', err)
  }
}

function drawPose(ctx, landmarks) {
  if (!landmarks || landmarks.length === 0) return
  
  ctx.strokeStyle = '#00ff00'
  ctx.lineWidth = 2
  ctx.fillStyle = '#ff0000'
  
  // 绘制关键点
  landmarks.forEach(landmark => {
    if (landmark.visibility > 0.5) {
      const x = mirror.value 
        ? canvasElement.value.width - (landmark.x * canvasElement.value.width)
        : landmark.x * canvasElement.value.width
      const y = landmark.y * canvasElement.value.height
      
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, 2 * Math.PI)
      ctx.fill()
    }
  })
  
  // 绘制骨架连接线
  const connections = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // 上半身
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28] // 下半身
  ]
  
  ctx.beginPath()
  connections.forEach(([start, end]) => {
    if (landmarks[start]?.visibility > 0.5 && landmarks[end]?.visibility > 0.5) {
      const x1 = mirror.value 
        ? canvasElement.value.width - (landmarks[start].x * canvasElement.value.width)
        : landmarks[start].x * canvasElement.value.width
      const y1 = landmarks[start].y * canvasElement.value.height
      const x2 = mirror.value 
        ? canvasElement.value.width - (landmarks[end].x * canvasElement.value.width)
        : landmarks[end].x * canvasElement.value.width
      const y2 = landmarks[end].y * canvasElement.value.height
      
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
    }
  })
  ctx.stroke()
}

// 公开方法给父组件
defineExpose({
  startAnalysis,
  stopAnalysis
})

onMounted(() => {
  // 自动启动摄像头
  startCamera()
})

onUnmounted(() => {
  stopCamera()
})
</script>

<style scoped>
.webcam-analyzer {
  width: 100%;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.video-container {
  position: relative;
  width: 100%;
  padding-bottom: 56.25%; /* 16:9 比例 */
  background: #000;
  overflow: hidden;
}

.video-feed,
.pose-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.pose-overlay {
  pointer-events: none;
}

.mirror {
  transform: scaleX(-1);
}

.video-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
}

.placeholder-content {
  text-align: center;
}

.placeholder-content p {
  font-size: 24px;
  margin-bottom: 20px;
  color: #666;
}

.start-button {
  padding: 12px 30px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s;
}

.start-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.error-message {
  position: absolute;
  top: 20px;
  left: 20px;
  right: 20px;
  background: #ff4757;
  color: white;
  padding: 10px 15px;
  border-radius: 8px;
  font-size: 14px;
}

.controls {
  display: flex;
  gap: 10px;
  padding: 15px;
  background: #f8f9fa;
}

.control-button {
  flex: 1;
  padding: 10px 20px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s;
}

.control-button:hover {
  background: #f0f0f0;
  border-color: #667eea;
  color: #667eea;
}
</style>