<template>
  <div v-if="show" class="briefing-overlay" @click.self="handleClose">
    <div class="briefing-content">
      <!-- 右上角关闭按钮 -->
      <button @click="handleClose" class="close-button">&times;</button>

      <div class="briefing-body">
        <!-- 左侧视频 -->
        <div class="video-section">
          <div class="video-wrapper">
            <video
              ref="videoRef"
              :src="videoUrl"
              class="briefing-video"
              muted
              loop
              playsinline
            ></video>
          </div>
        </div>

        <!-- 右侧信息 -->
        <div class="info-section">
          <h2 class="briefing-title">{{ exerciseData?.name }}</h2>
          
          <div class="info-field">
            <span class="field-label">動作說明</span>
            <p class="field-description">{{ exerciseData?.description }}</p>
          </div>

          <div class="info-field" v-if="exerciseData?.tips && exerciseData.tips.length">
            <span class="field-label">動作要點</span>
            <ul class="tips-list">
              <li v-for="(tip, index) in exerciseData.tips" :key="index">{{ tip }}</li>
            </ul>
          </div>

          <!-- 开始按钮 -->
          <div class="action-section">
            <button @click="handleClose" class="btn-start">
              開始訓練
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 音频元素 -->
    <audio ref="audioRef"></audio>
  </div>
</template>

<script setup>
import { ref, computed, watch, onBeforeUnmount, nextTick } from 'vue'

const props = defineProps({
  show: {
    type: Boolean,
    default: false
  },
  exerciseData: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['close', 'start'])

const videoRef = ref(null)
const audioRef = ref(null)

// 根据 exerciseData.id 生成视频 URL
const videoUrl = computed(() => {
  if (!props.exerciseData?.id) {
    console.warn('⚠️ 缺少 exerciseData.id')
    return ''
  }
  const url = `/videos/${props.exerciseData.id}_demo.mp4`
  console.log('🎬 视频 URL:', url)
  return url
})

// 根据 exerciseData.id 生成音频 URL
const audioUrl = computed(() => {
  if (!props.exerciseData?.id) {
    console.warn('⚠️ 缺少 exerciseData.id，无法生成音频 URL')
    return ''
  }
  const url = `/sounds/${props.exerciseData.id}_briefing.mp3`
  console.log('🔊 音频 URL:', url)
  return url
})

// 播放视频
const playVideo = async () => {
  if (!videoRef.value) {
    console.error('❌ 视频元素引用不存在')
    return
  }
  
  try {
    await videoRef.value.play()
    console.log('✅ 视频开始播放')
  } catch (err) {
    console.warn('⚠️ 视频自动播放失败:', err.message)
  }
}

// 播放音频
const playAudio = async () => {
  if (!audioRef.value) {
    console.error('❌ 音频元素引用不存在')
    return
  }
  
  if (!audioUrl.value) {
    console.error('❌ 音频 URL 为空')
    return
  }
  
  try {
    audioRef.value.src = audioUrl.value
    await audioRef.value.play()
    console.log('✅ 音频开始播放:', audioUrl.value)
  } catch (err) {
    console.warn('⚠️ 音频播放失败:', err.message)
  }
}

const handleClose = () => {
  // 停止视频
  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.currentTime = 0
  }
  
  // 停止音频
  if (audioRef.value) {
    audioRef.value.pause()
    audioRef.value.currentTime = 0
  }
  
  emit('close')
}

// 监听弹窗显示状态
watch(() => props.show, async (newVal) => {
  if (!newVal) {
    // 关闭弹窗时停止视频和音频
    if (videoRef.value) {
      videoRef.value.pause()
      videoRef.value.currentTime = 0
    }
    if (audioRef.value) {
      audioRef.value.pause()
      audioRef.value.currentTime = 0
    }
  } else {
    // 打开弹窗时等待 DOM 更新后播放
    await nextTick()
    
    console.log('📍 弹窗已打开，准备播放视频和音频')
    
    // 先播放视频
    if (videoRef.value) {
      console.log('🎬 尝试播放视频...')
      await playVideo()
    }
    
    // 1.5秒后播放音频
    setTimeout(() => {
      console.log('⏰ 1.5秒后，准备播放音频')
      playAudio()
    }, 1500)
  }
})

onBeforeUnmount(() => {
  // 清理视频
  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.src = ''
  }
  
  // 清理音频
  if (audioRef.value) {
    audioRef.value.pause()
    audioRef.value.src = ''
  }
})
</script>

<style scoped>
.briefing-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 20px;
  animation: fadeIn 0.3s ease;
}

.briefing-content {
  position: relative;
  background: white;
  border-radius: 20px;
  max-width: 900px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.close-button {
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(255, 255, 255, 0.9);
  border: none;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  font-size: 1.5rem;
  line-height: 1;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.close-button:hover {
  background: white;
  color: #111827;
  transform: rotate(90deg);
}

.briefing-body {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 0;
  min-height: 500px;
}

.video-section {
  background: #f9fafb;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  border-radius: 20px 0 0 20px;
}

.video-wrapper {
  width: 100%;
  max-width: 400px;
  aspect-ratio: 9 / 16;
  background: #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.briefing-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.info-section {
  padding: 50px 40px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.briefing-title {
  font-size: 2.5rem;
  font-weight: 800;
  color: #111827;
  margin: 0;
  line-height: 1.2;
}

.info-field {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.field-label {
  font-size: 40px;
  font-weight: 700;
  color: #667eea;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.field-description {
  font-size: 40px;
  color: #4b5563;
  line-height: 1.8;
  margin: 0;
}

.tips-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tips-list li {
  font-size: 30px;
  color: #374151;
  padding-left: 28px;
  position: relative;
  line-height: 1.6;
}

.tips-list li::before {
  content: "→";
  position: absolute;
  left: 0;
  color: #667eea;
  font-weight: bold;
  font-size: 1.2rem;
}

.action-section {
  margin-top: auto;
  padding-top: 20px;
}

.btn-start {
  display: block;
  width: 100%;
  padding: 18px 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 12px;
  font-size: 1.2rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn-start:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

/* 动画 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

/* 响应式设计 */
@media (max-width: 900px) {
  .briefing-body {
    grid-template-columns: 1fr;
  }

  .video-section {
    border-radius: 20px 20px 0 0;
    padding: 30px;
    min-height: 300px;
  }

  .info-section {
    padding: 30px 24px;
    gap: 24px;
  }

  .briefing-title {
    font-size: 2rem;
  }

  .field-description {
    font-size: 1rem;
  }

  .tips-list li {
    font-size: 0.95rem;
  }
}

@media (max-width: 640px) {
  .briefing-title {
    font-size: 1.6rem;
  }

  .video-wrapper {
    max-width: 100%;
  }

  .info-section {
    padding: 24px 20px;
  }

  .close-button {
    width: 36px;
    height: 36px;
    font-size: 1.25rem;
  }
}
</style>