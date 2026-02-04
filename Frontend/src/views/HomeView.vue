<template>
  <div class="home-view">
    <div class="hero-section">
      <div class="hero-container">
        <div class="hero-left">
          <h1>Rubber Club</h1>
          <p>Your Digital Fitness Helper</p>
        </div>
        <div class="hero-center" v-if="authStore.isLoggedIn">
          <div class="welcome-line">
            <span class="welcome-icon">👋</span>
            <span>  歡迎</span>
          </div>
          <div class="user-name">{{ authStore.userName || '用戶' }}</div>
        </div>
        <div class="hero-right">
          <div class="playlist-card" @click="startPlaylistMode">
            <div class="playlist-info">
              <h3>整組訓練</h3>
              <p>6 個動作完整鍛鍊</p>
            </div>
            <span class="playlist-arrow">→</span>
          </div>
        </div>
      </div>
    </div>    

    <div v-if="showModal" class="modal-backdrop" @click.self="showModal = false">
      <div class="modal-content">
        <button @click="showModal = false" class="close-button">&times;</button>
        
        <h2 id="modal-title">選擇您的「{{ selectedExercise?.displayName }}」難度</h2>
        
        <div class="level-options">
          <div 
            class="level-card"
            :class="{ selected: selectedLevel === 'beginner' }"
            @click="selectAndStart('beginner')"
          >
            <div>初學者</div>
            <p>輕鬆上手</p>
          </div>
          
          <div 
            class="level-card"
            :class="{ selected: selectedLevel === 'advanced' }"
            @click="selectAndStart('advanced')"
          >
            <div>進階</div>
            <p>嚴格指導</p>
          </div>
        </div>
      </div>
    </div>

    <div v-if="isChatbotVisible" class="modal-backdrop" @click.self="isChatbotVisible = false">
      <ChatbotWindow 
        @close="isChatbotVisible = false"
        class="chatbot-container"
      />
    </div>
    <div 
      v-if="showDetailModal"
      class="detail-modal-overlay"
      @click.self="closeDetail"
    >
      <div class="detail-modal-content">
        <button @click="closeDetail" class="close-button">&times;</button>
        
        <div class="modal-body">
          <div class="image-section">
            <img 
              :src="detailExercise?.imageUrl" 
              :alt="detailExercise?.name"
              class="detail-image"
            >
          </div>

          <div class="info-section">
            <h2 class="detail-title">{{ detailExercise?.displayName }}</h2>
            
            <div class="detail-field">
              <span class="field-label">動作說明</span>
              <p class="field-description">{{ detailExercise?.description }}</p>
            </div>

            <div class="detail-field" v-if="detailExercise?.tips && detailExercise.tips.length">
              <span class="field-label">動作要點</span>
              <ul class="tips-list">
                <li v-for="(tip, index) in detailExercise.tips" :key="index">{{ tip }}</li>
              </ul>
            </div>

            <div class="modal-actions">
              <button 
                @click="startFromDetail"
                class="btn btn-start"
              >
                開始訓練
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="exercises-grid">
      <div 
        v-for="exercise in exercises" 
        :key="exercise.id"
        class="exercise-card"
        @mouseenter="handleMouseEnter(exercise.id)"
        @mouseleave="handleMouseLeave(exercise.id)"
        @click="openDifficultyModal(exercise)"
      >
        <div class="video-container">
          <video
            :ref="el => setVideoRef(el, exercise.id)"
            :src="exercise.videoUrl"
            class="exercise-video"
            muted
            loop
            playsinline
          ></video>
        </div>

        <div class="card-footer">
          <h3 class="exercise-name">{{ exercise.displayName }}</h3>
          <button 
            class="more-btn"
            @click.stop="openDetail(exercise)"
          >
            更多...
          </button>
        </div>
      </div>
    </div>
    <div class="fab-group">
      <button @click="goToHistory" class="fab-btn history-fab" title="運動歷史">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
      </button>
      <button @click="isChatbotVisible = true" class="fab-btn help-fab" title="需要幫助">
        ?
      </button>
    </div>
  </div>
</template>

<script setup>
import ChatbotWindow from '@/components/ChatbotWindow.vue'; 
import { ref, onBeforeUnmount } from 'vue';
import { useRouter } from 'vue-router';
import { useExerciseStore } from '@/stores/exercise'
import { useAuthStore } from '@/stores/auth';
const authStore = useAuthStore();
const router = useRouter();

const isChatbotVisible = ref(false);
const showModal = ref(false); 
const showDetailModal = ref(false);
const selectedExercise = ref(null); 
const detailExercise = ref(null);
const selectedLevel = ref('beginner');

const videoRefs = ref({});

const exercises = ref([
  { 
    id: 'bicep_curl', 
    name: '二頭肌彎舉 - 把手從大腿旁提到肩膀的彎手動作',
    displayName: '二頭肌彎舉',
    description: '通過該動作增強肩膀穩定度與肌群，幫助長者改善肩膀穩定與受控能力。溫和啟動胸背與肩部穩定肌群，改善肩帶穩定與姿勢控制。', 
    imageUrl: '/images/bicep_curl_image_1.png',
    videoUrl: '/videos/bicep_curl_demo.mp4',
    orientation: 'portrait',
    tips: [
      '保持核心穩定，避免身體搖晃',
      '動作要緩慢控制，感受肌肉收縮',
      '肘關節保持在身體兩側'
    ]
  },
  { 
    id: 'lateral_raise', 
    name: '側平舉 - 把手從身體兩側平舉到肩膀高度',
    displayName: '側平舉',
    description: '針對三角肌外側的輕量訓練，協助長者提升抬臂與側向拿取物品的能力。', 
    imageUrl: '/images/lateral_raise_image_1.png',
    videoUrl: '/videos/lateral_raise_demo.mp4',
    orientation: 'portrait',
    tips: [
      '手臂微彎，避免完全伸直',
      '抬起時不要聳肩',
      '控制下放速度'
    ]
  },
  { 
    id: 'chest_pull', 
    name: '胸部側拉 - 模擬打開櫃子門的動作',
    displayName: '胸部側拉',
    description: '強化上背與肩後肌群的穩健訓練，協助長者改善肩胛後收與挺胸姿勢。', 
    imageUrl: '/images/chest_pull_image_1.png',
    videoUrl: '/videos/chest_pull_demo.mp4',
    orientation: 'landscape',
    tips: [
      '挺胸收腹，保持良好姿勢',
      '肩胛骨向後夾緊',
      '感受胸部拉伸'
    ]
  },
  { 
    id: 'front_raise', 
    name: '前平舉 - 把手從大腿前方平舉到眼前高度',
    displayName: '前平舉',
    description: '強化前三角肌與肩前穩定度，幫助長者安全抬手至胸前/眼前高度。', 
    imageUrl: '/images/front_raise_image_1.png',
    videoUrl: '/videos/front_raise_demo.mp4',
    orientation: 'portrait',
    tips: [
      '保持手臂伸直但不鎖死',
      '抬起高度不超過肩膀',
      '避免使用慣性'
    ]
  },
  { 
    id: 'overhead_press', 
    name: '過頭推舉 - 把手從肩膀以下推舉到頭頂上方',
    displayName: '過頭推舉',
    description: '逐步訓練肩部與上背推舉能力，協助長者改善頭上取物與伸手動作。', 
    imageUrl: '/images/overhead_press_image_1.png',
    videoUrl: '/videos/overhead_press_demo.mp4',
    orientation: 'portrait',
    tips: [
      '核心收緊，避免腰部過度後仰',
      '推舉時保持肘部向前',
      '頂端時手臂接近伸直'
    ]
  },
  { 
    id: 'diagonal_lift', 
    name: '對角線啞鈴推舉 - 單手從肩膀斜向推至對側上方',
    displayName: '對角線推舉',
    description: '進階肩部與核心穩定訓練,強化單側肩部力量與身體協調性,改善日常生活中斜向抬舉物品的能力。', 
    imageUrl: '/images/diagonal_lift_image_1.png',
    videoUrl: '/videos/diagonal_lift_demo.mp4',
    orientation: 'landscape',
    tips: [
      '單手持啞鈴,從肩膀斜向推至對側上方',
      '保持核心穩定,避免身體過度旋轉',
      '非訓練側肩膀保持穩定,不可聳肩',
      '控制速度,感受肩部與核心發力',
      '兩側交替訓練,保持平衡'
    ]
  }
]);

const setVideoRef = (el, id) => {
  if (el) {
    videoRefs.value[id] = el;
  }
};

const handleMouseEnter = (id) => {
  const video = videoRefs.value[id];
  if (video) {
    video.currentTime = 0;
    video.play().catch(err => {
      console.log('Video play failed:', err);
    });
  }
};

const handleMouseLeave = (id) => {
  const video = videoRefs.value[id];
  if (video) {
    video.pause();
    video.currentTime = 0;
  }
};

function openDifficultyModal(exercise) {
  selectedExercise.value = exercise;
  selectedLevel.value = 'beginner';
  showModal.value = true; 
}

function selectAndStart(level) {
  selectedLevel.value = level;
  startExercise(); 
}

function openDetail(exercise) {
  detailExercise.value = exercise;
  showDetailModal.value = true;
}

function closeDetail() {
  showDetailModal.value = false;
  detailExercise.value = null;
}

function startFromDetail() {
  selectedExercise.value = detailExercise.value;
  selectedLevel.value = 'beginner';
  showDetailModal.value = false;
  showModal.value = true; 
}

function startExercise() {
  if (!selectedExercise.value) return;

  router.push({ 
    name: 'exercise', 
    params: { type: selectedExercise.value.id },
    query: { 
      style: selectedLevel.value,
      
      autoPlayVoice: 'true' 
    }
  });

  showModal.value = false; 
}

onBeforeUnmount(() => {
  Object.values(videoRefs.value).forEach(video => {
    if (video) {
      video.pause();
      video.src = '';
    }
  });
});

// new
const exerciseStore = useExerciseStore()
function startPlaylistMode() {
  const defaultPlaylist = ['bicep_curl', 'lateral_raise', 'front_raise', 'overhead_press', 'chest_pull', 'diagonal_lift']
  const firstExercise = exerciseStore.startPlaylist(defaultPlaylist)
  
  router.push({
    name: 'exercise',
    params: { type: firstExercise },
    query: {
      style: 'beginner',
      autoPlayVoice: 'true'
    }
  })
}

function goToHistory() {
  router.push('/history')
}
</script>

<style scoped>
.home-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  background-color: #f0f2f5;
  padding-bottom: 100px;
}

.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 40px 24px;
  text-align: left;
  border-radius: 20px;
  margin-bottom: 40px;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.hero-section h1 {
  font-size: 48px;
  margin-bottom: 16px;
  font-weight: 700;
}

.hero-section p {
  font-size: 20px;
  opacity: 0.9;
  margin: 0;
}

.exercises-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  padding: 20px 0;
}

.exercise-card {
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
}

.exercise-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 9 / 16;
  background: #e5e7eb;
  overflow: hidden;
}

.exercise-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.card-footer {
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  background: white;
}

.exercise-name {
  font-size: 1rem;
  font-weight: 700;
  color: #111827;
  margin: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.more-btn {
  background: none;
  border: none;
  color: #667eea;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 6px;
  transition: all 0.2s ease;
  white-space: nowrap;
  flex-shrink: 0;
}

.more-btn:hover {
  background: #f3f4f6;
  color: #764ba2;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.3s ease;
}

.modal-content {
  position: relative; 
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  padding: 40px;
  width: 90%;
  max-width: 500px;
  text-align: center;
  animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.modal-content .close-button {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 36px;
  height: 36px;
  background: #f3f4f6;
  border: none;
  border-radius: 50%;
  font-size: 24px;
  line-height: 1;
  color: #6b7280;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  z-index: 10;
}

.modal-content .close-button:hover {
  background: #e5e7eb;
  color: #374151;
  transform: rotate(90deg);
}

.modal-content h2 { 
  margin-top: 0; 
  margin-bottom: 30px;
  font-size: 1.75rem;
  color: #111827;
  padding-right: 30px; 
}


.level-options { 
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 0; 
}

.level-card { 
  padding: 24px 16px;
  border: 2px solid #e5e7eb;
  border-radius: 16px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  text-align: center;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.level-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
  opacity: 0;
  transition: opacity 0.3s ease;
}

.level-card:hover {
  border-color: #667eea;
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
}

.level-card:hover::before {
  opacity: 1;
}

.level-card.selected {
  border-color: #667eea;
  background-color: #f3f1ff;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
}

.level-card div { 
  font-weight: 600;
  font-size: 1.75rem;
  color: #1f2937;
  margin-bottom: 8px;
  position: relative;
  z-index: 1;
}

.level-card p { 
  font-size: 1.25rem;
  color: #6b7280;
  margin: 0;
  position: relative;
  z-index: 1;
}

.detail-modal-overlay {
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

.detail-modal-content {
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

.detail-modal-content .close-button {
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

.detail-modal-content .close-button:hover {
  background: white;
  color: #111827;
  transform: rotate(90deg);
}

.modal-body {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 0;
  min-height: 500px;
}

.image-section {
  background: #f9fafb;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  border-radius: 20px 0 0 20px;
}

.detail-image {
  width: 100%;
  height: auto;
  max-height: 600px;
  object-fit: contain;
  border-radius: 12px;
}

.info-section {
  padding: 50px 40px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.detail-title {
  font-size: 2.5rem;
  font-weight: 800;
  color: #111827;
  margin: 0;
  line-height: 1.2;
}

.detail-field {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.field-label {
  font-size: 1.1rem;
  font-weight: 700;
  color: #667eea;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.field-value {
  font-size: 1.3rem;
  font-weight: 600;
  color: #1f2937;
}

.field-description {
  font-size: 1.15rem;
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
  font-size: 1.05rem;
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

.modal-actions {
  margin-top: auto;
  padding-top: 20px;
}

.btn-start {
  display: block;
  width: 100%;
  padding: 18px 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-decoration: none;
  text-align: center;
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

.fab-group {
  position: fixed;
  bottom: 30px;
  right: 30px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  z-index: 999;
}

.fab-btn {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.history-fab {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
}

.history-fab:hover {
  transform: scale(1.1);
  box-shadow: 0 8px 24px rgba(16, 185, 129, 0.5);
}

.help-fab {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 24px;
  font-weight: bold;
}

.help-fab:hover {
  transform: scale(1.1);
  box-shadow: 0 8px 24px rgba(106, 90, 249, 0.5);
}

@media (max-width: 640px) {
  .fab-group {
    bottom: 24px;
    right: 24px;
    gap: 12px;
  }

  .fab-btn {
    width: 56px;
    height: 56px;
  }

  .help-fab {
    font-size: 20px;
  }
}

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

@media (max-width: 1200px) {
  .exercises-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 900px) {
  .exercises-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
  }

  .hero-section h1 {
    font-size: 36px;
  }

  .hero-section p {
    font-size: 18px;
  }

  .modal-body {
    grid-template-columns: 1fr;
  }

  .image-section {
    border-radius: 20px 20px 0 0;
    padding: 30px;
    min-height: 300px;
  }

  .info-section {
    padding: 30px 24px;
    gap: 24px;
  }

  .detail-title {
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
  .exercises-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }

  .hero-section {
    padding: 30px 20px;
  }

  .hero-section h1 {
    font-size: 28px;
  }

  .hero-section p {
    font-size: 16px;
  }

  .exercise-name {
    font-size: 0.9rem;
  }

  #need-help-fab {
    width: 56px;
    height: 56px;
    bottom: 24px;
    right: 24px;
    font-size: 20px;
  }

  .level-options {
    grid-template-columns: 1fr;
  }

  .modal-content {
    padding: 32px 24px;
  }

  .detail-title {
    font-size: 1.6rem;
  }
}

@media (max-width: 480px) {
  .exercises-grid {
    grid-template-columns: 1fr;
  }
}

.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 60px 24px;
}

.hero-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: 40px;
}

.hero-left {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 24px;
  min-height: 120px;
  background: transparent;
  border: none;
}

.hero-left h1 {
  font-size: 3.5rem;
  font-weight: 800;
  color: white;
  margin: 0 0 16px 0;
}

.hero-title {
  font-size: 3rem;
  font-weight: 800;
  color: white;
  margin: 0 0 8px 0;
}

.hero-left p {
  font-size: 1.25rem;
  color: rgba(255, 255, 255, 0.9);
  margin: 0 0 16px 0;
}

.hero-desc {
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.7);
  margin: 0;
}

.hero-right {
  flex: 1;
  display: flex;
  justify-content: flex-end;
}

.playlist-card {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  padding: 24px 32px;
  display: flex;
  align-items: center;
  gap: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 300px;
}

.playlist-card:hover {
  background: #ffffff40;
  transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
}

.playlist-icon {
  font-size: 2.5rem;
}

.playlist-info h3 {
  font-size: 36px;
  font-weight: 700;
  color: white;
  margin: 0 0 4px 0;
}

.playlist-info p {
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
}

.playlist-arrow {
  font-size: 1.5rem;
  color: white;
  margin-left: auto;
  transition: transform 0.3s ease;
}

.playlist-card:hover .playlist-arrow {
  transform: translateX(4px);
}

@media (max-width: 768px) {
  .hero-container {
    flex-direction: column;
    text-align: center;
  }
  
  .hero-right {
    justify-content: center;
    width: 100%;
  }
  
  .playlist-card {
    width: 100%;
    max-width: 320px;
  }
  
  .hero-title {
    font-size: 2rem;
  }
}

.hero-center {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 24px;
  padding: 24px 48px;
  gap: 8px;
}

.hero-center .welcome-line {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 36px;
  font-weight: 700;
  color: white;
}

.hero-center .welcome-icon {
  font-size: 36px;
  animation: wave 1.5s ease-in-out infinite;
}

.hero-center .user-name {
  font-size: 36px;
  font-weight: 700;
  color: white;
}

@keyframes wave {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(20deg); }
  75% { transform: rotate(-10deg); }
}

@media (max-width: 768px) {
  .hero-container {
    flex-direction: column;
    text-align: center;
  }
  
  .hero-center {
    order: 2;
    margin: 16px 0;
    padding: 14px 28px;
  }
  
  .hero-right {
    justify-content: center;
    width: 100%;
    order: 3;
  }
  
  .playlist-card {
    width: 100%;
    max-width: 320px;
  }
  
  .hero-title {
    font-size: 2rem;
  }
}
</style>