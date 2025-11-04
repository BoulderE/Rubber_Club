<template>
  <div class="home-view">
    <div class="hero-section">
      <h1>Rubber Club</h1>
      <p>Your Digital Fitness Helper</p>
    </div>

    <!-- 难度选择弹窗 -->
    <div v-if="showModal" class="modal-backdrop" @click.self="showModal = false">
      <div class="modal-content">
        <h2 id="modal-title">選擇您的級別「{{ selectedExercise?.displayName }}」</h2>
        <div class="level-options">
          <label>
            <input type="radio" name="level" value="beginner" v-model="selectedLevel">
            <div class="level-card">
              <span class="emoji">🥳</span>
              <div>初學者</div>
              <p>輕鬆上手。</p>
            </div>
          </label>
          <label>
            <input type="radio" name="level" value="intermediate" v-model="selectedLevel">
            <div class="level-card">
              <span class="emoji">🎯</span>
              <div>進階</div>
              <p>嚴格指導。</p>
            </div>
          </label>
        </div>
        <div class="modal-buttons">
          <button class="cancel-btn" @click="showModal = false">Cancel</button>
          <button class="confirm-btn" @click="startExercise">Confirm Selection</button>
        </div>
      </div>
    </div>

    <!-- 聊天机器人弹窗 -->
    <div v-if="isChatbotVisible" class="modal-backdrop" @click.self="isChatbotVisible = false">
      <ChatbotWindow 
        @close="isChatbotVisible = false"
        class="chatbot-container"
      />
    </div>

    <!-- 详情弹窗 -->
    <div 
      v-if="showDetailModal"
      class="detail-modal-overlay"
      @click.self="closeDetail"
    >
      <div class="detail-modal-content">
        <button @click="closeDetail" class="close-button">&times;</button>
        
        <div class="modal-body">
          <!-- 左侧图片 -->
          <div class="image-section">
            <img 
              :src="detailExercise?.imageUrl" 
              :alt="detailExercise?.name"
              class="detail-image"
            >
          </div>

          <!-- 右侧信息 -->
          <div class="info-section">
            <h2 class="detail-title">{{ detailExercise?.displayName }}</h2>
            
            <div class="detail-field">
              <span class="field-label">動作方向</span>
              <span class="field-value">{{ detailExercise?.orientation === 'portrait' ? '縱向' : '橫向' }}</span>
            </div>

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

            <!-- 开始训练按钮 -->
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

    <!-- 运动卡片网格 -->
    <div class="exercises-grid">
      <div 
        v-for="exercise in exercises" 
        :key="exercise.id"
        class="exercise-card"
        @mouseenter="handleMouseEnter(exercise.id)"
        @mouseleave="handleMouseLeave(exercise.id)"
        @click="openDifficultyModal(exercise)"
      >
        <!-- 视频容器 -->
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

        <!-- 卡片底部信息 -->
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

    <button @click="isChatbotVisible = true" id="need-help-fab">?</button>
  </div>
</template>

<script setup>
import ChatbotWindow from '@/components/ChatbotWindow.vue'; 
import { ref, onBeforeUnmount } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

const isChatbotVisible = ref(false);
const showModal = ref(false); 
const showDetailModal = ref(false); // 详情弹窗控制
const selectedExercise = ref(null); 
const detailExercise = ref(null); // 详情弹窗的运动数据
const selectedLevel = ref('beginner');

// 视频引用存储
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
    orientation: 'portrait',
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
    orientation: 'landscape', // 横向拍摄
    tips: [
      '單手持啞鈴,從肩膀斜向推至對側上方',
      '保持核心穩定,避免身體過度旋轉',
      '非訓練側肩膀保持穩定,不可聳肩',
      '控制速度,感受肩部與核心發力',
      '兩側交替訓練,保持平衡'
    ]
  }
]);

// 设置视频引用
const setVideoRef = (el, id) => {
  if (el) {
    videoRefs.value[id] = el;
  }
};

// 鼠标进入 - 播放视频
const handleMouseEnter = (id) => {
  const video = videoRefs.value[id];
  if (video) {
    video.currentTime = 0;
    video.play().catch(err => {
      console.log('Video play failed:', err);
    });
  }
};

// 鼠标离开 - 暂停视频
const handleMouseLeave = (id) => {
  const video = videoRefs.value[id];
  if (video) {
    video.pause();
    video.currentTime = 0;
  }
};

// 打开难度选择弹窗（点击整个卡片）
function openDifficultyModal(exercise) {
  selectedExercise.value = exercise;
  selectedLevel.value = 'beginner';
  showModal.value = true;
}

// 打开详情弹窗（点击"更多..."）
function openDetail(exercise) {
  detailExercise.value = exercise;
  showDetailModal.value = true;
}

// 关闭详情弹窗
function closeDetail() {
  showDetailModal.value = false;
  detailExercise.value = null;
}

// 从详情页开始训练（先打开难度选择）
function startFromDetail() {
  selectedExercise.value = detailExercise.value;
  selectedLevel.value = 'beginner';
  showDetailModal.value = false;
  showModal.value = true;
}

// 开始运动（从难度选择弹窗）
function startExercise() {
  if (!selectedExercise.value) return;

  router.push({ 
    name: 'exercise', 
    params: { type: selectedExercise.value.id },
    query: { style: selectedLevel.value }
  });

  showModal.value = false;
}

// 组件卸载时清理视频
onBeforeUnmount(() => {
  Object.values(videoRefs.value).forEach(video => {
    if (video) {
      video.pause();
      video.src = '';
    }
  });
});
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
  text-align: center;
  padding: 40px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
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

/* ========== 网格布局 - 4列 ========== */
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
  cursor: pointer; /* 添加指针光标 */
}

.exercise-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

/* ========== 视频容器 - 9:16 纵向 ========== */
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

/* 难度标签 */
.difficulty-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(255, 255, 255, 0.95);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  backdrop-filter: blur(8px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 卡片底部 */
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

/* ========== 难度选择弹窗 ========== */
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
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  padding: 30px;
  width: 90%;
  max-width: 500px;
  text-align: center;
  animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.modal-content h2 { 
  margin-top: 0; 
  margin-bottom: 25px;
  font-size: 1.5rem;
  color: #111827;
}

.level-options { 
  display: flex; 
  gap: 20px; 
  margin-bottom: 30px; 
}

.level-options input[type="radio"] { 
  display: none; 
}

.level-options label { 
  flex: 1; 
  cursor: pointer; 
}

.level-card { 
  padding: 20px; 
  border: 2px solid #e0e0e0; 
  border-radius: 12px; 
  transition: all 0.2s ease; 
  text-align: left; 
}

.level-card .emoji { 
  font-size: 1.5em; 
  margin-right: 10px; 
}

.level-card div { 
  font-weight: bold; 
  font-size: 1.1em; 
  color: #111827;
}

.level-card p { 
  font-size: 0.9em; 
  color: #6c757d; 
  margin: 5px 0 0; 
}

.level-options input[type="radio"]:checked + .level-card {
  border-color: #667eea;
  background-color: #f3f1ff;
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
}

.modal-buttons {
  display: flex;
  gap: 15px;
}

.confirm-btn {
  flex: 1; 
  padding: 15px;
  font-size: 1.1em;
  font-weight: bold;
  color: white;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.confirm-btn:hover { 
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.cancel-btn {
  flex: 1; 
  padding: 15px;
  font-size: 1.1em;
  font-weight: bold;
  color: white; 
  background-color: #e74c3c;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
}

.cancel-btn:hover {
  background-color: #c0392b;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(231, 76, 60, 0.4);
}

/* ========== 详情弹窗 ========== */
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

.modal-body {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 0;
  min-height: 500px;
}

/* 左侧图片区域 */
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

/* 右侧信息区域 */
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

/* ========== 帮助按钮 FAB ========== */
#need-help-fab {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 60px;
  height: 60px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 50%;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  font-weight: bold;
  box-shadow: 0 5px 15px rgba(106, 90, 249, 0.4);
  cursor: pointer;
  z-index: 999;
  transition: all 0.3s ease;
}

#need-help-fab:hover {
  transform: scale(1.1);
  box-shadow: 0 8px 24px rgba(106, 90, 249, 0.5);
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

/* ========== 响应式设计 ========== */
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
    flex-direction: column;
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
</style>