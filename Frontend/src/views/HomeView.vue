<template>
  <div class="home-view">
    <div class="hero-section">
      <h1>Rubber Club</h1>
      <p>Your Digital Fitness Helper</p>
    </div>

     <div v-if="showModal" class="modal-backdrop">
      <div class="modal-content">
        <h2 id="modal-title">Select your level for「{{ selectedExercise.name }}」</h2>
        <div class="level-options">
          <label>
            <input type="radio" name="level" value="beginner" v-model="globalLevel">
            <div class="level-card">
              <span class="emoji">🥳</span>
              <div>Beginner</div>
              <p>Get started with ease.</p>
            </div>
          </label>
          <label>
            <input type="radio" name="level" value="intermediate" v-model="globalLevel">
            <div class="level-card">
              <span class="emoji">🎯</span>
              <div>Intermediate</div>
              <p>Strict guidance.</p>
            </div>
          </label>
        </div>
        <div class="modal-buttons">
          <button class="cancel-btn" @click="showModal = false">Cancel</button>
          <button class="confirm-btn" @click="startExercise">Confirm Selection</button>
        </div>
      </div>
    </div>

     <div v-if="isChatbotVisible" class="modal-backdrop" @click.self="isChatbotVisible = false">
      <ChatbotWindow 
        @close="isChatbotVisible = false"
        class="chatbot-container"
      />
    </div>

    <div class="exercise-cards">
      <div 
        v-for="exercise in exercises" 
        :key="exercise.id"
        class="exercise-card"
        @click="openDifficultyModal(exercise)"
      >
        <div class="card-icon">{{ exercise.icon }}</div>
        <h3>{{ exercise.name }}</h3>
        <p>{{ exercise.description }}</p>
        <span class="card-arrow">→</span>
      </div>
    </div>
    <button @click="isChatbotVisible = true" id="need-help-fab">?</button>
  </div>
</template>

<script setup>
import ChatbotWindow from '@/components/ChatbotWindow.vue'; 
import { ref } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

const isChatbotVisible = ref(false);
const showModal = ref(false); 
const selectedExercise = ref(null); 
const selectedLevel = ref('beginner');

function openDifficultyModal(exercise) {
  selectedExercise.value = exercise; // 记住是哪个运动
  selectedLevel.value = 'beginner'; // 每次打开都重置为默认值
  showModal.value = true; // 显示弹窗
}

const exercises = ref([
  { 
    id: 'bicep_curl', 
    name: 'Bicep Curl', 
    description: '溫和啟動胸背與肩部穩定肌群，幫助長者改善肩帶穩定與姿勢控制。', 
    icon: '🏋️' 
  },
  { 
    id: 'lateral_raise', 
    name: 'Lateral Raise', 
    description: '針對三角肌外側的輕量訓練，協助長者提升抬臂與側向拿取物品的能力。', 
    icon: '💪' 
  },
  { 
    id: 'chest_pull', 
    name: 'Chest Pull', 
    description: '強化上背與肩後肌群的穩健訓練，協助長者改善肩胛後收與挺胸姿勢', 
    icon: '💪' 
  },
  { 
    id: 'front_raise', 
    name: 'Front Raise', 
    description: '強化前三角肌與肩前穩定度，幫助長者安全抬手至胸前/眼前高度', 
    icon: '💪' 
  },
  { 
    id: 'overhead_press', 
    name: 'Overhead Press', 
    description: '逐步訓練肩部與上背推舉能力，協助長者改善頭上取物與伸手動作。', 
    icon: '🏋️' 
  }
]);

function startExercise() {
  if (!selectedExercise.value) return;

  router.push({ 
    name: 'exercise', 
    params: { type: selectedExercise.value.id },
    query: { style: selectedLevel.value } // 使用全局选择的难度
  });

  showModal.value = false;
}
</script>

<style scoped>
.home-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background-color: #f0f2f5;
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
}

.hero-section p {
  font-size: 20px;
  opacity: 0.9;
}

.exercise-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
  margin-bottom: 60px;
}

.exercise-card {
  background: white;
  border-radius: 16px;
  padding: 25px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  cursor: pointer;
}

.exercise-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.card-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.exercise-card h3 {
  font-size: 24px;
  margin-bottom: 8px;
  color: #333;
}

.exercise-card p {
  color: #666;
  margin-bottom: 0;
}

.card-arrow {
  position: absolute;
  right: 30px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 24px;
  color: #667eea;
  transition: transform 0.3s ease;
}

.exercise-card:hover .card-arrow {
  transform: translateY(-50%) translateX(5px);
}

.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 30px;
  text-align: center;
}

.feature {
  padding: 20px;
}

.feature-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 16px;
}

.feature h4 {
  font-size: 20px;
  margin-bottom: 8px;
  color: #333;
}

.feature p {
  color: #666;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  padding: 30px;
  width: 90%;
  max-width: 500px;
  text-align: center;
}

.modal-content h2 { margin-top: 0; margin-bottom: 25px; }

.level-options { display: flex; gap: 20px; margin-bottom: 30px; }
.level-options input[type="radio"] { display: none; }
.level-options label { flex: 1; cursor: pointer; }
.level-card { padding: 20px; border: 2px solid #e0e0e0; border-radius: 12px; transition: all 0.2s ease; text-align: left; }
.level-card .emoji { font-size: 1.5em; margin-right: 10px; }
.level-card div { font-weight: bold; font-size: 1.1em; }
.level-card p { font-size: 0.9em; color: #6c757d; margin: 5px 0 0; }
.level-options input[type="radio"]:checked + .level-card {
  border-color: #6a5af9;
  background-color: #f3f1ff;
  box-shadow: 0 0 10px rgba(106, 90, 249, 0.2);
}

.modal-buttons {
  display: flex;
  gap: 15px; /* 按钮之间的间距 */
}

.confirm-btn {
  flex: 1; 
  padding: 15px;
  font-size: 1.1em;
  font-weight: bold;
  color: white;
  background: #6a5af9;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: background-color 0.2s;
}
.confirm-btn:hover { background: #5a4ae9; }

.cancel-btn {
  flex: 1; 
  padding: 15px;
  font-size: 1.1em;
  font-weight: bold;
  color: white; 
  background-color: #e74c3c; /* 红底 */
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: background-color 0.2s;
}
.cancel-btn:hover {
  background-color: #c0392b; /* 悬停时更深的红色 */
}


#need-help-fab {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 60px;
  height: 60px;
  background: #6a5af9;
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
  transition: transform 0.2s ease;
}
#need-help-fab:hover {
  transform: scale(1.1);
  background: #5a4ae9;
}

@media (max-width: 768px) {
  .level-options { flex-direction: column; }
}
</style>