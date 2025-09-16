<template>
  <div class="home-view">
    <div class="hero-section">
      <h1>AI 健身教练</h1>
      <p>使用 MediaPipe 技术，实时分析您的运动姿势</p>
    </div>

    <button @click="toggleChatbot" class="chatbot-toggle-btn">
      需要帮助？
    </button>

    <!-- Chatbot 窗口 -->
    <ChatbotWindow 
      v-if="isChatbotVisible" 
      @close="isChatbotVisible = false"
      class="chatbot-container"
    />
    
    <div class="exercise-cards">
      <router-link 
        v-for="exercise in exercises" 
        :key="exercise.id"
        :to="exercise.route"
        class="exercise-card"
      >
        <div class="card-icon">{{ exercise.icon }}</div>
        <h3>{{ exercise.name }}</h3>
        <p>{{ exercise.description }}</p>
        <span class="card-arrow">→</span>
      </router-link>
    </div>
    
    <div class="features">
      <div class="feature">
        <span class="feature-icon">📹</span>
        <h4>实时分析</h4>
        <p>通过摄像头实时捕捉动作</p>
      </div>
      <div class="feature">
        <span class="feature-icon">🎯</span>
        <h4>精准反馈</h4>
        <p>AI 分析动作准确度</p>
      </div>
      <div class="feature">
        <span class="feature-icon">📊</span>
        <h4>数据统计</h4>
        <p>记录运动数据和进度</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useExerciseStore } from '@/stores/exercise'
import ChatbotWindow from '@/components/ChatbotWindow.vue'; 
import { ref, computed } from 'vue';

const isChatbotVisible = ref(false);
const toggleChatbot = () => {
  isChatbotVisible.value = !isChatbotVisible.value;
};

const exerciseStore = useExerciseStore()

const exercises = computed(() => 
  exerciseStore.exerciseTypes.map(ex => ({
    ...ex,
    route: `/${ex.id.replace('_', '-')}`
  }))
)
</script>

<style scoped>
.home-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.hero-section {
  text-align: center;
  padding: 60px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 20px;
  margin-bottom: 40px;
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
  padding: 30px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
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
</style>