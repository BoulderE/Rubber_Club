<template>
  <div class="chatbot-window">
    <div class="chat-header">
      <span>智能助手</span>
      <button @click="emit('close')" class="close-btn">×</button>
    </div>
    <div class="chat-body" ref="chatBody">
      <div v-for="message in messages" :key="message.id" class="message-container">
        <div :class="['message', message.role]">
          {{ message.content }}
        </div>
      </div>
    </div>
    <div class="chat-footer">
      <form @submit.prevent="sendMessage">
        <input
          type="text"
          v-model="newMessage"
          placeholder="输入消息..."
          class="message-input"
          :disabled="isLoading"
        />
        <button type="submit" class="send-btn" :disabled="isLoading">
          发送
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue';

// 定义组件可以发出的事件
const emit = defineEmits(['close']);

// --- 响应式变量 ---
const messages = ref([]);
const newMessage = ref('');
const isLoading = ref(false);
const chatBody = ref(null); // 关键2：创建一个 ref 来引用 DOM 元素

// --- 核心函数 ---

/**
 * 滚动聊天窗口到底部
 * 使用 nextTick 确保在 DOM 更新后再执行滚动
 */
const scrollToBottom = () => {
  nextTick(() => {
    if (chatBody.value) {
      chatBody.value.scrollTop = chatBody.value.scrollHeight;
    }
  });
};

/**
 * 发送消息的异步函数
 */
const sendMessage = async () => {
  const content = newMessage.value.trim();
  if (!content || isLoading.value) return;

  // 1. 将用户消息添加到界面
  messages.value.push({ id: Date.now(), role: 'user', content });
  const currentMessageHistory = messages.value.map(({ role, content }) => ({ role, content }));
  newMessage.value = '';
  scrollToBottom(); // 关键3：直接调用函数，而不是 this.scrollToBottom()

  isLoading.value = true;

  try {
    // 2. 发送请求到后端
    const response = await fetch('http://127.0.0.1:5001/api/chatbot/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: content,
        history: currentMessageHistory.slice(0, -1), // 发送除当前用户消息外的所有历史
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // 3. 将机器人回复添加到界面
    messages.value.push({ id: Date.now() + 1, role: 'assistant', content: data.reply });
    scrollToBottom(); // 关键4：再次直接调用函数

  } catch (error) {
    console.error("Error sending message:", error);
    messages.value.push({
      id: Date.now() + 1,
      role: 'assistant',
      content: '抱歉，我好像遇到了一点网络问题，请稍后再试。'
    });
    scrollToBottom();
  } finally {
    isLoading.value = false;
  }
};

// --- 生命周期钩子 ---

// 组件加载时，添加欢迎语
onMounted(() => {
  messages.value.push({
    id: Date.now(),
    role: 'assistant',
    content: '您好！我是您的智能健身助手。我可以帮助您选择最适合您的教练风格。您在健身体验中，最看重的是什么呢？'
  });
  scrollToBottom();
});
</script>

<style scoped>
.chatbot-window {
  width: 350px;
  height: 500px;
  background-color: #f9f9f9;
  border-radius: 10px;
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-family: sans-serif;
}

.chatbot-header {
  background-color: #4a90e2;
  color: white;
  padding: 10px 15px;
  font-weight: bold;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
}

.messages-container {
  flex-grow: 1;
  padding: 15px;
  overflow-y: auto;
  background-color: #e5ddd5;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message {
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 18px;
  word-wrap: break-word;
}

.message.user {
  background-color: #dcf8c6;
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}

.message.bot {
  background-color: #ffffff;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.input-container {
  padding: 10px;
  background-color: #f0f0f0;
}

.input-container form {
  display: flex;
}

.input-container input {
  flex-grow: 1;
  border: 1px solid #ccc;
  padding: 10px;
  border-radius: 20px;
  margin-right: 10px;
}

.input-container button {
  padding: 10px 15px;
  border: none;
  background-color: #4a90e2;
  color: white;
  border-radius: 20px;
  cursor: pointer;
}
</style>