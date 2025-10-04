<template>
  <div class="chatbot-window">
    <div class="chat-header">
      <span>
        AI Fitness Assistant
      </span>
      <button @click="emit('close')" class="close-btn">×</button>
    </div>
    <div class="chat-body" ref="chatBody">
      <div v-for="message in messages" :key="message.id" class="message-container">
        <div :class="['message', message.role]">
          {{ message.content }}
        </div>
      </div>
    </div>
    <div class="input-container">
      <form @submit.prevent="sendMessage">
        <input
          type="text"
          v-model="newMessage"
          placeholder="Message"
          :disabled="isLoading"
        />
        <button type="submit" :disabled="isLoading">
          Send
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue';
import { getApiBase } from '@/api/base';
const emit = defineEmits(['close']);

const messages = ref([]);
const newMessage = ref('');
const isLoading = ref(false);
const chatBody = ref(null); 
const base = getApiBase();

const scrollToBottom = () => {
  nextTick(() => {
    if (chatBody.value) {
      chatBody.value.scrollTop = chatBody.value.scrollHeight;
    }
  });
};

const sendMessage = async () => {
  const content = newMessage.value.trim();
  if (!content || isLoading.value) return;

  // 1. 将用户消息添加到界面
  messages.value.push({ id: Date.now(), role: 'user', content });
  const currentMessageHistory = messages.value.map(({ role, content }) => ({ role, content }));
  newMessage.value = '';
  scrollToBottom(); 

  isLoading.value = true;

  try {
    // 2. 发送请求到后端
    const response = await fetch(`${base}/api/chatbot/chat`, {
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
    content: "Hello! I'm your smart fitness assistant. I can help you choose the coaching style that best suits you. What do you value most in your fitness experience?"
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

.chat-body {
  flex-grow: 1;
  padding: 15px;
  overflow-y: auto; 
  background-color: #e5ddd5;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.chat-body::-webkit-scrollbar {
  width: 6px;
}
.chat-body::-webkit-scrollbar-thumb {
  background: #bdbdbd;
  border-radius: 3px;
}
.chat-body::-webkit-scrollbar-thumb:hover {
  background: #a5a5a5;
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

.message.assistant {
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