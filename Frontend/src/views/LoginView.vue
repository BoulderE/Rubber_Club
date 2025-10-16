<template>
  <div class="login-page">
    <div class="login-container" v-if="!isLoggedIn">
      <h1>Welcome Back</h1>
      <p>Please Enter Your 4-digit PIN</p>

      <div class="pin-display" :class="{ shake: isShaking }">
        <div v-for="i in 4" :key="i" class="pin-dot" :class="{ filled: currentPin.length >= i }"></div>
      </div>
      
      <div class="message" :style="{ color: messageColor }">{{ message }}</div>

      <div class="keypad">
        <button v-for="key in keypadLayout" :key="key" 
                :class="{ 'action-key': typeof key !== 'number' }"
                @click="handleKeyPress(key)">
          {{ key === 'delete' ? '⌫' : key }}
        </button>
      </div>
    </div>

    <div class="welcome-screen" v-else>
      <h1>Logged In!</h1>
      <p>Loading your profile...</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { getApiBase } from '@/api/base';

// --- 1. 响应式状态定义 ---
const currentPin = ref(''); // 使用 ref 创建响应式变量
const message = ref('');
const messageColor = ref('#e74c3c'); // 默认是错误颜色
const isShaking = ref(false);
const router = useRouter(); // 获取router实例
const isLoggedIn = ref(false); // 登录状态
const base = getApiBase();
// --- 2. 逻辑与计算属性 ---

const keypadLayout = [1, 2, 3, 4, 5, 6, 7, 8, 9, null, 0, 'delete'];

// --- 3. 方法定义 ---
const handleKeyPress = (key) => {
  if (isLoggedIn.value) return; // 已登录则不响应按键
  if (key === null) return; // 点击了空白格

  message.value = ''; // 每次按键都清除消息

  if (key === 'delete') {
    currentPin.value = currentPin.value.slice(0, -1);
  } else if (currentPin.value.length < 4) {
    currentPin.value += key;
  }

  // 当输入满4位时，自动触发登录检查
  if (currentPin.value.length === 4) {
    checkPin();
  }
};

const checkPin = async () => {
  try {
    const response = await axios.post(`${base}/api/login`, {
      pin: currentPin.value
    });

    if (response.data?.token) {
      loginSuccess(response.data.token);
    } else {
      throw new Error('Invalid response');
    }
  } catch (error) {
    console.error('Login request failed:', error);
    loginFailure();
  }
};

const loginSuccess = (token) => {
  message.value = '登入成功！';
  messageColor.value = '#2ecc71'; // 成功消息颜色
  isLoggedIn.value = true; 
  
  localStorage.setItem('user-token', token)
  setTimeout(() => {
    router.push('/home'); 
  }, 50);
};

const loginFailure = () => {
  message.value = 'PIN碼錯誤，請重試';
  messageColor.value = '#e74c3c';
  isShaking.value = true; // 触发抖动动画

  setTimeout(() => {
    isShaking.value = false;
    currentPin.value = ""; // 自动清空
  }, 600);
};
</script>

<style scoped>
.login-page {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background-color: #f0f2f5;
}

.login-container {
    width: 100%;
    max-width: 360px;
    padding: 40px 30px;
    background-color: #ffffff;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    text-align: center;
    box-sizing: border-box;
}

.login-container h1 { margin: 0 0 10px 0; font-size: 28px; color: #333; }
.login-container p { margin: 0 0 30px 0; font-size: 16px; color: #666; }

.pin-display { display: flex; justify-content: center; gap: 15px; margin-bottom: 30px; }
.pin-dot { width: 20px; height: 20px; border: 2px solid #ccc; border-radius: 50%; transition: background-color 0.2s ease; }
.pin-dot.filled { background-color: #3498db; border-color: #3498db; }

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-8px); }
    20%, 40%, 60%, 80% { transform: translateX(8px); }
}
.shake { animation: shake 0.5s ease-in-out; }

.keypad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; justify-items: center; }
.keypad button {
    width: 70px; height: 70px; border-radius: 50%; border: none; background-color: #e9ecf0;
    font-size: 28px; font-weight: 700; color: #333; cursor: pointer;
    transition: background-color 0.2s, transform 0.1s;
    -webkit-tap-highlight-color: transparent;
}
.keypad button:hover { background-color: #d1d8e0; }
.keypad button:active { transform: scale(0.92); }
.keypad .action-key { background-color: transparent !important; cursor: default; }
.keypad button:not(.action-key):hover { background-color: #d1d8e0; }

.message { margin-top: 15px; font-size: 14px; height: 20px; }
.welcome-screen { text-align: center; }
</style>