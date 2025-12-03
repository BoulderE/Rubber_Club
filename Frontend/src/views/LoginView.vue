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
import { ref, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { getApiBase } from '@/api/base';

const currentPin = ref('');
const message = ref('');
const messageColor = ref('#e74c3c');
const isShaking = ref(false);
const isLoggedIn = ref(false);
const isSubmitting = ref(false);

const router = useRouter();
const base = getApiBase();

const keypadLayout = [1, 2, 3, 4, 5, 6, 7, 8, 9, null, 0, 'delete'];

const handleKeyPress = (key) => {
  if (isLoggedIn.value || isSubmitting.value) return;
  if (key === null) return;

  message.value = '';

  if (key === 'delete') {
    currentPin.value = currentPin.value.slice(0, -1);
  } else if (currentPin.value.length < 4) {
    currentPin.value += String(key);
  }

  if (currentPin.value.length === 4) {
    checkPin();
  }
};

const checkPin = async () => {
  if (isSubmitting.value) return;
  isSubmitting.value = true;

  try {
    const response = await axios.post(`${base}/api/login`, {
      pin: currentPin.value
    }, { timeout: 8000 });

    const token = response.data?.token || response.data?.access_token || response.data?.data?.token;
    if (token) {
      await loginSuccess(token);
    } else {
      throw new Error('Invalid response shape');
    }
  } catch (error) {
    console.error('Login request failed:', error);
    if (error?.response?.status === 401) {
      loginFailure('PIN碼錯誤，請重試');
    } else {
      loginFailure('登入失敗，請稍後再試');
    }
  } finally {
    isSubmitting.value = false;
  }
};

const loginSuccess = async (token) => {
  message.value = '登入成功！';
  messageColor.value = '#2ecc71';
  isLoggedIn.value = true;

  try {
    localStorage.setItem('user-token', token);
  } catch (e) {
    console.warn('localStorage 寫入失敗：', e);
  }

  await nextTick();
  setTimeout(() => {
    router.push('/home');
  }, 300);
};

const loginFailure = (msg = 'PIN碼錯誤，請重試') => {
  message.value = msg;
  messageColor.value = '#e74c3c';
  isShaking.value = true;

  setTimeout(() => {
    isShaking.value = false;
    currentPin.value = '';
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