<template>
  <div id="app">
    <nav class="navbar">
      <router-link to="/home" class="logo">
        Rubber Club
      </router-link>
      <div class="nav-links" v-if="!isLoginPage">
        <router-link to="/exercise/lateral_raise">Lateral Raise</router-link>
        <router-link to="/exercise/chest_pull">Chest Pull</router-link>
        <router-link to="/exercise/squat">Bicep Curl</router-link>
        <router-link to="/exercise/front_raise">Front Raise</router-link>
        <router-link to="/exercise/overhead_press">Overhead Press</router-link>
      </div>
      <div v-if="isLoggedIn" class="user-actions">
        <button @click="logout" class="logout-btn">Log out</button>
      </div>
    </nav>
  
    
    <main>
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';

const route = useRoute();
const isLoginPage = computed(() => route.name === 'Login');
const router = useRouter();
const isLoggedIn = ref(!!localStorage.getItem('user-token'));

const logout = () => {
  localStorage.removeItem('user-token');
  isLoggedIn.value = false;
  router.replace({ name: 'login' });
};

watch(
  () => route.path,
  () => {
    isLoggedIn.value = !!localStorage.getItem('user-token');
  }
);
</script>

<style>
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --text-primary: #333;
  --text-secondary: #666;
  --background: #f8f9fa;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background-color: var(--background);
  color: var(--text-primary);
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  background: white;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: 15px 30px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  font-size: 24px;
  font-weight: bold;
  text-decoration: none;
  color: var(--primary-color);
}

.nav-links {
  display: flex;
  gap: 30px;
}

.nav-links a {
  text-decoration: none;
  color: var(--text-secondary);
  font-weight: 500;
  transition: color 0.3s;
}

.nav-links a:hover,
.nav-links a.router-link-active {
  color: var(--primary-color);
}

main {
  flex: 1;
  padding: 20px;
}
</style>