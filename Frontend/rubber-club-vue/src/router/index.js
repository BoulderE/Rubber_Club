import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/login'
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginView.vue')
    },
    {
      path: '/home',
      name: 'home',
      component: HomeView,
      meta: { requiresAuth: true }
    },
    {
      path: '/lateral-raise',
      name: 'lateral-raise',
      component: () => import('../views/LateralRaiseView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/chest-pull',
      name: 'chest-pull',
      component: () => import('../views/ChestPullView.vue'),
      meta: { requiresAuth: true }
    }
  ]
})

router.beforeEach((to, from, next) => {
  // 检查即将进入的路由是否需要认证
  if (to.meta.requiresAuth) {
    const isLoggedIn = !!localStorage.getItem('user-token'); // 这是一个简单的示例

    if (isLoggedIn) {
      next(); 
    } else {
      next('/login'); 
    }
  } else {
    next(); 
  }
});

export default router