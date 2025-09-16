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
  const requiresAuth = to.meta.requiresAuth;
  const hasToken = !!localStorage.getItem('user-token');

  if (requiresAuth) {
    if (hasToken) {
      next();
    } else {
      next({ name: 'login' });
    }
  } else {
    if (to.name === 'login' && hasToken) {
      next({ name: 'home' });
    } else {
      next();
    }
  }
});
export default router