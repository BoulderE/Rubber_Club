import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ExerciseView from '../views/ExerciseView.vue'

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
      path: '/exercise/:type',
      name: 'exercise',
      component: ExerciseView,
      props: true, 
      meta: { requiresAuth: true }
    },
    {
      path: '/history',
      name: 'History',
      component: () => import('../views/HistoryView.vue')
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