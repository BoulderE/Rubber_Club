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
    },
    // Admin routes
    {
      path: '/admin/login',
      name: 'adminLogin',
      component: () => import('../views/AdminLogin.vue')
    },
    {
      path: '/admin/dashboard',
      name: 'adminDashboard',
      component: () => import('../views/AdminDashboard.vue'),
      meta: { requiresAdmin: true }
    },
    {
      path: '/admin/users',
      name: 'adminUsers',
      component: () => import('../views/AdminUsers.vue'),
      meta: { requiresAdmin: true }
    },
    {
      path: '/admin/assignments',
      name: 'adminAssignments',
      component: () => import('../views/AdminAssignments.vue'),
      meta: { requiresAdmin: true }
    }
  ]
})

router.beforeEach((to, from, next) => {
  const requiresAuth = to.meta.requiresAuth
  const requiresAdmin = to.meta.requiresAdmin
  const hasToken = !!localStorage.getItem('token')
  const hasAdminToken = !!localStorage.getItem('adminToken')

  if (requiresAdmin) {
    if (hasAdminToken) {
      next()
    } else {
      next({ name: 'adminLogin' })
    }
  } else if (requiresAuth) {
    if (hasToken) {
      next()
    } else {
      next({ name: 'login' })
    }
  } else {
    if (to.name === 'login' && hasToken) {
      next({ name: 'home' })
    } else {
      next()
    }
  }
})

export default router