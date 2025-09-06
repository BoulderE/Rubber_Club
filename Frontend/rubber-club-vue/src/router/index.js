import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/lateral-raise',
      name: 'lateral-raise',
      component: () => import('../views/LateralRaiseView.vue')
    },
    {
      path: '/chest-pull',
      name: 'chest-pull',
      component: () => import('../views/ChestPullView.vue')
    }
  ]
})

export default router