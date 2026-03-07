import { defineStore } from 'pinia'
import * as adminApi from '@/api/admin'
import { getApiBase } from '@/api/base'

export const useAdminStore = defineStore('admin', {
  state: () => ({
    token: localStorage.getItem('adminToken') || null,
    stats: null,
    users: [],
    assignments: []
  }),

  getters: {
    isAuthenticated: (state) => !!state.token
  },

  actions: {
    async login(pin) {
      const res = await adminApi.adminLogin(pin)
      if (res.admin) {
        this.token = res.admin.pin
        localStorage.setItem('adminToken', res.admin.pin)
      }
      return res
    },

    logout() {
      this.token = null
      localStorage.removeItem('adminToken')
    },

    async fetchStats() {
      const res = await adminApi.getStats(this.token)
      if (!res.error) this.stats = res
      return res
    },

    async fetchUsers() {
      const res = await adminApi.getUsers(this.token)
      if (res.users) this.users = res.users
      return res
    },

    async fetchUserHistory(userId) {
      return await adminApi.getUserHistory(this.token, userId)
    },

    async fetchExercises() {
      const API_BASE = getApiBase()
      const res = await fetch(`${API_BASE}/api/admin/exercises`, {
        headers: { 'X-Admin-Pin': this.token }
      })
      if (!res.ok) return []
      return await res.json()
    },

    async assignExercise(data) {
      return await adminApi.assignExercise(this.token, data)
    },

    async fetchAssignments(userId = null) {
      const res = await adminApi.getAssignments(this.token, userId)
      if (res.assignments) this.assignments = res.assignments
      return res
    },

    async updateAssignment(id, data) {
      return await adminApi.updateAssignment(this.token, id, data)
    },

    async deleteAssignment(id) {
      return await adminApi.deleteAssignment(this.token, id)
    },

    async createUser(data) {
      const res = await adminApi.createUser(this.token, data)
      return res
    },

    async updateUser(userId, data) {
      const res = await adminApi.updateUser(this.token, userId, data)
      return res
    },

    async deleteUser(userId) {
      const res = await adminApi.deleteUser(this.token, userId)
      return res
    }
  }
})