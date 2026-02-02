import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || null)
  const userId = ref(localStorage.getItem('userId') || null)
  const userName = ref(localStorage.getItem('userName') || null)

  const isLoggedIn = computed(() => !!token.value)
  
  const user = computed(() => {
    if (!userId.value) return null
    return {
      id: userId.value,
      name: userName.value
    }
  })

  async function login(pin) {
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ pin })
      })

      const data = await response.json()

      if (response.ok) {
        token.value = data.token
        userId.value = data.user_id
        userName.value = data.user_name

        localStorage.setItem('token', data.token)
        localStorage.setItem('userId', data.user_id)
        localStorage.setItem('userName', data.user_name)

        console.log('[AuthStore] ✅ 登入成功:', userName.value)
        return { success: true }
      } else {
        console.error('[AuthStore] ❌ 登入失敗:', data.message)
        return { success: false, message: data.message }
      }
    } catch (error) {
      console.error('[AuthStore] ❌ 網絡錯誤:', error)
      return { success: false, message: '網絡連接失敗' }
    }
  }

  function logout() {
    token.value = null
    userId.value = null
    userName.value = null

    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('userName')

    console.log('[AuthStore] 已登出')
  }

  function checkTokenExpiry() {
    if (!token.value) return false

    try {
      const payload = JSON.parse(atob(token.value.split('.')[1]))
      const exp = payload.exp * 1000
      
      if (Date.now() >= exp) {
        console.log('[AuthStore] Token 已過期，自動登出')
        logout()
        return false
      }
      return true
    } catch (e) {
      console.error('[AuthStore] Token 解析失敗:', e)
      logout()
      return false
    }
  }

  return {
    token,
    userId,
    userName,
    
    isLoggedIn,
    user,
    
    login,
    logout,
    checkTokenExpiry
  }
})