import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'
console.log('API_BASE:', API_BASE)
console.log('All env:', import.meta.env)

function getAuthHeaders() {
  const userPin = localStorage.getItem('userPin')
  return {
    'X-User-Pin': userPin
  }
}

export async function fetchMyTasks() {
  const response = await axios.get(`${API_BASE}/api/tasks/my-tasks`, {
    headers: getAuthHeaders()
  })
  return response.data
}

export async function startTaskApi(taskId) {
  const response = await axios.post(
    `${API_BASE}/api/tasks/my-tasks/${taskId}/start`,
    {},
    { headers: getAuthHeaders() }
  )
  return response.data
}

export async function updateTaskProgress(taskId, data) {
  const response = await axios.post(
    `${API_BASE}/api/tasks/my-tasks/${taskId}/progress`,
    data,
    { headers: getAuthHeaders() }
  )
  return response.data
}