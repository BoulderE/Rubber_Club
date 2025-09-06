import apiClient from './client'

export const mediapipeAPI = {
  analyzeStream(formData) {
    return apiClient.post('/mediapipe/analyze-stream', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  control(action) {
    return apiClient.post('/mediapipe/control', { action })
  },

  getStatus() {
    return apiClient.get('/mediapipe/status')
  }
}