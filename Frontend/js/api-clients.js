class ApiClient {
    constructor(baseUrl = 'http://localhost:5001') {
      this.baseUrl = baseUrl;
    }
  
    /**
     * 分析单张图片
     */
    async analyzeImage(imageFile) {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        body: formData
      });
      
      return response.json();
    }
  
    /**
     * 分析视频流帧
     */
    async analyzeStreamFrame(imageBlob) {
      const formData = new FormData();
      formData.append('file', imageBlob, 'frame.jpg');
      
      const response = await fetch(`${this.baseUrl}/analyze-stream`, {
        method: 'POST',
        body: formData
      });
      
      return response.json();
    }
  
    /**
     * 获取当前状态
     */
    async getStatus() {
      const response = await fetch(`${this.baseUrl}/status`);
      return response.json();
    }
  
    /**
     * 控制训练状态
     */
    async controlWorkout(action) {
      const response = await fetch(`${this.baseUrl}/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action })
      });
      
      return response.json();
    }
  }
  
  // 创建一个全局API客户端实例
  const apiClient = new ApiClient();