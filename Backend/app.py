from flask import Flask
from flask_cors import CORS
import mediapipe as mp

from .api.routes import register_routes

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 启用跨域资源共享
    
    # 注册API路由
    register_routes(app, mp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)