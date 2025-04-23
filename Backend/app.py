from flask import Flask, jsonify
from flask_cors import CORS
from api.routes import mediapipe_bp

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # 启用跨域资源共享
    
    # 注册API路由
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')

    @app.route('/')
    def index():
        return jsonify({
            'name': 'MediaPipe 动作分析 API',
            'version': '1.0',
            'endpoints': [
                '/mediapipe/analyze-stream',
                '/mediapipe/control',
                '/mediapipe/status'
            ]
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)