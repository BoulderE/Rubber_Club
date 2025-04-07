from flask import Flask
from .blueprints.mediapipe_bp import mediapipe_bp

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../config.py')
    
    # 注册蓝图
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')
    
    return app