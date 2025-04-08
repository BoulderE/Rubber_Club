from flask import Flask
from .blueprints.mediapipe_bp import mediapipe_bp

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../config.py')
    
    # 注册蓝图
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')
    register_error_handlers(app)
    return app

def register_error_handlers(app):
    """注册全局错误处理器，返回JSON格式错误"""
    @app.errorhandler(404)
    def not_found(e):
        return {'error': 'Resource not found'}, 404
        
    @app.errorhandler(500)
    def server_error(e):
        return {'error': 'Internal server error'}, 500