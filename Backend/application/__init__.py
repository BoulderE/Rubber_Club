from flask import Flask
from application.config import Config
from .blueprints.mediapipe_bp import mediapipe_bp
from .blueprints.auth_bp import auth_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    # 注册蓝图
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')
    app.register_blueprint(auth_bp, url_prefix='/api')
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