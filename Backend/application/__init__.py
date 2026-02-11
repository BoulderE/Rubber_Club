from flask import Flask
from flask_cors import CORS
from application.config import Config
from .blueprints.mediapipe_bp import mediapipe_bp
from .blueprints.auth_bp import auth_bp
from .blueprints.admin_bp import admin_bp
from .blueprints.task_bp import task_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 啟用 CORS
    CORS(app, resources={
        r"/api/*": {"origins": "*"},
        r"/mediapipe/*": {"origins": "*"}
    })
    
    # 註冊藍圖
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(task_bp, url_prefix='/api/tasks')
    
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """註冊全局錯誤處理器，返回JSON格式錯誤"""
    @app.errorhandler(400)
    def bad_request(e):
        return {'error': 'Bad request', 'message': str(e)}, 400
    
    @app.errorhandler(401)
    def unauthorized(e):
        return {'error': 'Unauthorized', 'message': '需要認證'}, 401
    
    @app.errorhandler(403)
    def forbidden(e):
        return {'error': 'Forbidden', 'message': '權限不足'}, 403
    
    @app.errorhandler(404)
    def not_found(e):
        return {'error': 'Resource not found'}, 404
        
    @app.errorhandler(500)
    def server_error(e):
        return {'error': 'Internal server error'}, 500