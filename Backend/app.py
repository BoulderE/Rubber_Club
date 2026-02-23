from flask import Flask, jsonify
from flask_cors import CORS
from api.routes import mediapipe_bp
from application.blueprints.auth_bp import auth_bp
from application.blueprints.chatbot_bp import chatbot_bp
from application.blueprints.records_bp import records_bp
from application.blueprints.task_bp import task_bp
from application.blueprints.admin_bp import admin_bp

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    app.register_blueprint(mediapipe_bp, url_prefix='/mediapipe')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
    app.register_blueprint(records_bp, url_prefix='/api')
    app.register_blueprint(task_bp, url_prefix='/api/tasks')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    @app.route('/')
    def index():
        return jsonify({
            'name': 'MediaPipe 动作分析 API',
            'version': '1.0',
            'endpoints': {
            'mediapipe': [
                '/mediapipe/analyze-stream',
                '/mediapipe/control',
                '/mediapipe/status',
            ],
            'auth': [
                '/api/login',
            ],
            'chatbot': [
                '/api/chatbot/chat',
            ],
            'records': [
                '/api/records',
            ],
            'tasks': [
                '/api/my-tasks',
                '/api/my-tasks/completed',
                '/api/my-tasks/active',
                '/api/my-tasks/<id>/start',
                '/api/my-tasks/<id>/progress',
            ],
            'admin': [
                '/api/admin/login',
                '/api/admin/dashboard',
                '/api/admin/users',
                '/api/admin/users/<id>',
                '/api/admin/users/<id>/history',
                '/api/admin/users/<id>/summary',
                '/api/admin/exercises',
                '/api/admin/assign',
                '/api/admin/assignments',
                '/api/admin/assignments/<id>',
            ],
            'health': [
                '/health',
            ]
        }  
    })
    
    @app.get('/health')
    def health():
        return jsonify(ok=True), 200
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)