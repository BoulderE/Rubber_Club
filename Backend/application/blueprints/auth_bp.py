from flask import Blueprint, request, jsonify
import jwt
from datetime import datetime, timedelta
from models import get_session, User

auth_bp = Blueprint('auth_bp', __name__)

SECRET_KEY = "this-secret-key"  # 之後改成從環境變量讀取

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'pin' not in data:
        return jsonify({'message': 'PIN is missing'}), 400

    pin_attempt = data.get('pin')
    
    # 從數據庫查詢用戶
    session = get_session()
    user = session.query(User).filter_by(pin=pin_attempt).first()
    session.close()
    
    if user:
        # PIN 正確，生成 JWT
        try:
            payload = {
                'exp': datetime.utcnow() + timedelta(hours=1),
                'iat': datetime.utcnow(),
                'sub': user.id,  # 使用用戶 ID
                'name': user.name
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
            return jsonify({
                'token': token,
                'user_id': user.id,
                'user_name': user.name
            }), 200
        except Exception as e:
            return jsonify({'message': str(e)}), 500
    else:
        return jsonify({'message': 'Invalid credentials'}), 401