from flask import Blueprint, request, jsonify, current_app
import jwt
from datetime import datetime, timedelta

# 1. 创建一个名为 'auth' 的蓝图
auth_bp = Blueprint('auth_bp', __name__)

# 2. 在这个蓝图上定义 /login 路由
@auth_bp.route('/login', methods=['POST'])
def login():
    # 获取前端发来的PIN码
    data = request.get_json()
    if not data or 'pin' not in data:
        return jsonify({'message': 'PIN is missing'}), 400

    correct_pin = "1234" # 暂时硬编码
    secret_key = "this-secret-key"
    pin_attempt = data.get('pin')
    # 3. 验证PIN码
    if pin_attempt == correct_pin:
        # 4. PIN正确，生成JWT
        try:
            payload = {
                'exp': datetime.utcnow() + timedelta(hours=1), # 过期时间
                'iat': datetime.utcnow(), # 签发时间
                'sub': 'user'  # 主题，用一个唯一标识符
            }
            # 使用 current_app 从配置中安全地获取 SECRET_KEY 来签名
            token = jwt.encode(
                payload,
                secret_key,
                algorithm='HS256'
            )
            # 5. 将生成的Token返回给前端
            return jsonify({'token': token}), 200
        except Exception as e:
            return jsonify({'message': str(e)}), 500
    else:
        # PIN错误
        return jsonify({'message': 'Invalid credentials'}), 401