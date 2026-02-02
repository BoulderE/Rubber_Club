from flask import Blueprint, request, jsonify
from models.db_models import get_session, ExerciseRecord, User
from datetime import datetime

records_bp = Blueprint('records_bp', __name__)

# 保存運動記錄
@records_bp.route('/records', methods=['POST'])
def save_record():
    data = request.get_json()
    
    required_fields = ['user_id', 'exercise_name']
    for field in required_fields:
        if field not in data:
            return jsonify({'message': f'{field} is required'}), 400
    
    session = get_session()
    
    # 驗證用戶存在
    user = session.query(User).filter_by(id=data['user_id']).first()
    if not user:
        session.close()
        return jsonify({'message': 'User not found'}), 404
    
    # 創建記錄
    record = ExerciseRecord(
        user_id=data['user_id'],
        exercise_name=data['exercise_name'],
        accuracy=data.get('accuracy'),
        smoothness=data.get('smoothness'),
        duration=data.get('duration'),
        rep_count=data.get('rep_count')
    )
    
    session.add(record)
    session.commit()
    
    record_id = record.id
    session.close()
    
    return jsonify({
        'message': 'Record saved successfully',
        'record_id': record_id
    }), 201


# 查詢用戶的運動記錄
@records_bp.route('/records/<int:user_id>', methods=['GET'])
def get_records(user_id):
    session = get_session()
    
    # 驗證用戶存在
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        session.close()
        return jsonify({'message': 'User not found'}), 404
    
    # 查詢記錄（最新的在前）
    records = session.query(ExerciseRecord)\
        .filter_by(user_id=user_id)\
        .order_by(ExerciseRecord.created_at.desc())\
        .limit(50)\
        .all()
    
    result = []
    for r in records:
        result.append({
            'id': r.id,
            'exercise_name': r.exercise_name,
            'accuracy': r.accuracy,
            'smoothness': r.smoothness,
            'duration': r.duration,
            'rep_count': r.rep_count,
            'created_at': r.created_at.isoformat() if r.created_at else None
        })
    
    session.close()
    
    return jsonify({
        'user_id': user_id,
        'user_name': user.name,
        'total_records': len(result),
        'records': result
    }), 200


# 獲取用戶運動統計
@records_bp.route('/records/<int:user_id>/stats', methods=['GET'])
def get_stats(user_id):
    session = get_session()
    
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        session.close()
        return jsonify({'message': 'User not found'}), 404
    
    records = session.query(ExerciseRecord).filter_by(user_id=user_id).all()
    
    if not records:
        session.close()
        return jsonify({
            'user_id': user_id,
            'user_name': user.name,
            'total_workouts': 0,
            'total_reps': 0,
            'avg_smoothness': 0,
            'by_exercise': {}
        }), 200
    
    # 計算統計
    total_reps = sum(r.rep_count or 0 for r in records)
    smoothness_values = [r.smoothness for r in records if r.smoothness]
    avg_smoothness = sum(smoothness_values) / len(smoothness_values) if smoothness_values else 0
    
    # 按運動類型分組
    by_exercise = {}
    for r in records:
        name = r.exercise_name
        if name not in by_exercise:
            by_exercise[name] = {'count': 0, 'total_reps': 0}
        by_exercise[name]['count'] += 1
        by_exercise[name]['total_reps'] += r.rep_count or 0
    
    session.close()
    
    return jsonify({
        'user_id': user_id,
        'user_name': user.name,
        'total_workouts': len(records),
        'total_reps': total_reps,
        'avg_smoothness': round(avg_smoothness, 1),
        'by_exercise': by_exercise
    }), 200