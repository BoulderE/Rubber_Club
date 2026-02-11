from flask import Blueprint, request, jsonify
from models.db_models import get_session, User, AssignedExercise, ExerciseRecord
from datetime import datetime
from sqlalchemy import desc

task_bp = Blueprint('tasks', __name__)

@task_bp.route('/my-tasks', methods=['GET'])
def get_my_tasks():
    """獲取當前用戶的待完成任務"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        tasks = session.query(AssignedExercise).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.status.in_(['pending', 'in_progress'])
        ).order_by(
            desc(AssignedExercise.status == 'in_progress'),
            AssignedExercise.due_date.asc().nullslast()
        ).all()
        
        return jsonify([{
            'id': t.id,
            'exercise_key': t.exercise_key,
            'exercise_name': t.exercise_name,
            'target_reps': t.target_reps,
            'target_sets': t.target_sets,
            'completed_sets': t.completed_sets,
            'completed_reps_total': t.completed_reps_total,
            'status': t.status,
            'difficulty': t.difficulty,
            'due_date': t.due_date.isoformat() if t.due_date else None,
            'admin_notes': t.admin_notes,
            'is_overdue': t.due_date < datetime.now().date() if t.due_date else False
        } for t in tasks])
    finally:
        session.close()


@task_bp.route('/my-tasks/completed', methods=['GET'])
def get_completed_tasks():
    """獲取已完成的任務歷史"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        limit = request.args.get('limit', 20, type=int)
        
        tasks = session.query(AssignedExercise).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.status == 'completed'
        ).order_by(desc(AssignedExercise.completed_at)).limit(limit).all()
        
        return jsonify([{
            'id': t.id,
            'exercise_name': t.exercise_name,
            'target_reps': t.target_reps,
            'target_sets': t.target_sets,
            'completed_reps_total': t.completed_reps_total,
            'avg_smoothness': t.avg_smoothness,
            'completed_at': t.completed_at.isoformat() if t.completed_at else None
        } for t in tasks])
    finally:
        session.close()


@task_bp.route('/my-tasks/active', methods=['GET'])
def get_active_task():
    """獲取當前進行中的任務"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        task = session.query(AssignedExercise).filter_by(
            user_id=user.id,
            status='in_progress'
        ).first()
        
        if not task:
            return jsonify({'active_task': None})
        
        return jsonify({
            'active_task': {
                'id': task.id,
                'exercise_key': task.exercise_key,
                'exercise_name': task.exercise_name,
                'target_reps': task.target_reps,
                'target_sets': task.target_sets,
                'completed_sets': task.completed_sets,
                'remaining_sets': task.target_sets - task.completed_sets,
                'difficulty': task.difficulty,
                'admin_notes': task.admin_notes
            }
        })
    finally:
        session.close()


@task_bp.route('/my-tasks/<int:task_id>/start', methods=['POST'])
def start_task(task_id):
    """開始任務"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        task = session.query(AssignedExercise).filter_by(
            id=task_id,
            user_id=user.id
        ).first()
        
        if not task:
            return jsonify({'error': '任務不存在'}), 404
        
        if task.status == 'completed':
            return jsonify({'error': '任務已完成'}), 400
        
        task.status = 'in_progress'
        session.commit()
        
        return jsonify({
            'message': '任務已開始',
            'task': {
                'id': task.id,
                'exercise_key': task.exercise_key,
                'exercise_name': task.exercise_name,
                'target_reps': task.target_reps,
                'target_sets': task.target_sets,
                'completed_sets': task.completed_sets,
                'difficulty': task.difficulty
            }
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/my-tasks/<int:task_id>/progress', methods=['POST'])
def update_task_progress(task_id):
    """更新任務進度"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        data = request.get_json()
        completed_reps = data.get('completed_reps', 0)
        smoothness = data.get('smoothness', 0)
        duration = data.get('duration', 0)
        
        task = session.query(AssignedExercise).filter_by(
            id=task_id,
            user_id=user.id
        ).first()
        
        if not task:
            return jsonify({'error': '任務不存在'}), 404
        
        if task.status == 'completed':
            return jsonify({'error': '任務已完成'}), 400
        
        old_sets = task.completed_sets
        task.completed_sets += 1
        task.completed_reps_total += completed_reps
        
        if old_sets == 0:
            task.avg_smoothness = smoothness
        else:
            task.avg_smoothness = (
                (task.avg_smoothness * old_sets + smoothness) / task.completed_sets
            )
        
        task.status = 'in_progress'
        
        is_complete = task.completed_sets >= task.target_sets
        if is_complete:
            task.status = 'completed'
            task.completed_at = datetime.utcnow()
        
        record = ExerciseRecord(
            user_id=user.id,
            exercise_name=task.exercise_name,
            rep_count=completed_reps,
            smoothness=smoothness,
            duration=duration
        )
        session.add(record)
        
        session.commit()
        
        return jsonify({
            'message': '進度已更新',
            'task': {
                'id': task.id,
                'completed_sets': task.completed_sets,
                'target_sets': task.target_sets,
                'completed_reps_total': task.completed_reps_total,
                'avg_smoothness': round(task.avg_smoothness, 1),
                'status': task.status,
                'is_complete': is_complete
            }
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()