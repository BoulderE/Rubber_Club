from flask import Blueprint, request, jsonify
from models.db_models import get_session, User, AssignedExercise, ExerciseRecord
from datetime import datetime
from sqlalchemy import desc

task_bp = Blueprint('tasks', __name__)

# ============== Admin Routes ==============

@task_bp.route('/admin/stats', methods=['GET'])
def admin_stats():
    """獲取管理員儀表板統計數據"""
    session = get_session()
    try:
        total_users = session.query(User).filter(User.role != 'admin').count()
        total_assignments = session.query(AssignedExercise).count()
        pending_assignments = session.query(AssignedExercise).filter_by(status='pending').count()
        completed_assignments = session.query(AssignedExercise).filter_by(status='completed').count()
        
        stats = {
            'total_users': total_users,
            'total_assignments': total_assignments,
            'pending_assignments': pending_assignments,
            'completed_assignments': completed_assignments,
        }
        
        return jsonify(stats), 200
    finally:
        session.close()


@task_bp.route('/admin/users', methods=['GET'])
def admin_get_users():
    """獲取所有用戶列表（不包括管理員）"""
    session = get_session()
    try:
        users = session.query(User).filter(User.role != 'admin').all()
        
        user_list = []
        for user in users:
            total_tasks = session.query(AssignedExercise).filter_by(user_id=user.id).count()
            completed_tasks = session.query(AssignedExercise).filter_by(user_id=user.id, status='completed').count()
            
            user_list.append({
                'id': user.id,
                'name': user.name,
                'role': user.role,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks
            })
        
        return jsonify(user_list), 200
    finally:
        session.close()


@task_bp.route('/admin/users', methods=['POST'])
def admin_create_user():
    """創建新用戶"""
    session = get_session()
    try:
        data = request.get_json()
        name = data.get('name')
        pin = data.get('pin')
        role = data.get('role', 'user')
        
        if not name or not pin:
            return jsonify({'error': '名稱和PIN碼為必填'}), 400
        
        if len(pin) != 4 or not pin.isdigit():
            return jsonify({'error': 'PIN碼必須為4位數字'}), 400
        
        existing = session.query(User).filter_by(pin=pin).first()
        if existing:
            return jsonify({'error': 'PIN碼已被使用'}), 400
        
        new_user = User(name=name, pin=pin, role=role)
        session.add(new_user)
        session.commit()
        
        return jsonify({
            'message': '用戶創建成功',
            'user': {
                'id': new_user.id,
                'name': new_user.name,
                'role': new_user.role
            }
        }), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/admin/users/<int:user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    """刪除用戶"""
    session = get_session()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        if user.role == 'admin':
            return jsonify({'error': '無法刪除管理員'}), 400
        
        session.query(AssignedExercise).filter_by(user_id=user_id).delete()
        session.query(ExerciseRecord).filter_by(user_id=user_id).delete()
        session.delete(user)
        session.commit()
        
        return jsonify({'message': '用戶已刪除'}), 200
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/admin/assignments', methods=['GET'])
def admin_get_assignments():
    """獲取所有任務分配"""
    session = get_session()
    try:
        user_id = request.args.get('user_id', type=int)
        status = request.args.get('status')
        
        query = session.query(AssignedExercise)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        if status:
            query = query.filter_by(status=status)
        
        assignments = query.order_by(desc(AssignedExercise.id)).all()
        
        result = []
        for a in assignments:
            user = session.query(User).filter_by(id=a.user_id).first()
            result.append({
                'id': a.id,
                'user_id': a.user_id,
                'user_name': user.name if user else 'Unknown',
                'exercise_key': a.exercise_key,
                'exercise_name': a.exercise_name,
                'target_reps': a.target_reps,
                'target_sets': a.target_sets,
                'completed_sets': a.completed_sets,
                'completed_reps_total': a.completed_reps_total,
                'status': a.status,
                'difficulty': a.difficulty,
                'due_date': a.due_date.isoformat() if a.due_date else None,
                'admin_notes': a.admin_notes,
                'created_at': a.created_at.isoformat() if hasattr(a, 'created_at') and a.created_at else None,
                'completed_at': a.completed_at.isoformat() if a.completed_at else None
            })
        
        return jsonify(result), 200
    finally:
        session.close()


@task_bp.route('/admin/assignments', methods=['POST'])
def admin_create_assignment():
    """創建新任務分配"""
    session = get_session()
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        exercise_key = data.get('exercise_key')
        exercise_name = data.get('exercise_name')
        target_reps = data.get('target_reps', 10)
        target_sets = data.get('target_sets', 3)
        difficulty = data.get('difficulty', 'normal')
        due_date_str = data.get('due_date')
        admin_notes = data.get('admin_notes', '')
        
        if not user_id or not exercise_key or not exercise_name:
            return jsonify({'error': '用戶ID、運動代碼和運動名稱為必填'}), 400
        
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        due_date = None
        if due_date_str:
            try:
                due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': '日期格式錯誤，請使用 YYYY-MM-DD'}), 400
        
        new_assignment = AssignedExercise(
            user_id=user_id,
            exercise_key=exercise_key,
            exercise_name=exercise_name,
            target_reps=target_reps,
            target_sets=target_sets,
            completed_sets=0,
            completed_reps_total=0,
            status='pending',
            difficulty=difficulty,
            due_date=due_date,
            admin_notes=admin_notes
        )
        
        session.add(new_assignment)
        session.commit()
        
        return jsonify({
            'message': '任務分配成功',
            'assignment': {
                'id': new_assignment.id,
                'user_id': new_assignment.user_id,
                'exercise_name': new_assignment.exercise_name,
                'target_sets': new_assignment.target_sets,
                'target_reps': new_assignment.target_reps
            }
        }), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/admin/assignments/<int:assignment_id>', methods=['PUT'])
def admin_update_assignment(assignment_id):
    """更新任務分配"""
    session = get_session()
    try:
        data = request.get_json()
        
        assignment = session.query(AssignedExercise).filter_by(id=assignment_id).first()
        if not assignment:
            return jsonify({'error': '任務不存在'}), 404
        
        if 'target_reps' in data:
            assignment.target_reps = data['target_reps']
        if 'target_sets' in data:
            assignment.target_sets = data['target_sets']
        if 'difficulty' in data:
            assignment.difficulty = data['difficulty']
        if 'status' in data:
            assignment.status = data['status']
        if 'admin_notes' in data:
            assignment.admin_notes = data['admin_notes']
        if 'due_date' in data:
            if data['due_date']:
                assignment.due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date()
            else:
                assignment.due_date = None
        
        session.commit()
        
        return jsonify({
            'message': '任務更新成功',
            'assignment': {
                'id': assignment.id,
                'status': assignment.status,
                'target_reps': assignment.target_reps,
                'target_sets': assignment.target_sets
            }
        }), 200
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/admin/assignments/<int:assignment_id>', methods=['DELETE'])
def admin_delete_assignment(assignment_id):
    """刪除任務分配"""
    session = get_session()
    try:
        assignment = session.query(AssignedExercise).filter_by(id=assignment_id).first()
        if not assignment:
            return jsonify({'error': '任務不存在'}), 404
        
        session.delete(assignment)
        session.commit()
        
        return jsonify({'message': '任務已刪除'}), 200
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============== User Task Routes ==============

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