from flask import Blueprint, request, jsonify
from models.db_models import get_session, User, ExerciseRecord, AssignedExercise, ExerciseRule
from functools import wraps
from datetime import datetime, timedelta, date
from sqlalchemy import func, desc

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """檢查是否為管理員"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_pin = request.headers.get('X-Admin-Pin')
        
        if not admin_pin:
            return jsonify({'error': '需要管理員認證'}), 401
        
        session = get_session()
        try:
            admin = session.query(User).filter_by(pin=admin_pin, role='admin').first()
            if not admin:
                return jsonify({'error': '管理員權限不足'}), 403
            
            kwargs['admin_user'] = admin
        finally:
            session.close()
        
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/login', methods=['POST'])
def admin_login():
    """管理員登入驗證"""
    data = request.get_json()
    pin = data.get('pin')
    
    if not pin:
        return jsonify({'error': '請輸入 PIN 碼'}), 400
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=pin).first()
        
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        if user.role != 'admin':
            return jsonify({'error': '非管理員帳號'}), 403
        
        return jsonify({
            'message': '登入成功',
            'admin': {
                'id': user.id,
                'name': user.name,
                'pin': user.pin,
                'role': user.role
            }
        })
    finally:
        session.close()


@admin_bp.route('/dashboard', methods=['GET'])
@admin_required
def get_dashboard_stats(admin_user):
    """獲取儀表板統計數據"""
    session = get_session()
    try:
        total_users = session.query(func.count(User.id)).filter(
            User.role != 'admin'
        ).scalar() or 0
        
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_exercises = session.query(func.count(ExerciseRecord.id)).filter(
            ExerciseRecord.created_at >= today_start
        ).scalar() or 0
        
        pending_tasks = session.query(func.count(AssignedExercise.id)).filter(
            AssignedExercise.status.in_(['pending', 'in_progress'])
        ).scalar() or 0
        
        week_start = datetime.combine(
            date.today() - timedelta(days=date.today().weekday()),
            datetime.min.time()
        )
        weekly_stats = session.query(
            func.count(ExerciseRecord.id).label('total_exercises'),
            func.sum(ExerciseRecord.rep_count).label('total_reps'),
            func.avg(ExerciseRecord.smoothness).label('avg_smoothness')
        ).filter(ExerciseRecord.created_at >= week_start).first()
        
        return jsonify({
            'total_users': total_users,
            'today_exercises': today_exercises,
            'pending_tasks': pending_tasks,
            'weekly_stats': {
                'total_exercises': weekly_stats.total_exercises or 0,
                'total_reps': int(weekly_stats.total_reps or 0),
                'avg_smoothness': round(float(weekly_stats.avg_smoothness or 0), 1)
            }
        })
    finally:
        session.close()


@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_all_users(admin_user):
    """獲取所有用戶列表及統計"""
    session = get_session()
    try:
        users = session.query(User).filter(User.role != 'admin').all()
        
        result = []
        for user in users:
            stats = session.query(
                func.count(ExerciseRecord.id).label('total_exercises'),
                func.sum(ExerciseRecord.rep_count).label('total_reps'),
                func.avg(ExerciseRecord.smoothness).label('avg_smoothness'),
                func.max(ExerciseRecord.created_at).label('last_activity')
            ).filter(ExerciseRecord.user_id == user.id).first()
            
            pending = session.query(func.count(AssignedExercise.id)).filter(
                AssignedExercise.user_id == user.id,
                AssignedExercise.status.in_(['pending', 'in_progress'])
            ).scalar() or 0
            
            result.append({
                'id': user.id,
                'pin': user.pin,
                'name': user.name or f'用戶{user.pin}',
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'stats': {
                    'total_exercises': stats.total_exercises or 0,
                    'total_reps': int(stats.total_reps or 0),
                    'avg_smoothness': round(float(stats.avg_smoothness or 0), 1),
                    'last_activity': stats.last_activity.isoformat() if stats.last_activity else None
                },
                'pending_tasks': pending
            })
        
        return jsonify({'users': result})
    finally:
        session.close()


@admin_bp.route('/users/<int:user_id>', methods=['GET'])
@admin_required
def get_user_detail(user_id, admin_user):
    """獲取單一用戶詳細資訊"""
    session = get_session()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        stats = session.query(
            func.count(ExerciseRecord.id).label('total_exercises'),
            func.sum(ExerciseRecord.rep_count).label('total_reps'),
            func.avg(ExerciseRecord.smoothness).label('avg_smoothness'),
            func.max(ExerciseRecord.created_at).label('last_activity')
        ).filter(ExerciseRecord.user_id == user.id).first()
        
        return jsonify({
            'id': user.id,
            'pin': user.pin,
            'name': user.name,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'stats': {
                'total_exercises': stats.total_exercises or 0,
                'total_reps': int(stats.total_reps or 0),
                'avg_smoothness': round(float(stats.avg_smoothness or 0), 1),
                'last_activity': stats.last_activity.isoformat() if stats.last_activity else None
            }
        })
    finally:
        session.close()


@admin_bp.route('/users/<int:user_id>/history', methods=['GET'])
@admin_required
def get_user_history(user_id, admin_user):
    """獲取用戶運動歷史記錄"""
    session = get_session()
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 50, type=int)
        
        since = datetime.utcnow() - timedelta(days=days)
        
        records = session.query(ExerciseRecord).filter(
            ExerciseRecord.user_id == user_id,
            ExerciseRecord.created_at >= since
        ).order_by(desc(ExerciseRecord.created_at)).limit(limit).all()
        
        return jsonify([{
            'id': r.id,
            'exercise_name': r.exercise_name,
            'rep_count': r.rep_count,
            'accuracy': r.accuracy,
            'smoothness': r.smoothness,
            'duration': r.duration,
            'created_at': r.created_at.isoformat() if r.created_at else None
        } for r in records])
    finally:
        session.close()


@admin_bp.route('/users/<int:user_id>/summary', methods=['GET'])
@admin_required
def get_user_summary(user_id, admin_user):
    """獲取用戶運動摘要"""
    session = get_session()
    try:
        weekly_stats = []
        for i in range(7):
            day = date.today() - timedelta(days=6-i)
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            
            stats = session.query(
                func.count(ExerciseRecord.id).label('exercises'),
                func.sum(ExerciseRecord.rep_count).label('total_reps'),
                func.avg(ExerciseRecord.smoothness).label('avg_smoothness')
            ).filter(
                ExerciseRecord.user_id == user_id,
                ExerciseRecord.created_at >= start,
                ExerciseRecord.created_at <= end
            ).first()
            
            weekly_stats.append({
                'date': day.isoformat(),
                'day_name': ['一', '二', '三', '四', '五', '六', '日'][day.weekday()],
                'exercises': stats.exercises or 0,
                'total_reps': int(stats.total_reps or 0),
                'avg_smoothness': round(float(stats.avg_smoothness or 0), 1)
            })
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        breakdown = session.query(
            ExerciseRecord.exercise_name,
            func.count(ExerciseRecord.id).label('count'),
            func.sum(ExerciseRecord.rep_count).label('total_reps'),
            func.avg(ExerciseRecord.smoothness).label('avg_smoothness')
        ).filter(
            ExerciseRecord.user_id == user_id,
            ExerciseRecord.created_at >= thirty_days_ago
        ).group_by(ExerciseRecord.exercise_name).order_by(desc('count')).all()
        
        return jsonify({
            'weekly_stats': weekly_stats,
            'exercise_breakdown': [{
                'exercise_name': b.exercise_name,
                'count': b.count,
                'total_reps': int(b.total_reps or 0),
                'avg_smoothness': round(float(b.avg_smoothness or 0), 1)
            } for b in breakdown]
        })
    finally:
        session.close()

# Admin User Management Endpoints
@admin_bp.route('/users', methods=['POST'])
@admin_required
def create_user(admin_user):
    session = get_session()
    try:
        data = request.get_json()
        name = data.get('name')
        pin = data.get('pin')
        
        if not name or not pin:
            return jsonify({'error': '請輸入用戶名稱和 PIN 碼'}), 400
        
        if not pin.isdigit() or not (4 <= len(pin) <= 6):
            return jsonify({'error': 'PIN 碼必須是 4-6 位數字'}), 400
        
        existing_user = session.query(User).filter_by(pin=pin).first()
        if existing_user:
            return jsonify({'error': 'PIN 碼已存在'}), 400
        
        new_user = User(name=name, pin=pin, role='user', created_at=datetime.utcnow())
        session.add(new_user)
        session.commit()
        
        return jsonify({
            'message': '用戶創建成功',
            'user': {
                'id': new_user.id,
                'name': new_user.name,
                'pin': new_user.pin,
                'role': new_user.role,
                'created_at': new_user.created_at.isoformat() if new_user.created_at else None
            }
        }), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id, admin_user):
    """刪除用戶"""
    session = get_session()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        # Delete associated records first
        session.query(ExerciseRecord).filter_by(user_id=user_id).delete()
        session.query(AssignedExercise).filter_by(user_id=user_id).delete()
        
        # Delete the user
        session.delete(user)
        session.commit()
        
        return jsonify({'message': '用戶已刪除'})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@admin_bp.route('/users/<int:user_id>', methods=['PATCH'])
@admin_required
def update_user(user_id, admin_user):
    """更新用戶資訊"""
    session = get_session()
    try:
        data = request.get_json()
        
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        if 'name' in data:
            user.name = data['name'].strip()
        
        # Update PIN if provided
        if 'pin' in data:
            new_pin = data['pin']
            
            # Validate PIN format
            if not new_pin.isdigit() or not (4 <= len(new_pin) <= 6):
                return jsonify({'error': 'PIN 碼必須是 4-6 位數字'}), 400
            
            # Check for duplicate PIN (excluding current user)
            existing = session.query(User).filter(
                User.pin == new_pin,
                User.id != user_id
            ).first()
            if existing:
                return jsonify({'error': '此 PIN 碼已被使用'}), 400
            
            user.pin = new_pin
        
        # Update role if provided
        if 'role' in data:
            if data['role'] not in ['user', 'admin']:
                return jsonify({'error': '無效的角色'}), 400
            
            # Prevent demoting yourself
            if user.id == admin_user.id and data['role'] != 'admin':
                return jsonify({'error': '無法降級自己的管理員權限'}), 400
            
            user.role = data['role']
        
        session.commit()
        
        return jsonify({
            'message': '用戶更新成功',
            'user': {
                'id': user.id,
                'name': user.name,
                'pin': user.pin,
                'role': user.role,
                'created_at': user.created_at.isoformat() if user.created_at else None
            }
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# Admin Exercise Management Endpoints
@admin_bp.route('/exercises', methods=['GET'])
@admin_required
def get_available_exercises(admin_user):
    """獲取可分配的運動列表"""
    session = get_session()
    try:
        exercises = session.query(ExerciseRule).all()
        return jsonify([{
            'exercise_key': e.exercise_key,
            'name': e.name,
            'description': e.description,
            'difficulties': list(e.params.keys()) if e.params else ['beginner', 'intermediate']
        } for e in exercises])
    finally:
        session.close()


@admin_bp.route('/assign', methods=['POST'])
@admin_required
def assign_exercise(admin_user):
    """分配運動任務給用戶"""
    session = get_session()
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        exercise_key = data.get('exercise_key')
        difficulty = data.get('difficulty', 'beginner')
        target_sets = data.get('target_sets', 1)
        
        if not user_id or not exercise_key:
            return jsonify({'error': '請選擇用戶和運動'}), 400
        
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        exercise = session.query(ExerciseRule).filter_by(exercise_key=exercise_key).first()
        if not exercise:
            return jsonify({'error': '運動類型不存在'}), 404
        
        # Auto-calculate reps based on difficulty
        reps_per_set = 10 if difficulty == 'beginner' else 15
        
        assignment = AssignedExercise(
            user_id=user_id,
            exercise_key=exercise_key,
            exercise_name=exercise.name,
            target_reps=reps_per_set,       # Auto-calculated, NOT from admin input
            target_sets=target_sets,         # From admin input
            difficulty=difficulty,
            due_date=datetime.strptime(data['due_date'], '%Y-%m-%d').date() if data.get('due_date') else None,
            admin_notes=data.get('notes', ''),
            assigned_date=date.today()
        )
        
        session.add(assignment)
        session.commit()
        
        return jsonify({
            'message': '任務分配成功',
            'assignment': {
                'id': assignment.id,
                'user_name': user.name,
                'exercise_name': exercise.name,
                'target_sets': target_sets,
                'reps_per_set': reps_per_set,
                'difficulty': difficulty
            }
        }), 201
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@admin_bp.route('/assignments', methods=['GET'])
@admin_required
def get_all_assignments(admin_user):
    """獲取所有已分配的任務"""
    session = get_session()
    try:
        status_filter = request.args.get('status')
        user_filter = request.args.get('user_id', type=int)
        
        query = session.query(AssignedExercise, User.name.label('user_name')).join(
            User, AssignedExercise.user_id == User.id
        )
        
        if status_filter and status_filter != 'all':
            query = query.filter(AssignedExercise.status == status_filter)
        if user_filter:
            query = query.filter(AssignedExercise.user_id == user_filter)
        
        results = query.order_by(desc(AssignedExercise.created_at)).all()
        
        return jsonify([{
            'id': r.AssignedExercise.id,
            'user_id': r.AssignedExercise.user_id,
            'user_name': r.user_name or f'用戶{r.AssignedExercise.user_id}',
            'exercise_key': r.AssignedExercise.exercise_key,
            'exercise_name': r.AssignedExercise.exercise_name,
            'target_reps': r.AssignedExercise.target_reps,
            'target_sets': r.AssignedExercise.target_sets,
            'completed_sets': r.AssignedExercise.completed_sets,
            'completed_reps_total': r.AssignedExercise.completed_reps_total,
            'avg_smoothness': r.AssignedExercise.avg_smoothness,
            'status': r.AssignedExercise.status,
            'difficulty': r.AssignedExercise.difficulty,
            'assigned_date': r.AssignedExercise.assigned_date.isoformat() if r.AssignedExercise.assigned_date else None,
            'due_date': r.AssignedExercise.due_date.isoformat() if r.AssignedExercise.due_date else None,
            'admin_notes': r.AssignedExercise.admin_notes,
            'created_at': r.AssignedExercise.created_at.isoformat() if r.AssignedExercise.created_at else None
        } for r in results])
    finally:
        session.close()


@admin_bp.route('/assignments/<int:assignment_id>', methods=['PATCH'])
@admin_required
def update_assignment(assignment_id, admin_user):
    session = get_session()
    try:
        data = request.get_json()
        
        assignment = session.query(AssignedExercise).filter_by(id=assignment_id).first()
        if not assignment:
            return jsonify({'error': '任務不存在'}), 404
        
        if 'status' in data:
            assignment.status = data['status']
            if data['status'] == 'completed':
                assignment.completed_at = datetime.utcnow()
        if 'admin_notes' in data:
            assignment.admin_notes = data['admin_notes']
        if 'target_sets' in data:
            assignment.target_sets = data['target_sets']
        if 'due_date' in data:
            assignment.due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date() if data['due_date'] else None
        if 'difficulty' in data:
            assignment.difficulty = data['difficulty']
            assignment.target_reps = 10 if data['difficulty'] == 'beginner' else 15
        
        session.commit()
        return jsonify({'message': '任務更新成功'})
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@admin_bp.route('/assignments/<int:assignment_id>', methods=['DELETE'])
@admin_required
def delete_assignment(assignment_id, admin_user):
    """刪除任務"""
    session = get_session()
    try:
        assignment = session.query(AssignedExercise).filter_by(id=assignment_id).first()
        if not assignment:
            return jsonify({'error': '任務不存在'}), 404
        
        session.delete(assignment)
        session.commit()
        
        return jsonify({'message': '任務已刪除'})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()