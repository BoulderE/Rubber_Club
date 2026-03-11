from flask import Blueprint, request, jsonify
from models.db_models import ExerciseRule, get_session, User, AssignedExercise, ExerciseRecord
from datetime import datetime
from sqlalchemy import desc, func, case

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
            AssignedExercise.playlist_id.asc().nullslast(),
            AssignedExercise.sort_order.asc(),
            desc(AssignedExercise.status == 'in_progress'),
            AssignedExercise.due_date.asc().nullslast()
        ).all()

        playlists = {}
        standalone = []
        
        for t in tasks:
            task_data = {
                'id': t.id,
                'exercise_key': t.exercise_key,
                'exercise_name': t.exercise_name,
                'target_reps': t.target_reps,
                'target_sets': t.target_sets,
                'completed_sets': t.completed_sets,
                'completed_reps_total': t.completed_reps_total,
                'status': t.status,
                'difficulty': t.difficulty,
                'sort_order': t.sort_order,
                'due_date': t.due_date.isoformat() if t.due_date else None,
                'admin_notes': t.admin_notes,
                'is_overdue': t.due_date < datetime.now().date() if t.due_date else False
            }
            
            if t.playlist_id:
                if t.playlist_id not in playlists:
                    playlists[t.playlist_id] = {
                        'type': 'playlist',
                        'playlist_id': t.playlist_id,
                        'playlist_name': t.playlist_name,
                        'is_routine': t.is_routine,
                        'exercises': []
                    }
                playlists[t.playlist_id]['exercises'].append(task_data)
            else:
                task_data['type'] = 'single'
                standalone.append(task_data)
        
        # Calculate playlist-level progress
        for p in playlists.values():
            total = len(p['exercises'])
            completed = sum(1 for e in p['exercises'] if e['status'] == 'completed')
            p['progress'] = round(completed / total * 100) if total > 0 else 0
        
        result = list(playlists.values()) + standalone
        return jsonify(result)
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

        next_exercise = None
        if is_complete and task.playlist_id:
            next_task = session.query(AssignedExercise).filter(
                AssignedExercise.user_id == user.id,
                AssignedExercise.playlist_id == task.playlist_id,
                AssignedExercise.status.in_(['pending', 'in_progress']),
                AssignedExercise.sort_order > task.sort_order
            ).order_by(AssignedExercise.sort_order.asc()).first()
            
            if next_task:
                next_task.status = 'in_progress'
                session.commit()
                next_exercise = {
                    'id': next_task.id,
                    'exercise_key': next_task.exercise_key,
                    'exercise_name': next_task.exercise_name,
                    'target_reps': next_task.target_reps,
                    'target_sets': next_task.target_sets,
                    'sort_order': next_task.sort_order
                }
        
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
            },
            'next_exercise': next_exercise,
            'playlist_complete': is_complete and next_exercise is None and task.playlist_id is not None
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@task_bp.route('/my-playlists', methods=['GET'])
def get_my_playlists():
    """Get user's playlists (grouped tasks)"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        playlists = session.query(
            AssignedExercise.playlist_id,
            AssignedExercise.playlist_name,
            AssignedExercise.is_routine,
            func.count(AssignedExercise.id).label('exercise_count'),
            func.sum(case((AssignedExercise.status == 'completed', 1), else_=0)).label('completed_count')
        ).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.playlist_id.isnot(None)
        ).group_by(
            AssignedExercise.playlist_id,
            AssignedExercise.playlist_name,
            AssignedExercise.is_routine
        ).all()
        
        return jsonify([{
            'playlist_id': p.playlist_id,
            'playlist_name': p.playlist_name,
            'is_routine': p.is_routine,
            'exercise_count': p.exercise_count,
            'completed_count': p.completed_count,
            'progress': round(p.completed_count / p.exercise_count * 100) if p.exercise_count > 0 else 0
        } for p in playlists])
    finally:
        session.close()


@task_bp.route('/my-playlists/<int:playlist_id>', methods=['GET'])
def get_playlist_tasks(playlist_id):
    """Get all tasks in a playlist, ordered"""
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
            AssignedExercise.playlist_id == playlist_id
        ).order_by(AssignedExercise.sort_order).all()
        
        if not tasks:
            return jsonify({'error': '播放列表不存在'}), 404
        
        return jsonify({
            'playlist_id': playlist_id,
            'playlist_name': tasks[0].playlist_name,
            'is_routine': tasks[0].is_routine,
            'exercises': [{
                'id': t.id,
                'exercise_key': t.exercise_key,
                'exercise_name': t.exercise_name,
                'sort_order': t.sort_order,
                'target_reps': t.target_reps,
                'target_sets': t.target_sets,
                'completed_sets': t.completed_sets,
                'status': t.status
            } for t in tasks]
        })
    finally:
        session.close()


@task_bp.route('/my-playlists', methods=['POST'])
def create_playlist():
    """User creates a new playlist"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        data = request.get_json()
        playlist_name = data.get('name')
        exercises = data.get('exercises', [])  
        is_routine = data.get('is_routine', False)
        
        if not playlist_name or not exercises:
            return jsonify({'error': '需要播放列表名稱和運動項目'}), 400
        
        import time
        playlist_id = int(time.time() * 1000) % 2147483647
        
        for ex in exercises:
            task = AssignedExercise(
                user_id=user.id,
                playlist_id=playlist_id,
                playlist_name=playlist_name,
                exercise_key=ex.get('exercise_key'),
                exercise_name=ex.get('exercise_name'),
                target_reps=ex.get('target_reps', 10),
                target_sets=ex.get('target_sets', 3),
                sort_order=ex.get('sort_order', 0),
                is_routine=is_routine,
                status='pending',
                assigned_date=datetime.now().date()
            )
            session.add(task)
        
        session.commit()
        
        return jsonify({
            'message': '播放列表已創建',
            'playlist_id': playlist_id,
            'playlist_name': playlist_name,
            'exercise_count': len(exercises)
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@task_bp.route('/my-playlists/<int:playlist_id>', methods=['PUT'])
def update_playlist(playlist_id):
    """Update a playlist's name, routine status, and exercises"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        data = request.get_json()
        new_name = data.get('name')
        is_routine = data.get('is_routine')
        exercises = data.get('exercises', [])
        
        # Get existing tasks in playlist
        existing_tasks = session.query(AssignedExercise).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.playlist_id == playlist_id
        ).all()
        
        if not existing_tasks:
            return jsonify({'error': '播放列表不存在'}), 404
        
        # Update playlist name and routine status on all tasks
        for task in existing_tasks:
            if new_name:
                task.playlist_name = new_name
            if is_routine is not None:
                task.is_routine = is_routine
        
        # If exercises provided, update them
        if exercises:
            existing_ids = {t.id for t in existing_tasks}
            incoming_ids = {ex.get('id') for ex in exercises if ex.get('id')}
            
            # Delete removed exercises
            for task in existing_tasks:
                if task.id not in incoming_ids:
                    session.delete(task)
            
            # Update existing and add new exercises
            for ex in exercises:
                if ex.get('id') and ex['id'] in existing_ids:
                    # Update existing
                    task = session.query(AssignedExercise).get(ex['id'])
                    if task:
                        task.exercise_key = ex.get('exercise_key', task.exercise_key)
                        task.exercise_name = ex.get('exercise_name', task.exercise_name)
                        task.target_reps = ex.get('target_reps', task.target_reps)
                        task.target_sets = ex.get('target_sets', task.target_sets)
                        task.sort_order = ex.get('sort_order', task.sort_order)
                else:
                    # Add new exercise
                    new_task = AssignedExercise(
                        user_id=user.id,
                        playlist_id=playlist_id,
                        playlist_name=new_name or existing_tasks[0].playlist_name,
                        exercise_key=ex.get('exercise_key'),
                        exercise_name=ex.get('exercise_name'),
                        target_reps=ex.get('target_reps', 10),
                        target_sets=ex.get('target_sets', 3),
                        sort_order=ex.get('sort_order', 0),
                        is_routine=is_routine if is_routine is not None else existing_tasks[0].is_routine,
                        status='pending',
                        assigned_date=datetime.now().date()
                    )
                    session.add(new_task)
        
        session.commit()
        
        # Fetch updated playlist to return
        updated_tasks = session.query(AssignedExercise).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.playlist_id == playlist_id
        ).order_by(AssignedExercise.sort_order).all()
        
        return jsonify({
            'playlist_id': playlist_id,
            'playlist_name': updated_tasks[0].playlist_name if updated_tasks else new_name,
            'is_routine': updated_tasks[0].is_routine if updated_tasks else is_routine,
            'exercises': [{
                'id': t.id,
                'exercise_key': t.exercise_key,
                'exercise_name': t.exercise_name,
                'sort_order': t.sort_order,
                'target_reps': t.target_reps,
                'target_sets': t.target_sets,
                'completed_sets': t.completed_sets,
                'status': t.status
            } for t in updated_tasks]
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/my-playlists/<int:playlist_id>/save-routine', methods=['POST'])
def save_as_routine(playlist_id):
    """Save a playlist as a reusable routine"""
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
            AssignedExercise.playlist_id == playlist_id
        ).all()
        
        if not tasks:
            return jsonify({'error': '播放列表不存在'}), 404
        
        for task in tasks:
            task.is_routine = True
        
        session.commit()
        
        return jsonify({'message': '已保存為常規訓練'})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@task_bp.route('/my-routines/<int:playlist_id>/start', methods=['POST'])
def start_routine(playlist_id):
    """Start a saved routine (creates new active tasks from routine)"""
    user_pin = request.headers.get('X-User-Pin')
    if not user_pin:
        return jsonify({'error': '需要用戶認證'}), 401
    
    session = get_session()
    try:
        user = session.query(User).filter_by(pin=user_pin).first()
        if not user:
            return jsonify({'error': '用戶不存在'}), 404
        
        # Get routine template
        routine_tasks = session.query(AssignedExercise).filter(
            AssignedExercise.user_id == user.id,
            AssignedExercise.playlist_id == playlist_id,
            AssignedExercise.is_routine == True
        ).all()
        
        if not routine_tasks:
            return jsonify({'error': '常規訓練不存在'}), 404
        
        # Create new playlist from routine
        import time
        new_playlist_id = int(time.time() * 1000) % 2147483647
        
        for rt in routine_tasks:
            new_task = AssignedExercise(
                user_id=user.id,
                playlist_id=new_playlist_id,
                playlist_name=rt.playlist_name,
                exercise_key=rt.exercise_key,
                exercise_name=rt.exercise_name,
                target_reps=rt.target_reps,
                target_sets=rt.target_sets,
                sort_order=rt.sort_order,
                is_routine=False, 
                status='pending',
                assigned_date=datetime.now().date()
            )
            session.add(new_task)
        
        session.commit()
        
        return jsonify({
            'message': '常規訓練已開始',
            'new_playlist_id': new_playlist_id
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@task_bp.route('/exercises', methods=['GET'])
def get_available_exercises():
    session = get_session()
    try:
        exercises = session.query(ExerciseRule).all()
        return jsonify([{
            'exercise_key': e.exercise_key,
            'exercise_name': e.name, 
            'description': e.description
        } for e in exercises])
    finally:
        session.close()