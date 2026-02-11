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
        
        # Changed: wrap in object
        return jsonify({'users': result})
    finally:
        session.close()