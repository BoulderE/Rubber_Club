from models.db_models import init_db, get_session, User, ExerciseRule

EXERCISE_CONFIG = {
    'bicep_curl': {
        'name': '胸部拉伸',
        'landmarks_to_use': ['right_shoulder', 'right_wrist'],
        'logic_function': '_analyze_bicep_curl_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': -0.015,
                'end_threshold_y': 0.03,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': -0.010,
                'end_threshold_y': 0.04,
                'over_extension_threshold_y': -0.16,
                'min_distance': 0.010
            }
        }
    },
    'chest_pull': {
        'name': '胸前拉開',
        'landmarks_to_use': ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist'],
        'logic_function': '_analyze_chest_pull_logic',
        'params': {
            'intermediate': {
                'start_threshold_wx': 0.28,
                'end_threshold_wx': 0.30,
                'min_distance_wx': 0.01,
                'min_wrist_rel_y': -0.28,
                'max_wrist_rel_y': 0.38
            },
            'beginner': {
                'start_threshold_wx': 0.26,
                'end_threshold_wx': 0.275,
                'min_distance_wx': 0.01,
                'min_wrist_rel_y': -0.28,
                'max_wrist_rel_y': 0.38
            }
        }
    },
    'lateral_raise': {
        'name': '侧平举',
        'landmarks_to_use': ['right_shoulder', 'right_elbow', 'right_wrist'],
        'logic_function': '_analyze_lateral_raise_logic',
        'params': {
            'intermediate': {
                'start_threshold_x': 0.15,
                'end_threshold_x': 0.18,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_x': 0.12,
                'end_threshold_x': 0.16,
                'over_extension_threshold_y': -0.16,
                'min_distance': 0.015
            }
        }
    },
    'front_raise': {
        'name': '前平举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow', 'right_hip'],
        'logic_function': '_analyze_front_raise_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.70,
                'end_threshold_y': 0.20,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': 0.75,
                'end_threshold_y': 0.30,
                'min_distance': 0.012
            }
        }
    },
    'overhead_press': {
        'name': '过顶举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_overhead_press_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.08,
                'end_threshold_y': -0.18,
                'min_distance': 0.01
            },
            'beginner': {
                'start_threshold_y': 0.10,
                'end_threshold_y': -0.15,
                'min_distance': 0.008
            }
        }
    },
    'diagonal_lift': {
        'name': '對角線動作',
        'landmarks_to_use': [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip'
        ],
        'logic_function': '_analyze_diagonal_lift_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.20,
                'end_threshold_y': 0.00,
                'min_horizontal_disp': 0.08,
                'min_vertical_disp': 0.10,
                'min_distance': 0.18,
                'min_diagonal_angle': 8,
                'max_diagonal_angle': 85,
            },
            'beginner': {
                'start_threshold_y': 0.22,
                'end_threshold_y': 0.05,
                'min_horizontal_disp': 0.08,
                'min_vertical_disp': 0.10,
                'min_distance': 0.18,
                'min_diagonal_angle': 8,
                'max_diagonal_angle': 85,
            }
        }
    },
    'squat': {
        'name': '深蹲',
        'landmarks_to_use': ['right_hip', 'right_knee', 'right_ankle'],
        'logic_function': '_analyze_squat_logic',
        'params': {
            'intermediate': {
                'up_threshold_angle': 165.0,
                'down_threshold_angle': 95.0
            },
            'beginner': {
                'up_threshold_angle': 160.0,
                'down_threshold_angle': 110.0
            }
        }
    },
}

def seed_data():
    session = get_session()
    
    # 1. 創建測試用戶
    test_users = [
        {'pin': '1234', 'name': '測試用戶1'},
        {'pin': '5678', 'name': '測試用戶2'},
        {'pin': '0000', 'name': '管理員'},
    ]
    
    for user_data in test_users:
        existing = session.query(User).filter_by(pin=user_data['pin']).first()
        if not existing:
            user = User(**user_data)
            session.add(user)
            print(f"✅ 創建用戶: {user_data['name']}")
    
    # 2. 導入運動規則
    for exercise_key, config in EXERCISE_CONFIG.items():
        existing = session.query(ExerciseRule).filter_by(exercise_key=exercise_key).first()
        if not existing:
            rule = ExerciseRule(
                exercise_key=exercise_key,
                name=config['name'],
                landmarks_to_use=config['landmarks_to_use'],
                logic_function=config['logic_function'],
                params=config['params']
            )
            session.add(rule)
            print(f"✅ 創建運動規則: {config['name']}")
    
    session.commit()
    session.close()
    print("\n✅ 所有數據導入完成！")

if __name__ == '__main__':
    print("=== 初始化數據庫 ===")
    init_db()
    print("\n=== 導入初始數據 ===")
    seed_data()