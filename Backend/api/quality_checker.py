import os
import numpy as np
import mediapipe as mp


class QualityChecker:

    ANGLE_DEFS = {
        'r_shoulder_ang': ('RIGHT_HIP',      'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        'l_shoulder_ang': ('LEFT_HIP',       'LEFT_SHOULDER',  'LEFT_ELBOW'),
        'r_elbow_ang':    ('RIGHT_SHOULDER', 'RIGHT_ELBOW',    'RIGHT_WRIST'),
        'l_elbow_ang':    ('LEFT_SHOULDER',  'LEFT_ELBOW',     'LEFT_WRIST'),
    }

    def __init__(self, rules_path=None):
        if rules_path is None:
            rules_path = os.path.join(
                os.path.dirname(__file__), '..', 'quality_output', 'quality_rules.joblib'
            )
        self.rules = {}
        self.mp_pose = mp.solutions.pose
        self._consec_bad = {}         

        try:
            import joblib
            self.rules = joblib.load(rules_path)
            print(f"✅ Quality rules loaded: {list(self.rules.keys())}")
        except Exception as e:
            print(f"⚠️ Quality rules not loaded: {e}")

    @staticmethod
    def _angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def compute_angles(self, pose_landmarks):
        """Compute 4 joint angles from raw MediaPipe pose landmarks."""
        if not pose_landmarks:
            return None
        angles = {}
        for name, (a_n, b_n, c_n) in self.ANGLE_DEFS.items():
            try:
                a = pose_landmarks.landmark[self.mp_pose.PoseLandmark[a_n].value]
                b = pose_landmarks.landmark[self.mp_pose.PoseLandmark[b_n].value]
                c = pose_landmarks.landmark[self.mp_pose.PoseLandmark[c_n].value]
                if min(a.visibility, b.visibility, c.visibility) < 0.5:
                    continue
                angles[name] = self._angle(
                    [a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]
                )
            except (KeyError, IndexError):
                continue
        return angles

    def check(self, exercise_id, angles, smooth_frames=3):
        empty = {'is_bad': False, 'error_type': None, 'message': None, 'violations': []}

        if not angles or exercise_id not in self.rules:
            self._consec_bad[exercise_id] = 0
            return empty

        rule = self.rules[exercise_id]
        violations = []

        if rule['type'] == 'data_driven':
            for chk in rule['checks']:
                ang = chk['angle']
                if ang not in angles:
                    continue
                val = angles[ang]
                if chk['direction'] == 'above' and val > chk['threshold']:
                    violations.append(ang)
                elif chk['direction'] == 'below' and val < chk['threshold']:
                    violations.append(ang)
            raw_bad = len(violations) >= min(2, len(rule['checks']))

        elif rule['type'] == 'range_based':
            for ang, stats in rule['angle_stats'].items():
                if ang not in angles:
                    continue
                val = angles[ang]
                if val < stats['p5'] or val > stats['p95']:
                    violations.append(ang)
            raw_bad = len(violations) >= 2

        else:
            return empty

        cnt = self._consec_bad.get(exercise_id, 0)
        cnt = cnt + 1 if raw_bad else max(0, cnt - 1)
        self._consec_bad[exercise_id] = cnt

        if cnt >= smooth_frames:
            return {
                'is_bad': True,
                'error_type': rule['error_name'],
                'message': rule['message'],
                'violations': violations,
            }
        return {**empty, 'violations': violations}