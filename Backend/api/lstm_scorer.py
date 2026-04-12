import os, json
import numpy as np
from collections import deque


class LSTMScorer:

    _LM = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13,    'right_elbow': 14,
        'left_wrist': 15,    'right_wrist': 16,
        'left_hip': 23,      'right_hip': 24,
    }

    def __init__(self, models_dir='lstm_models', score_every_n=5):
        self.models = {}
        self.buffers = {}
        self._score_every_n = score_every_n
        self._frame_counts = {}
        self._cached_scores = {}
        self._load_all(models_dir)

    def _load_all(self, models_dir):
        if not os.path.isdir(models_dir):
            print(f"[LSTMScorer] 模型目錄不存在: {models_dir}")
            return

        try:
            import tf_keras
            self._tf_keras = tf_keras
        except ImportError:
            print("[LSTMScorer] tf_keras 未安裝，LSTM 評分不可用")
            return

        for name in sorted(os.listdir(models_dir)):
            meta_path  = os.path.join(models_dir, name, 'metadata.json')
            npz_path  = os.path.join(models_dir, name, 'weights.npz')

            if not os.path.isfile(meta_path) or not os.path.isfile(npz_path):
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                seq_len = meta['sequence_len']
                n_features = len(meta['angles'])

                # 用代碼重建模型
                model = self._build_model(n_features, seq_len)

                # 載入 npz 權重
                data = np.load(npz_path)
                weights = [data[f'w{i}'] for i in range(len(data.files))]
                model.set_weights(weights)

                self.models[name] = {
                    'model':        model,
                    'angles':       meta['angles'],
                    'seq_len':      seq_len,
                    'scaler_mean':  np.array(meta['scaler_mean'],  dtype=np.float32),
                    'scaler_scale': np.array(meta['scaler_scale'], dtype=np.float32),
                    'error_mean':   meta['error_mean'],
                    'error_std':    meta['error_std'],
                    'error_p95':    meta['error_p95'],
                }
                self.buffers[name] = deque(maxlen=seq_len)
                print(f"[LSTMScorer] {name}  features={n_features}  seq={seq_len}")
            except Exception as e:
                print(f"[LSTMScorer] {name}: {e}")

        print(f"[LSTMScorer] 共載入 {len(self.models)} 個模型")

    @staticmethod
    def _angle_3pt(a, b, c):
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def compute_angles(self, pose_landmarks):
        lm = pose_landmarks.landmark

        def xyz(name):
            i = self._LM[name]
            return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float64)

        ls, rs = xyz('left_shoulder'),  xyz('right_shoulder')
        le, re = xyz('left_elbow'),     xyz('right_elbow')
        lw, rw = xyz('left_wrist'),     xyz('right_wrist')
        lh, rh = xyz('left_hip'),       xyz('right_hip')

        angles = {
            'r_elbow_ang':    self._angle_3pt(rs, re, rw),   
            'l_elbow_ang':    self._angle_3pt(ls, le, lw),   
            'r_shoulder_ang': self._angle_3pt(rh, rs, re),  
            'l_shoulder_ang': self._angle_3pt(lh, ls, le),   
        }

        mid_hip = (lh + rh) / 2
        mid_sh  = (ls + rs) / 2
        spine   = mid_sh - mid_hip
        cos_lean = np.clip(-spine[1] / (np.linalg.norm(spine) + 1e-8), -1.0, 1.0)
        angles['torso_lean'] = float(np.degrees(np.arccos(cos_lean)))

        return angles

    def _resolve_key(self, exercise_key, active_side=None):
        if exercise_key == 'diagonal_lift' and active_side:
            return f'diagonal_lift_{active_side}'
        return exercise_key

    def score(self, exercise_key, pose_landmarks, active_side=None):
        key = self._resolve_key(exercise_key, active_side)
        if key not in self.models:
            return None

        info = self.models[key]
        buf  = self.buffers[key]

        all_angles = self.compute_angles(pose_landmarks)

        try:
            frame = np.array([all_angles[a] for a in info['angles']], dtype=np.float32)
        except KeyError:
            return None

        buf.append(frame)
        if len(buf) < info['seq_len']:
            return None   
        
        self._frame_counts[key] = self._frame_counts.get(key, 0) + 1
        if self._frame_counts[key] % self._score_every_n != 0:
            return self._cached_scores.get(key)  

        seq = np.array(buf, dtype=np.float32)                
        seq_norm = (seq - info['scaler_mean']) / info['scaler_scale']
        x = seq_norm[np.newaxis, ...]                        

        pred = info['model'](x, training=False).numpy()

        mse = float(np.mean((x - pred) ** 2))
        result = self._mse_to_score(mse, info)
        self._cached_scores[key] = result
        return result
    
    @staticmethod
    def _mse_to_score(mse, info):
        z = (mse - info['error_mean']) / (info['error_std'] + 1e-8)
        if z <= 0:
            return 100
        return max(0, min(100, int(round(100 - z * 25))))

    def reset(self, exercise_key=None, active_side=None):
        if exercise_key:
            key = self._resolve_key(exercise_key, active_side)
            if key in self.buffers:
                self.buffers[key].clear()
            self._frame_counts.pop(key, None)
            self._cached_scores.pop(key, None)
        else:
            for buf in self.buffers.values():
                buf.clear()
            self._frame_counts.clear()
            self._cached_scores.clear()