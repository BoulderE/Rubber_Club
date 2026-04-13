import os, json
import numpy as np
from collections import deque

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _lstm_forward(x_seq, kernel, rec_kernel, bias, return_sequences=True):
    """
    x_seq:      (seq_len, input_dim)
    kernel:     (input_dim, 4*units)
    rec_kernel: (units, 4*units)
    bias:       (4*units,)
    Keras gate order: i, f, g, o
    """
    units = rec_kernel.shape[0]
    seq_len = x_seq.shape[0]
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    outputs = []

    for t in range(seq_len):
        z = x_seq[t] @ kernel + h @ rec_kernel + bias
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units:2 * units])
        g = np.tanh(z[2 * units:3 * units])
        o = _sigmoid(z[3 * units:])
        c = f * c + i * g
        h = o * np.tanh(c)
        if return_sequences:
            outputs.append(h.copy())

    if return_sequences:
        return np.array(outputs, dtype=np.float32)
    return h


def _autoencoder_forward(seq, w):
    # Encoder
    x = _lstm_forward(seq, w[0], w[1], w[2], return_sequences=True)
    x = _lstm_forward(x,   w[3], w[4], w[5], return_sequences=False)

    # RepeatVector
    x = np.tile(x[np.newaxis, :], (seq.shape[0], 1))

    # Decoder
    x = _lstm_forward(x, w[6],  w[7],  w[8],  return_sequences=True)
    x = _lstm_forward(x, w[9],  w[10], w[11], return_sequences=True)

    # TimeDistributed Dense
    x = x @ w[12] + w[13]

    return x


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

        for name in sorted(os.listdir(models_dir)):
            meta_path = os.path.join(models_dir, name, 'metadata.json')
            npz_path  = os.path.join(models_dir, name, 'weights.npz')

            if not os.path.isfile(meta_path) or not os.path.isfile(npz_path):
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                seq_len    = meta['sequence_len']
                n_features = len(meta['angles'])

                data = np.load(npz_path)
                weights = [data[f'w{i}'].astype(np.float32) for i in range(14)]

                self.models[name] = {
                    'weights':      weights,
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

        print(f"[LSTMScorer] 共載入 {len(self.models)} 個模型 (pure numpy, 無需 tensorflow)")

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
        seq_norm = np.clip(seq_norm, -2.0, 2.0)

        print(f"[DEBUG] seq raw  mean={seq.mean():.2f}  std={seq.std():.2f}  "
          f"min={seq.min():.2f}  max={seq.max():.2f}")
        print(f"[DEBUG] scaler_mean={info['scaler_mean']}")
        print(f"[DEBUG] scaler_scale={info['scaler_scale']}")
        print(f"[DEBUG] seq_norm mean={seq_norm.mean():.2f}  std={seq_norm.std():.2f}  "
            f"min={seq_norm.min():.2f}  max={seq_norm.max():.2f}")

        pred = _autoencoder_forward(seq_norm, info['weights'])

        mse = float(np.mean((seq_norm - pred) ** 2))

        print(f"[DEBUG] pred mean={pred.mean():.4f}  std={pred.std():.4f}")
        print(f"[DEBUG] MSE={mse:.6f}  error_mean={info['error_mean']:.6f}  "
            f"error_std={info['error_std']:.6f}  error_p95={info['error_p95']:.6f}")
        z = (mse - info['error_mean']) / (info['error_std'] + 1e-8)
        print(f"[DEBUG] z={z:.2f}  → score={max(0, min(100, int(round(100 - z * 25))))}")
        
        result = self._mse_to_score(mse, info, key)
        self._cached_scores[key] = result
        return result

    @staticmethod
    def _mse_to_score(mse, info, exercise_key=''):
        import math, sys
        # Per-exercise calibration (accounts for differences in training data distribution)
        _PARAMS = {
            'bicep_curl':     {'mse_scale': 0.35, 'coeff': 9,  'min_w': 0.55},
            'front_raise':    {'mse_scale': 0.65, 'coeff': 10, 'min_w': 0.50},
            'overhead_press': {'mse_scale': 0.65, 'coeff': 10, 'min_w': 0.50},
        }
        params = None
        for pk in _PARAMS:
            if exercise_key.startswith(pk):
                params = _PARAMS[pk]
                break
        scale  = params['mse_scale'] if params else 1.0
        coeff  = params['coeff']     if params else 15
        min_w  = params['min_w']     if params else 0.3
        mse_adj = mse * scale
        p95 = info['error_p95']
        em  = info['error_mean']
        if mse_adj <= em:
            print(f"[LSTM_DEBUG] {exercise_key} mse={mse:.4f} s={scale} adj={mse_adj:.4f} <= em={em:.4f} -> 100", file=sys.stderr)
            return 100
        width = max(p95 - em, min_w)
        ratio = (mse_adj - em) / width
        score = 100 - coeff * math.log2(1 + ratio)
        final = max(30, min(100, int(round(score))))
        print(f"[LSTM_DEBUG] {exercise_key} mse={mse:.4f} s={scale} adj={mse_adj:.4f} em={em:.4f} w={width:.4f} c={coeff} ratio={ratio:.2f} raw={score:.1f} final={final}", file=sys.stderr)
        return final

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