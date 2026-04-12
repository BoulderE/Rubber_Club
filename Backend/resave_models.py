# resave_models_v3.py
import os, json, zipfile, tempfile
import numpy as np
import h5py

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras


def rebuild_model(n_features, seq_len):
    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.LSTM(64, return_sequences=True)(inp)
    x = keras.layers.LSTM(16)(x)
    x = keras.layers.RepeatVector(seq_len)(x)
    x = keras.layers.LSTM(16, return_sequences=True)(x)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)
    return keras.Model(inp, out)


def extract_weights_in_order(h5_path):
    weights = []
    with h5py.File(h5_path, 'r') as f:
        items = []
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                if 'metrics' not in name:
                    items.append((name, np.array(obj)))
        f.visititems(collect)
        items.sort(key=lambda x: x[0])
        for name, arr in items:
            print(f'    {name}: {arr.shape}')
            weights.append(arr)
    return weights


models_dir = 'lstm_models'

for exercise in sorted(os.listdir(models_dir)):
    meta_path = os.path.join(models_dir, exercise, 'metadata.json')
    keras_path = os.path.join(models_dir, exercise, 'model.keras')
    h5_path = os.path.join(models_dir, exercise, 'model.h5')

    if not os.path.isfile(keras_path) or not os.path.isfile(meta_path):
        continue

    with open(meta_path) as f:
        meta = json.load(f)

    n_features = len(meta['angles'])
    seq_len = meta['sequence_len']

    print(f'\n{exercise}: features={n_features}, seq={seq_len}')

    model = rebuild_model(n_features, seq_len)

    with zipfile.ZipFile(keras_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            weights_file = os.path.join(tmpdir, 'model.weights.h5')

            if not os.path.isfile(weights_file):
                print(f'  找不到 model.weights.h5')
                continue

            print('  原始權重:')
            file_weights = extract_weights_in_order(weights_file)

            model_weights = model.get_weights()
            print(f'  模型需要 {len(model_weights)} 個權重，檔案有 {len(file_weights)} 個')

            if len(file_weights) != len(model_weights):
                print(f'  權重數量不匹配')
                continue

            shape_ok = True
            for i, (fw, mw) in enumerate(zip(file_weights, model_weights)):
                if fw.shape != mw.shape:
                    print(f'  權重 #{i} 形狀不匹配: 檔案={fw.shape}, 模型={mw.shape}')
                    shape_ok = False

            if not shape_ok:
                continue

            model.set_weights(file_weights)
            model.save(h5_path)
            print(f'  已匯出 {h5_path}')