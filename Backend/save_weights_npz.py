import os, json, zipfile, tempfile
import numpy as np
import h5py

models_dir = 'lstm_models'

for exercise in sorted(os.listdir(models_dir)):
    keras_path = os.path.join(models_dir, exercise, 'model.keras')
    meta_path = os.path.join(models_dir, exercise, 'metadata.json')
    npz_path = os.path.join(models_dir, exercise, 'weights.npz')

    if not os.path.isfile(keras_path) or not os.path.isfile(meta_path):
        continue

    with zipfile.ZipFile(keras_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            h5_file = os.path.join(tmpdir, 'model.weights.h5')

            weight_list = []
            with h5py.File(h5_file, 'r') as f:
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and 'metrics' not in name:
                        weight_list.append((name, np.array(obj)))
                f.visititems(collect)

            weight_list.sort(key=lambda x: x[0])
            weights = {f'w{i}': arr for i, (name, arr) in enumerate(weight_list)}

            np.savez(npz_path, **weights)
            print(f'{exercise}: {len(weights)} weights -> weights.npz')