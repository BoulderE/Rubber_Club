# save_weights_npz.py
import zipfile, tempfile, os, h5py
import numpy as np
import tensorflow as tf

models_dir = 'lstm_models'

WEIGHT_PATHS = [
    '_layer_checkpoint_dependencies/lstm/cell/vars/0',
    '_layer_checkpoint_dependencies/lstm/cell/vars/1',
    '_layer_checkpoint_dependencies/lstm/cell/vars/2',
    '_layer_checkpoint_dependencies/lstm_2/cell/vars/0',
    '_layer_checkpoint_dependencies/lstm_2/cell/vars/1',
    '_layer_checkpoint_dependencies/lstm_2/cell/vars/2',
    '_layer_checkpoint_dependencies/lstm_4/cell/vars/0',
    '_layer_checkpoint_dependencies/lstm_4/cell/vars/1',
    '_layer_checkpoint_dependencies/lstm_4/cell/vars/2',
    '_layer_checkpoint_dependencies/lstm_6/cell/vars/0',
    '_layer_checkpoint_dependencies/lstm_6/cell/vars/1',
    '_layer_checkpoint_dependencies/lstm_6/cell/vars/2',
    '_layer_checkpoint_dependencies/time_distributed/layer/vars/0',
    '_layer_checkpoint_dependencies/time_distributed/layer/vars/1',
]

for exercise in sorted(os.listdir(models_dir)):
    keras_path = os.path.join(models_dir, exercise, 'model.keras')
    npz_path   = os.path.join(models_dir, exercise, 'weights.npz')
    if not os.path.isfile(keras_path):
        continue

    with zipfile.ZipFile(keras_path) as z:
        with tempfile.TemporaryDirectory() as d:
            z.extractall(d)
            h5_path = os.path.join(d, 'model.weights.h5')
            with h5py.File(h5_path, 'r') as f:
                data = {}
                for i, p in enumerate(WEIGHT_PATHS):
                    data[f'w{i}'] = np.array(f[p])

    # infer input_dim and output_dim from weights
    input_dim  = data['w0'].shape[0]   # first LSTM kernel: (input_dim, 4*units)
    output_dim = data['w13'].shape[0]  # Dense bias: (output_dim,)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, input_dim)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_dim))
    ])

    model.layers[0].set_weights([data['w0'], data['w1'], data['w2']])
    model.layers[1].set_weights([data['w3'], data['w4'], data['w5']])
    model.layers[2].set_weights([data['w6'], data['w7'], data['w8']])
    model.layers[3].set_weights([data['w9'], data['w10'], data['w11']])
    model.layers[4].set_weights([data['w12'], data['w13']])

    rng = np.random.default_rng(42)
    samples = rng.standard_normal((200, 30, input_dim)).astype(np.float32)
    preds = model(samples, training=False).numpy()
    mses = np.mean((samples - preds) ** 2, axis=(1, 2))
    data['error_mean'] = np.array(mses.mean())
    data['error_std']  = np.array(mses.std())
    data['input_dim']  = np.array(input_dim)

    np.savez(npz_path, **data)
    print(f' {exercise}: input_dim={input_dim}, '
          f'error_mean={mses.mean():.4f}, error_std={mses.std():.4f}')