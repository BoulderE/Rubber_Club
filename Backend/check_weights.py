import zipfile, tempfile, h5py, os

with zipfile.ZipFile('lstm_models/lateral_raise/model.keras', 'r') as z:
    with tempfile.TemporaryDirectory() as tmpdir:
        z.extractall(tmpdir)
        with h5py.File(os.path.join(tmpdir, 'model.weights.h5'), 'r') as f:
            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f'  {name}: {obj.shape}')
            f.visititems(show)
