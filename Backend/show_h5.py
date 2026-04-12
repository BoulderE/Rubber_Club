import zipfile, tempfile, os, h5py

with zipfile.ZipFile('lstm_models/bicep_curl/model.keras') as z:
    with tempfile.TemporaryDirectory() as d:
        z.extractall(d)
        print("ZIP contents:", z.namelist())
        h5 = os.path.join(d, 'model.weights.h5')
        with h5py.File(h5) as f:
            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f'  {name}  {obj.shape}')
            f.visititems(show)