import h5py
with h5py.File('/home/chenzc/le-wm/data/tworoom.h5', 'r') as f:
    print(list(f.keys()))