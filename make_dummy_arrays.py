import blosc2
import numpy as np
import pathlib
from tqdm import tqdm
import mpire

dummy_path = pathlib.Path('dummy_arrays')
dummy_path.mkdir(exist_ok=True)

# make a bunch of dummy arrays in 2560x143x64 chunks and compress them in individual schunk files
def create_dummy_array(idx):
    rand_arr = np.zeros((8192, 143, 64))
    rand_arr[:, 0:110, :] = np.random.rand(8192, 110, 64).astype(np.float32)
    
    full_path = dummy_path.joinpath(f'dummy_array_{idx}.bschunk')
    schunk_data = np.array(rand_arr, dtype=np.float32)
    args = {
        "chunksize": 32768*143*64,
        "data": schunk_data,
        "contiguous": True,
        "urlpath": full_path.as_posix(),
        "cparams": {
            "nthreads": 16,
            "clevel": 5
        },
        "mode": 'w'
    }

    schunk = blosc2.SChunk(**args)
    return

    
with mpire.WorkerPool(12) as pool:
    pool.map(create_dummy_array, range(100), progress_bar=True)