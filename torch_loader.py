import torch
from torch.utils.data import Dataset
import blosc2
import random
import math
import numpy as np
import pathlib

# have a process to the dataloader, and then have a thread to load the data to main.
# e.g. you have a manager.List() which will store batches
# and you will pop for the next one
# each one of these processes holds it's own tnp or whatever and loads to gpu
# and will conver to DLPack and send to the main process

def load_from_buffer(bits):
    # try: compress buffer, send to gpu, decompress buffer, load
    items = np.frombuffer(bits, dtype=np.float32)
    bs = items.shape[0] // (143 * 64)
    items = items.reshape((bs, 143, 64))
    return items


class BloscDataset(Dataset):
    def __init__(self, path, chunksize):
        self.chunksize = chunksize
        self.paths = []
        tmp_path = pathlib.Path(path)
        for i in tmp_path.iterdir():
            if i.is_file():
                self.paths.append(i)
                
        self.seen_indices = []
        self.batch_buffer = None

    def get_schunk(self, path):
        with open(path, "rb") as f:
            cframe = f.read()
        schunk = blosc2.schunk_from_cframe(cframe, copy=True)
        return schunk
    
    def get_random_schunk_batch(self, schunk):
        # figure out how many chunks to work with
        chunk_num = math.ceil(schunk.nbytes / schunk.chunksize)
        # get a couple random chunks
        i = random.randint(0, chunk_num - 1)
        bits = schunk.decompress_chunk(i)
        items = load_from_buffer(bits)
        return items
    
    def __len__(self):
        return self.chunksize
    
    def load_batch(self, path):
        schunk = self.get_schunk(path)
        schunk_batch = self.get_random_schunk_batch(schunk)
        return schunk_batch
    
    def __getitem__(self, idx):
        if self.batch_buffer is None or len(self.seen_indices) == self.chunksize:
            rand_path = random.choice(self.paths)
            self.batch_buffer = self.load_batch(rand_path)
        
        batch = self.batch_buffer[idx]
        self.seen_indices.append(idx)
        tensor = torch.from_numpy(batch)
        return tensor
    
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    import tqdm
    
    dataset = BloscDataset('dummy_arrays', 8192)
    data_loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=12)
    for i in tqdm.trange(10000):
        tensor = next(iter(data_loader))
        tensor = tensor.cuda()
        
        
        