import torch
import hashlib
import numpy as np

def convert_int(x):
    return int(x, 16)

convert_int_vec = np.vectorize(convert_int)


class HashEmbedding(torch.nn.Module):
    def __init__(
        self, config, output_dim, split_size, hashing_algorithm=hashlib.sha512
    ):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.split_size = split_size
        self.hashing_algorithm = hashing_algorithm
        self.__SAMPLE__HASH = self.hashing_algorithm(b"")
        self.string_size = self.__SAMPLE__HASH.digest_size * 2 # gives us n_hex
        self.parameter_space = torch.nn.EmbeddingBag(
            self.__SAMPLE__HASH.digest_size * 16, self.output_dim, mode="sum"
        )
        self.calc_split_offset()

    def calc_split_offset(self):
        self.SPLIT_OFFSET = np.array(
            list(range(self.string_size // self.split_size)), dtype=np.int32
        )
        self.SPLIT_OFFSET = np.repeat(self.SPLIT_OFFSET, self.split_size, axis=0)
        self.SPLIT_OFFSET = torch.tensor(self.SPLIT_OFFSET, dtype=torch.int32) * 16
        
    def hash_tensor(self, x):
        return self.hashing_algorithm(str(x.item()).encode("UTF-8"))
    
        
    def hash_label_encoding(self, x):
        with torch.no_grad():
            routing = [self.hash_tensor(i) for i in x]
            routing = np.array([list(i.hexdigest()) for i in routing]) # bs, 128, str
            routing = convert_int_vec(routing).astype(np.int32) # bs, 128, int32
            routing = self.SPLIT_OFFSET + routing
        return routing

    def forward(self, x):
        routing = self.hash_label_encoding(x)
        weights = self.parameter_space(routing)
        weights = torch.nn.functional.normalize(weights, dim=1, p=2)
        return weights


# for text classification
class BasicTextClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = HashEmbedding(
            config=None, output_dim=4, split_size=2, hashing_algorithm=hashlib.sha512
        )
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)