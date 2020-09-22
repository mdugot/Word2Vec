import torch
import numpy as np


class TestGen:

    def __init__(self, dict_length=100):
        self.dict_length = dict_length
        self.contexts = []
        for n in range(dict_length):
            context = {n}
            if n > 0:
                context.add(1)
            for f in range(1, n//2):
                if n % f == 0:
                    context = context.union(self.contexts[f])
            self.contexts.append(context)


    def batch(self, batch_size):
        inputs = np.zeros([batch_size, self.dict_length])
        targets = np.zeros([batch_size, self.dict_length])
        for idx in range(batch_size):
            n = np.random.randint(self.dict_length)
            inputs[idx, n] = 1.
            for c in self.contexts[n]:
                targets[idx, c] = 1.
        return torch.tensor(inputs, device='cuda', dtype=torch.float), torch.tensor(targets, device='cuda', dtype=torch.float)

    def __len__(self):
        return self.dict_length

    def __call__(self, words):
        vectors = np.zeros([len(words), len(self)])
        for idx, word in enumerate(words):
            n = int(word)
            assert n >= 0 and n < len(self)
            vectors[idx, n] = 1.
        return torch.tensor(vectors, device='cuda', dtype=torch.float)
