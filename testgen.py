import torch
from torch.utils.data import Dataset
import numpy as np


class TestGen(Dataset):

    def __init__(self, dict_length=100, negatives_size=10):
        self.dict_length = dict_length
        self.negatives_size = negatives_size
        self.contexts = []
        for n in range(dict_length):
            context = {n}
            if n > 0:
                context.add(1)
            for f in range(1, n//2 + 1):
                if n % f == 0:
                    context.add(f)
            self.contexts.append(context)
        self.max_context = max([len(context) for context in self.contexts])

    def negative_sampling(self, n=None):
        if n is None:
            n = np.random.randint(self.dict_length)
        # print(f'value : {n}')
        inputs = np.zeros([1, self.dict_length])
        inputs[0, n] = 1.
        targets = []
        for c_idx, c in enumerate(self.contexts[n]):
            targets.append(np.zeros([self.dict_length]))
            targets[c_idx][c] = 1.
        assert len(targets) <= self.max_context
        while len(targets) < self.max_context:
            targets.append(np.zeros([self.dict_length]))
        all_negs = []
        negatives = []
        for n_idx in range(self.negatives_size):
            neg = np.random.randint(self.dict_length)
            while neg == n or neg in self.contexts[n] or neg in all_negs:
                neg = np.random.randint(self.dict_length)
            # print(f'-negative : {neg}')
            all_negs.append(neg)
            negatives.append(np.zeros([self.dict_length]))
            negatives[n_idx][neg] = 1.
        return (
            torch.tensor(inputs, device='cuda', dtype=torch.float),
            torch.tensor(targets, device='cuda', dtype=torch.float),
            torch.tensor(negatives, device='cuda', dtype=torch.float)
        )

    def __len__(self):
        return self.dict_length

    def __getitem__(self, idx):
        return self.negative_sampling(idx)

    def __call__(self, words):
        if not isinstance(words, list):
            words = [words]
        vectors = np.zeros([len(words), len(self)])
        for idx, word in enumerate(words):
            n = int(word)
            assert n >= 0 and n < len(self)
            vectors[idx, n] = 1.
        return torch.tensor(vectors, device='cuda', dtype=torch.float)
