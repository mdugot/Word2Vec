import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import WikiText2, WikiText103
from torch.utils.data.dataloader import default_collate

class NLPLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=self.collate, **kwargs)
 
    def collate(self, samples):
        samples = default_collate(samples)
        samples[0] = samples[0].reshape([self.batch_size, -1])
        return samples

class WikiData(Dataset):

    def __init__(self, negatives_size=10, window=5, delim=['.', '=', '?', '!']):
        self.negatives_size = negatives_size
        self.train, self.test, self.valid = WikiText2()
        self.vocab = self.train.vocab
        self.delim = delim
        self.window = window
        self.dict_length = len(self.vocab.itos)
        print("Dict length : ", self.dict_length)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, word_idx):
        current_word = self.train[word_idx]
        inputs = np.zeros([1, len(self.vocab.itos)])
        inputs[0, current_word] = 1.
        targets = []

        context = []
        def append_context(c_idx):
            if c_idx < 0:
                return False
            if c_idx >= len(self):
                return False
            context_word = self.train[c_idx]
            if self.vocab.itos[context_word] in self.delim:
                return False
            context.append(context_word)
            targets.append(np.zeros([self.dict_length]))
            targets[-1][context_word] = 1.
            return True

        if self.vocab.itos[current_word] not in self.delim:
            for idx in range(1, self.window + 1):
                if append_context(word_idx - idx) is False:
                    break
            for idx in range(1, self.window + 1):
                if append_context(word_idx + idx) is False:
                    break
        while len(targets) < self.window*2:
            targets.append(np.zeros([self.dict_length]))

        all_negs = []
        negatives = []
        for n_idx in range(self.negatives_size):
            neg = np.random.randint(self.dict_length)
            while neg == word_idx or neg in context or neg in all_negs:
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

    def __call__(self, words):
        if isinstance(words, list) is False:
            words = [words]
        vectors = np.zeros([len(words), self.dict_length])
        for idx, word in enumerate(words):
            vectors[idx, self.vocab.stoi[word]] = 1.
        return torch.tensor(vectors, device='cuda', dtype=torch.float)
