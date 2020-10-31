import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import WikiText2, WikiText103
from torch.utils.data.dataloader import default_collate

class NLPLoader(DataLoader):

    def __init__(self, data, batch_size, **kwargs):
        super().__init__(data, batch_size=batch_size, collate_fn=self.collate, **kwargs)
        self.vocab = data.vocab
        self.inputs = torch.zeros([batch_size, data.dict_length], device='cuda', dtype=torch.float)
        self.context = torch.zeros([batch_size, data.window*2, data.dict_length], device='cuda', dtype=torch.float)
        self.negatives = torch.zeros([batch_size, data.negatives_size, data.dict_length], device='cuda', dtype=torch.float)
        self.context_weight = torch.zeros([batch_size, data.window*2], device='cuda', dtype=torch.float)
        self.negatives_weight = torch.zeros([batch_size, data.negatives_size], device='cuda', dtype=torch.float)

    def weight(self, word):
        if not isinstance(word, str):
            word = self.vocab.itos[word]
        freq = self.vocab.freqs[word]
        if freq < 3:
            return 1.
        return 1/np.log(freq)

    def collate(self, samples):
        self.inputs.zero_()
        self.context.zero_()
        self.negatives.zero_()
        self.context_weight.zero_()
        self.negatives_weight.zero_()
        for idx, (input_word, context_words, negative_words) in enumerate(samples):
            self.inputs[idx, input_word] = 1.
            for c_idx, context_word in enumerate(context_words):
                self.context[idx, c_idx, context_word] = 1.
                self.context_weight[idx, c_idx] = self.weight(context_word)
            for n_idx, negative_word in enumerate(negative_words):
                self.negatives[idx, n_idx, negative_word] = 1.
                self.negatives_weight[idx, n_idx] = self.weight(negative_word)
        return (self.inputs, self.context, self.negatives, self.context_weight, self.negatives_weight)

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
        current_word = int(self.train[word_idx])

        context = []
        def append_context(c_idx):
            if c_idx < 0:
                return False
            if c_idx >= len(self):
                return False
            context_word = int(self.train[c_idx])
            if self.vocab.itos[context_word] in self.delim:
                return False
            context.append(context_word)
            return True

        if self.vocab.itos[current_word] not in self.delim:
            for idx in range(1, self.window + 1):
                if append_context(word_idx - idx) is False:
                    break
            for idx in range(1, self.window + 1):
                if append_context(word_idx + idx) is False:
                    break

        negatives = []
        for n_idx in range(self.negatives_size):
            neg = np.random.randint(self.dict_length)
            while neg == word_idx or neg in context or neg in negatives:
                neg = np.random.randint(self.dict_length)
            # print(f'-negative : {neg}')
            negatives.append(neg)
        return (current_word, context, negatives)
        # return (
        #     torch.tensor(inputs, device='cuda', dtype=torch.float),
        #     torch.tensor(targets, device='cuda', dtype=torch.float),
        #     torch.tensor(negatives, device='cuda', dtype=torch.float)
        # )

    def __call__(self, words):
        if isinstance(words, list) is False:
            words = [words]
        vectors = np.zeros([len(words), self.dict_length])
        for idx, word in enumerate(words):
            vectors[idx, self.vocab.stoi[word]] = 1.
        return torch.tensor(vectors, device='cuda', dtype=torch.float)
