import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import WikiText2, WikiText103
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

class NLPLoader(DataLoader):

    def __init__(self, data, batch_size, device='cuda', **kwargs):
        super().__init__(data, batch_size=batch_size, collate_fn=self.collate, **kwargs)
        self.vocab = data.vocab
        self.inputs = torch.zeros([batch_size, data.dict_length], device=self.device, dtype=torch.float)
        self.context = torch.zeros([batch_size, data.window*2, data.dict_length], device=self.device, dtype=torch.float)
        self.negatives = torch.zeros([batch_size, data.negatives_size, data.dict_length], device=self.device, dtype=torch.float)
        self.context_weight = torch.zeros([batch_size, data.window*2], device=self.device, dtype=torch.float)
        self.negatives_weight = torch.zeros([batch_size, data.negatives_size], device=self.device, dtype=torch.float)

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

    def __init__(self, negatives_size=30, window=10, delim=['.', '=', '?', '!']):
        self.negatives_size = negatives_size
        self.train, self.test, self.valid = WikiText103()
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

    def __call__(self, words):
        if isinstance(words, list) is False:
            words = [words]
        vectors = np.zeros([len(words), self.dict_length])
        for idx, word in enumerate(words):
            vectors[idx, self.vocab.stoi[word]] = 1.
        return torch.tensor(vectors, device=self.device, dtype=torch.float)

class Dictionary:

    def __init__(self, data, encoder, device='cuda', pickle_dir=None):
        self.device = device
        self.code = {}
        self.target = {}
        self.freqs = data.vocab.freqs
        self.itos = data.vocab.itos
        self.stoi = data.vocab.stoi
        if pickle_dir is None:
            with torch.no_grad():
                inputs = torch.zeros([data.dict_length], device=self.device, dtype=torch.float)
                for idx in tqdm(range(len(data.vocab.itos))):
                    inputs.zero_()
                    inputs[idx] = 1.
                    self.code[data.vocab.itos[idx]] = encoder.latent_space(inputs.view([1, -1]))[0]
                    self.target[data.vocab.itos[idx]] = encoder.target_latent_space(inputs.view([1, -1]))[0]

    def synonyms(self, word, min_freqs=0):
        synonyms = {}
        for other in tqdm(self.code.keys()):
            if other != word and self.freqs[other] >= min_freqs:
                distance = np.sum(np.power((self.code[word] - self.code[other]), 2))
                synonyms[other] = distance
        return [key for key, value in sorted(synonyms.items(), key= lambda item: item[1])]

    def context(self, word, min_freqs=0):
        context = {}
        for other in tqdm(self.code.keys()):
            if other != word and self.freqs[other] >= min_freqs:
                distance = np.dot(self.code[word], self.code[other])
                context[other] = distance
        return [key for key, value in sorted(context.items(), key= lambda item: -item[1])]
