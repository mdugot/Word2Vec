import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import WikiText2, WikiText103
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from .config import CONFIG

class NLPLoader(DataLoader):

    def __init__(self, data, batch_size, device='cuda', **kwargs):
        super().__init__(data, batch_size=batch_size, collate_fn=self.collate, **kwargs)
        self.device = device
        self.data = data
        self.inputs = torch.zeros([batch_size, data.dict_length], device=self.device, dtype=torch.float)
        self.context = torch.zeros([batch_size, data.window*2, data.dict_length], device=self.device, dtype=torch.float)
        self.negatives = torch.zeros([batch_size, data.negatives_size, data.dict_length], device=self.device, dtype=torch.float)
        self.context_weight = torch.zeros([batch_size, data.window*2], device=self.device, dtype=torch.float)
        self.negatives_weight = torch.zeros([batch_size, data.negatives_size], device=self.device, dtype=torch.float)

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
                self.context_weight[idx, c_idx] = self.data.weights[context_word]
            for n_idx, negative_word in enumerate(negative_words):
                self.negatives[idx, n_idx, negative_word] = 1.
                self.negatives_weight[idx, n_idx] = self.data.weights[negative_word]
        return (self.inputs, self.context, self.negatives, self.context_weight, self.negatives_weight)

class WikiData(Dataset):

    def __init__(self):
        
        self.min_freq = CONFIG.min_freq
        self.negatives_size = CONFIG.negatives_size
        if not os.path.exists("./.data"):
            os.makedirs("./.data")
        text, _, _ = WikiText103()
        self.vocab = text.vocab
        self.delim = ['.', '=', '?', '!']
        self.window = CONFIG.window
        self.trainable_words = {}
        self.trainable_itos = {}
        self.weights = {}
        self.dict_length = 0
        print("Make trainable words dictionary...")
        for word, freq in tqdm(self.vocab.freqs.items()):
            if freq >= self.min_freq:
                word_id = self.dict_length
                self.trainable_words[self.vocab.stoi[word]] = word_id
                self.trainable_itos[word_id] = word
                self.weights[word_id] = 1/np.log((freq - self.min_freq)*0.01 + np.e)
                self.dict_length += 1
        self.trainable_text = []
        print("Remove untrainable words from text...")
        ltext = len(text)
        unk_count = 0
        for idx in tqdm(range(ltext)):
            original_id = int(text[idx])
            if original_id in self.trainable_words:
                new_id = self.trainable_words[original_id]
                self.trainable_text.append(new_id)
            else:
                unk_count += 1

        print(f"trainable text : {len(self.trainable_text)}/{ltext}")
        print(f"trainable words : {self.dict_length}/{len(self.vocab.freqs)}")
        print(f"Unknown words : {unk_count}/{ltext}")

    def __len__(self):
        return len(self.trainable_text)

    def idx_to_id(self, word_idx):
        return self.trainable_text[word_idx]

    def str_to_id(self, word):
        return self.trainable_words[int(self.vocab.stoi[word])]

    def __getitem__(self, word_idx):
        current_word = self.idx_to_id(word_idx)

        context = []
        def append_context(c_idx):
            if c_idx < 0:
                return False
            if c_idx >= len(self):
                return False
            context_word = self.idx_to_id(c_idx)
            if self.trainable_itos[context_word] in self.delim:
                return False
            context.append(context_word)
            return True

        if self.trainable_itos[current_word] not in self.delim:
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
            vectors[idx, self.str_to_id(word)] = 1.
        return torch.tensor(vectors, device=self.device, dtype=torch.float)
