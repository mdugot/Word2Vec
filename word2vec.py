import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from testgen import TestGen
from reduce_analyser import ReduceAnalyser
from data import NLPLoader, WikiData


class Word2Vec(nn.Module):

    def __init__(self, dict_length, latent_space=7):
        super().__init__()
        self.encode_inputs = nn.Linear(dict_length, latent_space, bias=False)
        self.encode_targets = nn.Linear(dict_length, latent_space, bias=False)

    def forward(self, x):
        return self.encode_inputs(x)

    def forward_targets(self, x):
        return self.encode_targets(x)

    def latent_space(self, x):
        return self(x).detach().cpu().numpy()

    def hist(self, word, targets):
        word_vector = self(word)
        target_vectors = self.forward_targets(targets)
        return -torch.log(torch.sigmoid(target_vectors.matmul(word_vector.T))).reshape([-1]).detach().cpu().numpy()


class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, target_vectors, negative_vectors):
        target_loss = -torch.sum(torch.log(torch.sigmoid(target_vectors.matmul(input_vectors.T))))
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_vectors.matmul(input_vectors.T))))
        return target_loss + negative_loss



print('Create network')
# data = TestGen(dict_length=100, negatives_size=20)
data = WikiData()
loader = NLPLoader(data, batch_size=100, shuffle=False)
net = Word2Vec(data.dict_length, latent_space=10)
net.to('cuda')
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = NegativeSamplingLoss()
print('Prepare data')
analyser = ReduceAnalyser(
    pca_words=['france', 'japan', 'bread', 'farm', 'factory', 'bicycle', 'paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
    hist_table={
        'france': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'japan': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'bread': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'farm': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'factory': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'bicycle': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car']
})


epoch = 10
running_loss = []

analyser.draw(data, net)
print("Train")
running_loss = []
t_idx = 0
for e in range(epoch):
    for inputs, targets, negatives in loader:
        optimizer.zero_grad()
        input_vectors = net(inputs)
        for idx in range(loader.batch_size):
            target_vectors = net.forward_targets(targets[idx])
            negative_vectors = net.forward_targets(negatives[idx])
            loss = criterion(input_vectors[idx], target_vectors, negative_vectors)
            running_loss.append(loss.item())
            loss.backward(retain_graph=True)
        optimizer.step()
        t_idx += 1
        if t_idx % 20 == 0:
            print(f'Epoch {e} ({t_idx}/{len(loader)}) - Loss {np.mean(running_loss)}')
            analyser.draw(data, net)
            running_loss = []
