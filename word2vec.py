import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from testgen import TestGen
from reduce_analyser import ReduceAnalyser
from data import NLPLoader, WikiData


class Word2Vec(nn.Module):

    def __init__(self, dict_length, writer, latent_space=7):
        super().__init__()
        self.encode_inputs = nn.Linear(dict_length, latent_space, bias=False)
        self.encode_targets = nn.Linear(dict_length, latent_space, bias=False)
        self.writer = writer

    def forward(self, x):
        return self.encode_inputs(x)

    def log(self, step):
        self.writer.add_histogram('gradient/inputs', self.encode_inputs.weight.grad, step)
        self.writer.add_histogram('gradient/targets', self.encode_targets.weight.grad, step)
        self.writer.add_histogram('weight/inputs', self.encode_inputs.weight, step)
        self.writer.add_histogram('weight/targets', self.encode_targets.weight, step)

    def forward_targets(self, x):
        return self.encode_targets(x)

    def latent_space(self, x):
        return self(x).detach().cpu().numpy()

    def hist(self, word, targets):
        word_vector = self(word)
        target_vectors = self.forward_targets(targets)
        return -torch.log(torch.sigmoid(target_vectors.matmul(word_vector.T))).reshape([-1]).detach().cpu().numpy()


class NegativeSamplingLoss(nn.Module):

    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.running_loss = []

    def forward(self, input_vectors, target_vectors, negative_vectors, targets_weight, negatives_weight):
        target_loss = -torch.sum(torch.log(torch.sigmoid(target_vectors.matmul(input_vectors.T))) * targets_weight)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_vectors.matmul(input_vectors.T))) * negatives_weight)
        loss = target_loss + negative_loss
        self.running_loss.append(loss.item())
        return loss

    def log(self, step):
        self.writer.add_scalar('running loss', np.mean(self.running_loss), step)
        self.running_loss = []



print('Create network')
# data = TestGen(dict_length=100, negatives_size=20)
data = WikiData()
loader = NLPLoader(data, batch_size=100, shuffle=False)
writer = SummaryWriter()
net = Word2Vec(data.dict_length, latent_space=100, writer=writer)
net.to('cuda')
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = NegativeSamplingLoss(writer=writer)
criterion.to('cuda')
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
step = 0
for e in range(epoch):
    print(f'Epoch {e}/epoch')
    for inputs, targets, negatives, targets_weight, negatives_weight in tqdm(loader):
        optimizer.zero_grad()
        input_vectors = net(inputs)
        for idx in range(loader.batch_size):
            target_vectors = net.forward_targets(targets[idx])
            negative_vectors = net.forward_targets(negatives[idx])
            loss = criterion(input_vectors[idx], target_vectors, negative_vectors, targets_weight[idx], negatives_weight[idx])
            running_loss.append(loss.item())
            loss.backward(retain_graph=True)
        if step % 10 == 0:
            criterion.log(step)
            net.log(step)
        optimizer.step()
        step += 1
    analyser.draw(data, net)
