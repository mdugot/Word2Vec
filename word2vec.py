import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from testgen import TestGen
from reduce_analyser import ReduceAnalyser


class Word2Vec(nn.Module):

    def __init__(self, dict_length, latent_space=7):
        super().__init__()
        self.encode_inputs = nn.Linear(dict_length, latent_space)
        self.encode_targets = nn.Linear(dict_length, latent_space)

    def forward(self, x):
        return self.encode_inputs(x)

    def forward_targets(self, x):
        return self.encode_targets(x)

    def latent_space(self, x):
        return self(x).detach().cpu().numpy()


class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, target_vectors, negative_vectors):
        target_loss = -torch.sum(torch.log(torch.sigmoid(target_vectors.matmul(input_vectors.T))))
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_vectors.matmul(input_vectors.T))))
        return target_loss + negative_loss



print('Create network')
data = TestGen(dict_length=100)
net = Word2Vec(len(data), latent_space=5)
net.to('cuda')
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = NegativeSamplingLoss()
print('Prepare data')
analyser = ReduceAnalyser(words=list(range(100)))
# analyser = ReduceAnalyser(words=[
#     40, 44, 80, 88, 
#     19, 61, 97,
#     5, 3, 15, 45
# ])



epoch = 1000
number_batch = 1000
batch_size = 500
running_loss = []

analyser.draw(data, net)
print("Train")
for e in range(epoch):
    print(f'Epoch {e}')
    running_loss = []
    for b in tqdm(range(number_batch)):
        optimizer.zero_grad()
        inputs, targets, negatives = data.negative_sampling(negatives_size=20)
        input_vectors = net(inputs)
        target_vectors = net.forward_targets(inputs)
        negative_vectors = net.forward_targets(negatives)
        loss = criterion(input_vectors, target_vectors, negative_vectors)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Loss {np.mean(running_loss)}')
    analyser.draw(data, net)
