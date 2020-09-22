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
        self.encode = nn.Linear(dict_length, latent_space)
        self.decode = nn.Linear(latent_space, dict_length)

    def forward(self, x):
        return torch.sigmoid(self.decode(self.encode(x)))

    def latent_space(self, x):
        return self.encode(x).detach().cpu().numpy()



print('Create network')
data = TestGen()
net = Word2Vec(len(data))
net.to('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.0002)
criterion = nn.MSELoss(reduction='sum')
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
        inputs, targets = data.batch(batch_size)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Loss {np.mean(running_loss)}')
    analyser.draw(data, net)

