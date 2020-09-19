import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from testgen import TestGen
from tqdm import tqdm
import numpy as np


class Word2Vec(nn.Module):

    def __init__(self, latent_space=10, dict_length=100):
        super().__init__()
        self.encode = nn.Linear(dict_length, latent_space)
        self.decode = nn.Linear(latent_space, dict_length)

    def forward(self, x):
        return torch.sigmoid(self.decode(self.encode(x)))

    def latent_space(self, x):
        return self.encode(x)



print('Create network')
net = Word2Vec()
net.to('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='sum')
print('Prepare data')
data = TestGen()



epoch = 100
number_batch = 1000
batch_size = 200
running_loss = []

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
