import os
from datetime import datetime

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from data import NLPLoader, WikiData
from word2vec import Word2Vec, NegativeSamplingLoss


epoch = 30
log_cycle = 300
save_cycle = 20000
latent_space = 100
learning_rate = 0.1
batch_size = 100

path = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = os.path.join('saves', path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
summary_path = os.path.join('summaries', path)

print('Prepare data')
data = WikiData()
loader = NLPLoader(data, batch_size=batch_size, shuffle=False)
writer = SummaryWriter(log_dir=summary_path)
print('Create network')
net = Word2Vec(data.dict_length, latent_space=latent_space, writer=writer)
net.to('cuda')
net.train()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = NegativeSamplingLoss(writer=writer)
criterion.to('cuda')


try:
    print("Train")
    step = 0
    for e in range(epoch):
        print(f'Epoch {e}/{epoch}')
        for inputs, targets, negatives, targets_weight, negatives_weight in tqdm(loader):
            optimizer.zero_grad()
            input_vectors = net(inputs)
            for idx in range(loader.batch_size):
                target_vectors = net.forward_targets(targets[idx])
                negative_vectors = net.forward_targets(negatives[idx])
                loss = criterion(input_vectors[idx], target_vectors, negative_vectors, targets_weight[idx], negatives_weight[idx])
                loss.backward(retain_graph=True)
            step += 1
            if step % log_cycle == 0:
                criterion.log(step)
                net.log(step)
            if step % save_cycle == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f'{step}'))
            optimizer.step()
except KeyboardInterrupt:
    print('\nInterrupt')
torch.save(net.state_dict(), os.path.join(save_path, f'{step}'))