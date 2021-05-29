import os

import time
from datetime import datetime

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim

from src.data import NLPLoader, WikiData
from src.word2vec import Word2Vec
from src.recorder import Recorder
from src.negative_sampling_loss import NegativeSamplingLoss
from src.config import CONFIG


session = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = os.path.join(CONFIG.save_path, session)
if not os.path.exists(save_path):
    os.makedirs(save_path)
recorder = Recorder(os.path.join(CONFIG.log_path, session))

print('Prepare data')
data = WikiData()
loader = NLPLoader(data, batch_size=CONFIG.batch_size, shuffle=True, num_workers=0)

print('Create network')
net = Word2Vec(data.dict_length)
net.to(CONFIG.device)
net.train()
if CONFIG.restore is not None:
    saved_dict = torch.load(CONFIG.restore)
    net.load_state_dict(saved_dict)
optimizer = optim.SGD(net.parameters(), lr=CONFIG.learning_rate)
scheduler = ExponentialLR(optimizer, gamma=CONFIG.learning_rate_decay)
criterion = NegativeSamplingLoss()
criterion.to(CONFIG.device)

try:
    print("Train")
    step = 0
    for e in range(CONFIG.epoch):
        print(f'Epoch {e}/{CONFIG.epoch}')
        for inputs, targets, negatives, targets_weight, negatives_weight in tqdm(loader):
            optimizer.zero_grad()
            input_vectors = net(inputs)
            for idx in range(loader.batch_size):
                target_vectors = net.forward_targets(targets[idx])
                negative_vectors = net.forward_targets(negatives[idx])
                loss = criterion(input_vectors[idx], target_vectors, negative_vectors, targets_weight[idx], negatives_weight[idx])
                recorder.record_loss(loss)
                loss.backward(retain_graph=True)
            optimizer.step()
            step += 1
            if step % CONFIG.log_cycle == 0:
                recorder.dump_loss(step)
                recorder.record_network(step, net.get_state())
            if step % CONFIG.save_cycle == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f'{step}'))
        scheduler.step()
except KeyboardInterrupt:
    print('\nInterrupt')
torch.save(net.state_dict(), os.path.join(save_path, f'{step}'))
