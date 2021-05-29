import torch
from torch.utils.tensorboard import SummaryWriter

from .config import CONFIG


class Recorder:

    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)
        self.running_loss = []

    def record_loss(self, loss):
        self.running_loss.append(loss.item())

    def dump_loss(self, step):
        self.writer.add_scalar('loss', torch.mean(torch.tensor(self.running_loss)), step)
        self.running_loss = []

    def record_network(self, step, net_state):
        for name, values in net_state.items():
            self.writer.add_histogram(name, values, step)
