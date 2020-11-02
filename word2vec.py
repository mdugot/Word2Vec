import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.writer.add_scalar('running loss', torch.mean(torch.tensor(self.running_loss)), step)
        self.running_loss = []
