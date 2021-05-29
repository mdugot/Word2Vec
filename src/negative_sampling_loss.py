import torch
import torch.nn as nn


class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, target_vectors, negative_vectors, targets_weight, negatives_weight):
        epsilon = 1e-10
        target_loss = -torch.sum(torch.log(torch.sigmoid(target_vectors.matmul(input_vectors.T)) + epsilon) * targets_weight)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_vectors.matmul(input_vectors.T)) + epsilon) * negatives_weight)
        loss = target_loss + negative_loss
        return loss
