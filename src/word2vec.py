import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG

class Word2Vec(nn.Module):

    def __init__(self, dict_length):
        print('number threads : ', torch.get_num_threads())
        super().__init__()
        self.encode_inputs = nn.Linear(dict_length, CONFIG.latent_space, bias=False)
        self.encode_targets = nn.Linear(dict_length, CONFIG.latent_space, bias=False)

    def forward(self, x):
        return self.encode_inputs(x)

    def get_state(self):
        return {
            'gradients/inputs': self.encode_inputs.weight.grad,
            'gradients/target': self.encode_targets.weight.grad,
            'weights/inputs': self.encode_inputs.weight,
            'weights/target': self.encode_targets.weight
        }

    def forward_targets(self, x):
        return self.encode_targets(x)
