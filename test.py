import torch
from data import WikiData
from word2vec import Word2Vec

data = WikiData()
saved_dict = torch.load('saves/Nov02_16-29-46/307')
net = Word2Vec(
    data.dict_length,
    latent_space=saved_dict['encode_inputs.weight'].shape[0],
    writer=None)
net.to('cuda')
net.load_state_dict(saved_dict)
