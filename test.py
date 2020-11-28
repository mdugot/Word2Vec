import torch
from data import WikiData, Dictionary
from word2vec import Word2Vec

print('test')
data = WikiData()
saved_dict = torch.load('saves/Nov18_07-22-53/427240')
net = Word2Vec(
    data.dict_length,
    latent_space=saved_dict['encode_inputs.weight'].shape[0],
    writer=None)
net.to('cuda')
net.load_state_dict(saved_dict)
net.eval()
dictionary = Dictionary(data, net, device='cuda')
