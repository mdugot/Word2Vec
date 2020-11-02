import torch
from reduce_analyser import ReduceAnalyser
from data import WikiData
from word2vec import Word2Vec

data = WikiData()
analyser = ReduceAnalyser(
    pca_words=['france', 'japan', 'bread', 'farm', 'factory', 'bicycle', 'paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
    hist_table={
        'france': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'japan': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'bread': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'farm': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'factory': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car'],
        'bicycle': ['paris', 'england', 'tokyo', 'wine', 'cheese', 'motor', 'rice', 'robot', 'electricity', 'car']
})

saved_dict = torch.load('saves/Nov02_16-29-46/307')
net = Word2Vec(
    data.dict_length,
    latent_space=saved_dict['encode_inputs.weight'].shape[0],
    writer=None)
net.to('cuda')
net.load_state_dict(saved_dict)
net.eval()
analyser.draw(data, net)
