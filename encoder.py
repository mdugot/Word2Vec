import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.data import WikiData
from src.word2vec import Word2Vec


class Encoder:

    def __init__(self, checkpoint=None, pickle=None):
        assert checkpoint is not None or pickle is not None
        if pickle is not None:
            np_data = np.load(pickle, allow_pickle=True)
            self.code = np_data[()]
        else:
            data = WikiData()
            saved_dict = torch.load(checkpoint)
            model = Word2Vec(data.dict_length)
            model = model.to("cpu")
            model.load_state_dict(saved_dict)
            self.code = {}
            print("Encode words")
            with torch.no_grad():
                inputs = torch.zeros([data.dict_length], device="cpu", dtype=torch.float)
                for idx in tqdm(range(data.dict_length)):
                    inputs.zero_()
                    inputs[idx] = 1.
                    self.code[data.trainable_itos[idx]] = model(inputs.view([1, -1])).detach().numpy()[0]

    def save(self, filename):
        np.save(filename, self.code)

    def __call__(self, word):
        return self.code[word]

    def similar_words(self, word, nwords=10):
        if isinstance(word, str):
            word_code = self.code[word]
        else:
            word_code = word
        synonyms = {}
        for other in tqdm(self.code.keys()):
            if not np.all(word_code == self(other)):
                # distance = np.sum(np.power((word_code - self.code[other]), 2))
                dist = distance.cosine(word_code, self(other))
                synonyms[other] = dist
        return [key for key, value in sorted(synonyms.items(), key= lambda item: item[1])][:nwords]

    def plot(self, *words):
        arrs = []
        for word in words:
            arrs.append(self(word))
        codes = np.array(arrs)
        pos2 = TSNE(n_components=2, learning_rate=10., perplexity=5., metric="cosine", init="pca").fit_transform(codes)
        fig, ax = plt.subplots()
        ax.scatter(pos2[:,0], pos2[:,1])
        for idx, word in enumerate(words):
            ax.annotate(word, pos2[idx])
        plt.show()
    
