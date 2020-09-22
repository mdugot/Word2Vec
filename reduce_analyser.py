from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ReduceAnalyser:

    def __init__(self, words):
        self.pca = PCA(n_components=2)
        self.words = words
        self.figure = plt.figure(figsize=(10, 10))

    def reduce(self, vectors):
        self.pca.fit_transform(vectors)

    def draw(self, data, network):
        ls = network.latent_space(data(self.words))
        red = self.pca.fit_transform(ls)
        self.figure.clear()
        plt.scatter(red[:,0], red[:,1], marker='.', c='black')
        for idx, word in enumerate(self.words):
            plt.annotate(word, red[idx])
        plt.axis('equal')
        plt.pause(0.01)
