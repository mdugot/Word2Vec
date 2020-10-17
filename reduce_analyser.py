import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ReduceAnalyser:

    def __init__(self, pca_words, hist_table):
        self.pca = PCA(n_components=2)
        self.words = pca_words
        self.figure = plt.figure(figsize=(13, 13))
        self.pca_plot = self.figure.add_subplot(4, 4, (1,11))
        self.pca_plot.axis('equal')
        self.hist_plots = []
        self.hist_table = hist_table
        for idx, plot_idx in enumerate([4,8,12,13,14,15,16]):
            if idx >= len(hist_table):
                break
            plot = self.figure.add_subplot(4, 4, plot_idx)
            self.hist_plots.append(plot)
        plt.tight_layout()

    def reduce(self, vectors):
        self.pca.fit_transform(vectors)

    def draw(self, data, network):
        ls = network.latent_space(data(self.words))
        red = self.pca.fit_transform(ls)
        self.pca_plot.clear()
        self.pca_plot.scatter(red[:,0], red[:,1], marker='.', c='black')
        for idx, word in enumerate(self.words):
            self.pca_plot.annotate(word, red[idx])
        for idx, (word, targets) in enumerate(self.hist_table.items()):
            hist = network.hist(data(word), data(targets))
            plot = self.hist_plots[idx]
            plot.clear()
            plot.bar(np.arange(len(hist)), hist, width=0.5)
            plot.set_xticks(np.arange(len(hist)))
            plot.set_xticklabels([str(target) for target in targets], rotation=45, fontsize=8)
            plot.set_title(str(word), y=1.0, pad=-14)
        plt.pause(0.01)
