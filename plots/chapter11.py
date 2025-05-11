import numpy as np 
import matplotlib 
from matplotlib import pyplot as plt 

# wv: full dict vectors
def plot_word_vectors(wv, words, other=None):
    vectors = []
    for word in words:
        try:
            vectors.append(wv[word])
        except KeyError:
            print("Word not in dict=>replace with 'other'")
            if other is not None:
                vectors.append(other[word])
    vectors = np.array(vectors)
    fig, axs = plt.subplots(len(words), 1, figsize=(18, len(words)*0.7))
    if len(words) == 1:
        axs = [axs]

    for i, word in enumerate(words):
        axs[i].imshow(vectors[i].reshape(1,-1), cmap=plt.cm.RdBu, vmin=vectors.min(), vmax=vectors.max())
        axs[i].set_xticks([])  # Remove x-axis ticks entirely
        axs[i].set_yticks([0])  # Set y-tick position
        axs[i].set_yticklabels([word])  # Set label at that position
        axs[i].grid(False)
    
    fig.tight_layout()
    return fig 
