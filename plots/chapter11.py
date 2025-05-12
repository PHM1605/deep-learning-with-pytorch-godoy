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

def plot_attention(tokens, alphas):
    # tokens: [ ['[CLS]','The','white',...], ['[CLS]','The','lion'...], [],... ]
    n_tokens = max(list(map(len, tokens)))
    # alphas: [N,n_heads,L,L]
    batch_size, n_heads, _ = alphas[:,:,0,:].shape 
    alphas = alphas.detach().cpu().numpy()[:,:,0,:n_tokens] # [N,n_heads,L]
    fig, axs = plt.subplots(n_heads, batch_size, figsize=(n_tokens*batch_size, n_heads))    

    textcolors = ['white', 'black']
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    for i, axr in enumerate(axs): # head index
        for j, ax in enumerate(axr): # column index i.e. which sentence
            data = alphas[j,i]
            im = ax.imshow(np.array(data.tolist()).reshape(1,-1), vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.grid(False)
            # first row: set x-axis marking
            if i==0:
                ax.set_xticks(np.arange(len(tokens[j]))) 
                ax.set_xticklabels(tokens[j])
            else:
                ax.set_xticks([])
            ax.set_yticks([0])
            ax.set_yticklabels([f'Head #{i}'])
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            for jp in range(data.shape[0]):
                kw.update(color=textcolors[int(im.norm(data[jp])>0.5)])
                text =im.axes.text(jp, 0, valfmt(data[jp], None), **kw)
    return fig 