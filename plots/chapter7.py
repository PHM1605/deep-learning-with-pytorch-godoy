import matplotlib.pyplot as plt 
import numpy as np

def compare_grayscale(converted, grayscale):
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    for img, ax, title in zip([converted, grayscale], axs, ['Converted, Grayscale']):
        ax.imshow(img, cmap=plt.cm.gray)
        ax.grid(False)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('test.png')
    return fig

def figure1():
    # (size-in-millions-params,GFLOPS,wrong-percentage)
    data = {
        'AlexNet': (61, 0.727, 41.8),
        'ResNet-18': (12, 2, 30.24),
        'ResNet-34': (22, 4, 26.7),
        'ResNet-50': (26, 4, 24.6),
        'ResNet-101': (45, 8, 23.4),
        'ResNet-152': (60, 11, 23),
        'VGG-16': (138, 16, 28.5),
        'VGG-19': (144, 20, 28.7),
        'Inception-V3': (27, 6, 22.5),
        'GoogLeNet': (13, 2, 34.2),
    }
    names = list(data.keys())
    stats = np.array(list(data.values()))
    # Offset of labeling text 
    xoff = [0, 0, 0, -0.5, 0, 0, 0, 0, -0.7, 0]
    yoff = [1.5, 0, -0.5, 0.5, 1.3, 1.5, 3.5, 3.5, 0.6, 0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # draw big circles representing size
    ax.scatter(stats[:,1], 100-stats[:,2], s=50*stats[:,0], c=np.arange(12,2,-1), cmap=plt.cm.jet)
    # draw white centerpoint representing GFLOPS/WrongPercentage
    ax.scatter(stats[:,1], 100-stats[:,2], c='w', s=4)
    for i, txt in enumerate(names):
        ax.annotate(txt, (stats[i,1]-0.6+xoff[i], 100-stats[i,2]+1.7+yoff[i]), fontsize=12)
    ax.set_xlim([0, 22]) # GFLOPS - Giga Floating Operations Per Second
    ax.set_ylim([50, 85]) # Accuracy
    ax.set_xlabel('Number of Operations - GFLOPS')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Comparing Architectures')
    plt.savefig('test.png')
    return fig 