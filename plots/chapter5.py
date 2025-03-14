import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 10 + ((n_plot % 10) > 0)
    fig, axes = plt.subplots(n_rows, 10, figsize=(15, 1.5*n_rows))
    axes = np.atleast_2d(axes)
    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
        row, col = i//10, i%10
        ax = axes[row, col]
        ax.set_title(f'#{i} - Label:{target}', {'size': 12})
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()
    fig.tight_layout()
    plt.savefig('test.png')
    return plt
