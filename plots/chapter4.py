import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, targets, n_plot=30):
  n_rows = n_plot // 6 + ((n_plot%6)>0) # 1 extra row if necessary
  fig, axes = plt.subplots(n_rows, 6, figsize=(9, 1.5*n_rows))
  axes = np.atleast_2d(axes)
  for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
    row, col = i//6, i%6
    ax = axes[row, col]
    ax.set_title('#{} - Label:{}'.format(i, target), {'size': 12})
    ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
  for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.label_outer()
  plt.tight_layout()
  plt.savefig('test.png')
  return fig

def image_channels(red, green, blue, rgb, gray, rows=(0, 1, 2)):
  fig, axs = plt.subplots(len(rows), 4, figsize=(15, 5.5))
  zeros = np.zeros((5, 5), dtype=np.uint8)
  titles1 = ['Red', 'Green', 'Blue', 'Grayscale Image']
  titles0 = ['image_r', 'image_g', 'image_b', 'image_gray']
  titles2 = ['as first channel', 'as second channel', 'as third channel', 'RGB Image']
  idx0 = np.argmax(np.array(rows) == 0)
  idx1 = np.argmax(np.array(rows) == 1)
  idx2 = np.argmax(np.array(rows) == 2)

  for i, m in enumerate([red, green, blue, gray]):
    if 0 in rows:
      axs[idx0, i].axis('off')
      axs[idx0, i].invert_yaxis()
    
    if 1 in rows:
      axs[idx1, i].set_title(titles1[i], fontsize=16)
      axs[idx1, i].set_xlabel('5x5', fontsize=14)
      axs[idx1, i].imshow(m, cmap=plt.cm.gray)
  
  if 1 in rows:
    axs[idx1, 0].set_ylabel('Single\nChannel\n(grayscale)', rotation=0, labelpad=40, fontsize=12)
    axs[idx1, 3].set_xlabel('5x5 = 0.21R + 0.72G + 0.07B')

  fig.tight_layout()
  plt.savefig('test.png')
  return fig