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
      # display all 4 columns in Figure 1, only 3 columns in Figure 2
      if (1 in rows) or (i<3):
        axs[idx0, i].text(0.15, 0.25, str(m.astype(np.uint8)), verticalalignment='top')
        axs[idx0, i].set_title(titles0[i], fontsize=16)
    
    if 1 in rows:
      axs[idx1, i].set_title(titles1[i], fontsize=16)
      axs[idx1, i].set_xlabel('5x5', fontsize=14)
      axs[idx1, i].imshow(m, cmap=plt.cm.gray)
    
    if 2 in rows:
      axs[idx2, i].set_title(titles2[i], fontsize=16)
      axs[idx2, i].set_xlabel(f'5x5x3 - {titles1[i][0]} only', fontsize=14)
      if i<3:
        stacked = [zeros] * 3
        stacked[i] = m 
        axs[idx2, i].imshow(np.stack(stacked, axis=2))
      else:
        axs[idx2, i].imshow(rgb)

    for r in [1, 2]:
      if r in rows:
        idx = idx1 if r == 1 else idx2 
        axs[idx, i].set_xticks([])
        axs[idx, i].set_yticks([])
        for k, v in axs[idx, i].spines.items():
          v.set_color('black')
          v.set_linewidth(.8)

  if 1 in rows:
    axs[idx1, 0].set_ylabel('Single\nChannel\n(grayscale)', rotation=0, labelpad=40, fontsize=12)
    axs[idx1, 3].set_xlabel('5x5 = 0.21R + 0.72G + 0.07B')
  
  if 2 in rows:
    axs[idx2, 0].set_ylabel('Three\nChannels\n(color)', rotation=0, labelpad=40, fontsize=12)
    axs[idx2, 3].set_xlabel('5x5x3 = (R,G,B) stacked')

  fig.tight_layout()
  plt.savefig('test.png')
  return fig