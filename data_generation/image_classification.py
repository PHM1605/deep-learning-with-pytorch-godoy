import numpy as np

# start: [-9, 9]; target: [0,1,2]
def gen_img(start, target, fill=1, img_size=10):
  img = np.zeros((img_size, img_size), dtype=float)
  start_row, start_col = None, None 
  if start>0:
    start_row = start 
  else:
    start_col = np.abs(start)
  # vertical or horizontal line
  if target==0:
    if start_row is None:
      img[:, start_col] = fill # a vertical line at start_col
    else:
      img[start_row, :] = fill # a horizontal line at start_row
  else:
    if start_col == 0:
      start_col = 1 # start col only from column 1 to column 9
    if target==1:
      # diagonal, row-starting from bottom-left to up-right
      if start_row is not None:
        up = (range(start_row, -1, -1), range(0, start_row+1))
      # diagonal, column-starting from bottom-left to up-right
      else:
        up = (range(img_size-1, start_col-1, -1), range(start_col, img_size))
      img[up] = fill
    else: # target == 2
      # diagonal, row-starting from top-left to bottom-right
      if start_row is not None:
        down = (range(start_row, img_size, 1), range(0, img_size-start_row))
      # diagonal, column-starting from top-left to bottom-right
      else:
        down = (range(0, img_size-start_col), range(start_col, img_size))
      img[down] = fill 
  return 255 * img.reshape(1, img_size, img_size)

def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
  np.random.seed(seed)
  starts = np.random.randint(-(img_size-1), img_size, size=(n_images,)) # generate 100 numbers in range [-9,9]
  targets = np.random.randint(0, 3, size=(n_images,)) # generate 100 numbers in range [0,1,2]
  images = np.array([
    gen_img(s, t, img_size=img_size)
    for s, t in zip(starts, targets)
    ], dtype=np.uint8)
  # target if 1 or 2 -> 'diagonal' class, else if 0 -> 'straight' class
  if binary:
    targets = (targets>0).astype(int)
  return images, targets
