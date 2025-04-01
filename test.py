import numpy as np

basic_colors = ['gray', 'g', 'b', 'r']
for b in range(4):
    tmp = np.array(basic_colors)[[(b+i)%4 for i in range(4)]][slice(None, None, 0*2-1)][:1]
    print(tmp)
