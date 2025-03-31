import numpy as np
import matplotlib.pyplot as plt

# draw_arrows: we want to draw two figures (clock- and counterclockwise) or not 
def counter_vs_clock(basic_corners=None, basic_colors=None, basic_letters=None, draw_arrows=True, binary=True):
    if basic_corners is None: 
        basic_corners = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        clock_arrows = np.array([
            [0,-1], # arrow from lower-left: no change in x-direction, pointing up in y-direction
            [-1,0],
            [0,1],
            [1,0]
        ]) 
    else:
        clock_arrows = np.array([
            [0,basic_corners[0][1]], # arrow from lower-left: no change in x-direction, arrow stemming from 0th point's negative-y-direction
            [basic_corners[1][0],0], # arrow from top-left: no change in y-direction, arrow stemming from 1st point's negative-x-direction
            [0,basic_corners[2][1]], 
            [basic_corners[3,0],0]
        ])    
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None: 
        basic_letters = ['A', 'B', 'C', 'D']
    
    fig, axs = plt.subplots(1, 1+draw_arrows, figsize=(3+3*draw_arrows, 3))
    if not draw_arrows:
        axs = [axs]

    corners = basic_corners[:] # why??
    print(corners.max(axis=0))

    print("CRONERSL:", corners)
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 
