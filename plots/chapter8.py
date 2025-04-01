import numpy as np
import matplotlib.pyplot as plt

def add_arrow(line, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0, 0)):
    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    if direction=='right':
        start_ind, end_ind = 0, 1
    else:
        start_ind, end_ind = 1, 0
    
    line.axes.annotate('', 
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha<1 else '-', alpha=alpha),
        size=size
        )
    if text is not None:
        line.axes.annotate(text,
            xytext=(xdata[end_ind]+text_offset[0], ydata[end_ind]+text_offset[1]),
            xy=(xdata[end_ind], ydata[end_ind]),
            size=size
            )


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
    factor = (corners.max(axis=0) - corners.min(axis=0)).max()/2 # ([1,1]-[-1,-1]).max()/2=1
    # in one-image mode: draw counter-clockwise in axis0
    # in two-image mode: draw counter-clockwise in axis0, clockwise in axis1
    for is_clock in range(1+draw_arrows):
        if draw_arrows:
            if binary:
                if is_clock:
                    axs[is_clock].text(-0.5, 0, 'Clockwise')
                    axs[is_clock].text(-0.2, -0.25, 'y=1')
                else:
                    axs[is_clock].text(-0.5, 0, 'Counter-\nClockwise')
                    axs[is_clock].text(-0.2, -0.25, 'y=0')

        for i in range(4):
            coords = corners[i]
            color = basic_colors[i]
            letter = basic_letters[i]

            if not binary:
                targets = [2,3] if is_clock else [1,2]
            else:
                targets = [] 
            
            alpha=0.3 if i in targets else 1.0
            axs[is_clock].scatter(*coords, c=color, s=400, alpha=alpha)

            start = i 
            if is_clock:
                end = i+1 if i<3 else 0
                # for start = lower-left, end = top-left
                arrow_coords = np.stack([
                    corners[start] - clock_arrows[start]*0.15, # tail of ^ from lower-left
                    corners[end] + clock_arrows[start]*0.15]) # head of ^ point to top-left
            else: # counter-clockwise
                end = i-1 if i>0 else -1
                # e.g. for start = lower-left, end = bottom-right
                arrow_coords = np.stack([
                    corners[start] + clock_arrows[end]*0.15, # tail of > point from lower-left
                    corners[end] - clock_arrows[end]*0.15 # head of > point to lower-right  
                ])
            if draw_arrows:
                alpha = 0.3 if ((start in targets) or (end in targets)) else 1.0
            # draw line only when we don't draw arrows (in one-image mode)
            # line is '--' only in one-image-mode and alpha is 0.3 (blurred)
            line = axs[is_clock].plot(
                *arrow_coords.T, 
                c=color, 
                lw=0 if draw_arrows else 2, 
                alpha=alpha, 
                linestyle='--' if (alpha<1) and (not draw_arrows) else '-')[0]

            if draw_arrows:
                add_arrow(line, lw=3, alpha=alpha)

            # points in 'targets" => black char, else white
            axs[is_clock].text(*(coords-factor*np.array([0.05,0.05])), letter, c='k' if i in targets else 'w')
            axs[is_clock].grid(False)

        limits = np.stack([corners.min(axis=0), corners.max(axis=0)]) # [[xmin,ymin],[xmax,ymax]]
        limits = limits.mean(axis=0).reshape(2,1) + 1.2*np.array([[-factor, factor]]) # 1.2*midpoint
        axs[is_clock].set_xlim(limits[0])
        axs[is_clock].set_ylim(limits[1])
        axs[is_clock].set_xlabel(r'$x_0$')
        axs[is_clock].set_ylabel(r'$x_1$', rotation=0)

    fig.tight_layout()
    plt.savefig('test.png')
    return fig 

def plot_sequences(basic_corners=None, basic_colors=None, basic_letters=None, binary=True, target_len=0):
    if basic_corners is None:
        basic_corners = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None:
        basic_letters = ['A','B','C','D']
    fig, axs = plt.subplots(4, 2, figsize=(6,3))
    # cols: class 0 or class 1
    for d in range(2):
        # rows: sequences to give class 0 (or class 1)
        for b in range(4):
            corners = basic_corners[[(b+i)%4 for i in range(4)]][slice(None,None,2*d-1)]
            # before 'slice': ['gray','g','b','r']/['g','b','r','gray']/['b','r','gray','g']/['r','gray','g','b']
            colors = np.array(basic_colors)[[(b+i)%4 for i in range(4)]][slice(None,None,2*d-1)]
            # before 'slice': ['A','B','C','D']/['B','C','D','A']/['C','D','A','B']/['D','A','B','C']
            # [slice(,,-1)] means counter-direction (A-B-C-D); [slice(,,+1)] mean clockwise-direction (D-C-B-A)
            letters = np.array(basic_letters)[[(b+i)%4 for i in range(4)]][slice(None,None,2*d-1)]
            for i in range(4):
                axs[b, d].scatter(i, 0, c=colors[i], s=600, alpha=0.3 if (i+target_len)>=4 else 1.0)
                axs[b, d].text(i-0.125, -0.2, letters[i], c='k' if (i+target_len)>=4 else 'w', fontsize=14)
            axs[b, d].grid(False)
            axs[b, d].set_xticks([])
            axs[b, d].set_yticks([])
            axs[b, d].set_xlim([-0.5, 4])
            axs[b, d].set_ylim([-1, 1])
            if binary:
                axs[b, d].text(4, -0.1, f'y={d}')
            
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 

def plot_data(points, directions, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axs = axs.flatten()
    for e, ax in enumerate(axs):
        pred_corners = points[e]
        clockwise = directions[e]
        color='k'
        ax.scatter(*pred_corners.T, c=color, s=400)
        for i in range(4): # bug here if generate_sequences(..., variable_len=True)
            if i==3:
                start = -1 # so that pred_corners[start+1] turns to pred_corners[0]
            else:
                start = i
            ax.plot(*pred_corners[[start, start+1]].T, c='k', lw=2, alpha=0.5, linestyle='-')
            ax.text(*(pred_corners[i]-np.array([0.04, 0.04])), str(i+1), c='w', fontsize=12)
            if directions is not None:
                ax.set_title(f'{"Counter-" if not clockwise else ""}Clockwise (y={clockwise})', fontsize=14)
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 