import matplotlib.pyplot as plt 
import matplotlib 
import numpy as np 

def add_arrow(line, position=None, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0,0)):
    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if position is None:
        position = xdata.mean()
    start_ind = 0
    end_ind = 1

    line.axes.annotate('',
        # xytext=(xdata[start_ind], ydata[start_ind]),
        xytext=(0, 0),
        xy = (xdata[end_ind], ydata[end_ind]),
        arrowprops = dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha<1 else '-', alpha=alpha),
        size=size
        )
    if text is not None:
        line.axes.annotate(text, 
            color=color, 
            # xy=(xdata[end_ind], ydata[end_ind]),
            xy=(0, 0),
            xytext=(xdata[end_ind]+text_offset[0], ydata[end_ind]+text_offset[1]),
            size=size
            )


# X: [128,4,2]
def sequence_pred(sbs_obj, X, directions=None, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axs = axs.flatten()

    for e, ax in enumerate(axs):
        first_corners = X[e, :2, :] # [2,2]
        sbs_obj.model.eval()
        next_corners = sbs_obj.model(X[e:e+1, :2].to(sbs_obj.device)).squeeze().detach().cpu().numpy() # predictions
        pred_corners = np.concatenate([first_corners, next_corners], axis=0) # [4,2]

        for j, corners in enumerate([X[e], pred_corners]):
            for i in range(4):
                coords = corners[i] # [2]
                color = 'k'
                ax.scatter(*coords, c=color, s=400)
                if i==3:
                    start = -1
                else:
                    start = i
                # if we are drawing the truth OR (when we are drawing the predictions AND corners is 1,2,3 
                if j==0 or (j and i):
                    ax.plot(*corners[[start, start+1]].T, c='k', lw=2, alpha=0.5, linestyle='--' if j else '-')
                ax.text(*(coords-np.array([0.04,0.04])), str(i+1), c='w', fontsize=12)
                if directions is not None:
                    ax.set_title(f'{"Counter-" if not directions[e] else ""}Clockwise')
        
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_xlim([-1.45, 1.45])
        ax.set_ylim([-1.45, 1.45])
    fig.tight_layout()
    return fig 

def make_line(ax, point):
    point = np.vstack([[0,0], np.array(point.squeeze().tolist())])
    line = ax.plot(*point.T, lw=0)[0]
    return line

# q: [H], ks: [L,H] with L=sequence length, H=hidden size
# result: one point
def query_and_keys(q, ks, result=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
    else:
        fig = ax.get_figure()
    norm_q = np.linalg.norm(q)
    line_q = make_line(ax, q)
    line_k = []
    norm_k = []
    cos_k = []
    for k in ks:
        line_k.append(make_line(ax, k))
        norm_k.append(np.linalg.norm(k))
        cos_k.append(np.dot(q, k)/(norm_k[-1]*norm_q))
    add_arrow(line_q, lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(line_k[0], lw=2, color='k', text=r'$||K_0'+f'||={norm_k[0]:.2f}$'+'\n'+r'$cos\theta_0='+f'{cos_k[0]:.2f}$', size=12, text_offset=(-0.33,0.1))
    add_arrow(line_k[1], lw=2, color='k', text=r'$||K_1'+f'||={norm_k[1]:.2f}$'+'\n'+r'$cos\theta_1='+f'{cos_k[1]:.2f}$', size=12, text_offset=(-0.63,-0.18))
    add_arrow(line_k[2], lw=2, color='k', text=r'$||K_2'+f'||={norm_k[2]:.2f}$'+'\n'+r'$cos\theta_2='+f'{cos_k[2]:.2f}$', size=12, text_offset=(.05, .58))

    if result is not None:
        add_arrow(make_line(ax, result), lw=2, color='g', text=f'Context Vector', size=12, text_offset=(-0.26, 0.1))

    circle1 = plt.Circle((0,0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_xticklabels([-1.0, 0, 1.0], fontsize=12)
    ax.set_yticks([-1.0, 0, 1.0])
    ax.set_yticklabels([-1.0, 0, 1.0], fontsize=12)

    ax.set_ylim([-1.02, 1.02])
    ax.set_xlim([-1.02, 1.02])

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'$Query\ and\ Keys$')
    fig.tight_layout()
    return fig

def plot_attention(model, inputs, point_labels=None, source_labels=None, target_labels=None, decoder=False, self_attn=False, n_cols=5, alphas_attr='alphas'):
    textcolors = ["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
    model.eval()
    device = list(model.parameters())[0].device.type
    predicted_seqs = model(inputs.to(device))
    for attr in alphas_attr.split('.'): # 'alphas'
        alphas = getattr(model, attr)
    if len(alphas.shape) < 4: # [N,L(target),L(source)]=[10,2,2]=>[1,10,2,2]
        alphas = alphas.unsqueeze(0)
    alphas = np.array(alphas.tolist())
    n_heads, n_points, target_len, input_len = alphas.shape

    if point_labels is None:
        point_labels = [f'Point #{i}' for i in range(n_points)]
    
    if source_labels is None:
        source_labels = [f'Input #{i}' for i in range(input_len)]
    
    if target_labels is None:
        target_labels = [f'Target #{i}' for i in range(target_len)]
    
    if n_heads == 1:
        n_rows = (n_points//n_cols) + int((n_points%n_cols) > 0)
    else:
        n_cols = n_heads 
        n_rows = n_points
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for i in range(n_points):
        for head in range(n_heads):
            data = alphas[head][i].squeeze() # [L(target), L(source)]
            if n_heads > 1:
                if n_points > 1:
                    ax = axs[i, head]
                else:
                    ax = axs[head]
            else:
                ax = axs.flat[i]
            im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.grid(False)
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_xticklabels(source_labels)
            if n_heads == 1:
                ax.set_title(point_labels[i], fontsize=14)
            else:
                if i==0:
                    ax.set_title(f'Attention Head #{head+1}', fontsize=14)
                if head==0:
                    ax.set_ylabel(point_labels[i], fontsize=14)

            ax.set_yticklabels([])
            if n_heads == 1:
                if not(i%n_cols): # i=0 or 5 i.e. column 0th
                    ax.set_yticklabels(target_labels)
            else:
                if head==0:
                    ax.set_yticklabels(target_labels) # attention #0 i.e. column 0th
            ax.set_xticks(np.arange(data.shape[1]+1)-0.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-0.5, minor=True)
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            threshold = im.norm(data.max())/2
            for ip in range(data.shape[0]):
                for jp in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[ip,jp]) > threshold)])
                    text = im.axes.text(jp, ip, valfmt(data[ip, jp], None), **kw)

    fig.subplots_adjust(wspace=0.8, hspace=1.0)
    fig.tight_layout()
    return fig 

def plot_text(x, y, text, ax, fontsize=24):
    ax.text(x, y, text, fontsize=fontsize)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-1,1])
    ax.set_xlim([-1,1])

# dims: number of rows to encode; tot: number of columns = length of sequences
# seqs=(4,5,7); tot=8
def encoding_degrees(dims, seqs, tot):
    fig, axs = plt.subplots(dims+1, tot+1, figsize=(2*(tot+1), 2*(dims+1)))
    axs = np.atleast_2d(axs)
    plot_text(-0.5, -0.5, 'Position', axs[0,0])
    for dim in range(dims):
        tmp = np.linspace(0, tot/seqs[dim], tot+1)*2*np.pi # how many % of a cycle
        xs, ys = np.cos(tmp).reshape(-1,1), np.sin(tmp).reshape(-1,1)
        for seq in range(tot): # 0->7
            plot_text(-0.1, -0.5, seq, axs[0, seq+1])
            if seq == 0:
                plot_text(-0.5, -0.2, f'Base {seqs[dim]}', axs[dim+1,0])
                plot_text(-0.5, -1.3, f'(sine, cosine)', axs[dim+1,0], fontsize=16)
            # plot_dial(xs, ys, seq=seq, dim=0, scale=f'1/{seqs[dim]}', ax=axs[dim+1, seq+1], has_coords=True)
        seqs *= 2
        print("AB:", seqs)
    fig.tight_layout()
    return fig

def figure9():
    english = ['the', 'European', 'economic', 'zone']
    french = ['la', 'zone', 'économique', 'européenne']
    source_labels = english 
    target_labels = french 
    data = np.array([
        [0.8, 0, 0, 0.2],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0.8, 0, 0.2]
        ])
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    # color the cells
    im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
    ax.grid(False)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(source_labels, rotation=90)
    ax.set_yticklabels(target_labels)
    # move x label from bottom to top
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    textcolors = ["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center") # set text alignment in each cell in axis
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}") # format the displayed text (number) in each cell in axis
    threshold = im.norm(data.max())/2
    for ip in range(data.shape[0]):
        for jp in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[ip,jp])) > threshold])
            text = im.axes.text(jp, ip, valfmt(data[ip,jp],None), **kw)

    fig.tight_layout()
    return fig 
