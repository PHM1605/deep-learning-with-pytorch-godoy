import matplotlib.pyplot as plt 
import numpy as np 

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

def query_and_keys(q, ks, result=None, ax=None):
    pass