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
                    pass
    fig.tight_layout()
    return fig 
