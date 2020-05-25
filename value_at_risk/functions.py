#%%
import numpy as np
def VAR(r,alpha):
    return -np.quantile(r,alpha)

def CVAR(r,alpha):
    return -np.mean(r[r <= np.quantile(r,alpha)])
