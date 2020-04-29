import numpy as np
import math as m

def histnorm(x,n_bins=20):
    count, bin_edges = np.histogram(x,n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])*0.5
    widths = np.diff(bin_edges)
    f = count/(count*widths).sum()
    return (f,bin_centers)



def VAR(ts,alpha=0.05):
    return -np.quantile(ts,alpha)

def CVAR(ts,alpha=0.05):
    k = m.ceil(len(ts)*alpha) # Ceil due to Zero-based numbering
    return -np.mean(np.sort(ts)[:k]) 
