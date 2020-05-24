#%% Imports
import numpy as np
from entropy import entropy, joint_entropy

# np.where() evalutes before condition
np.seterr(divide='ignore',invalid='ignore')

#%% Calculate the mutual information in terms of entropy

def mutual_information(x,y,nbins=20):
    Hx = entropy(x,nbins)
    Hy = entropy(y,nbins)
    Hxy = joint_entropy(x,y,nbins)
    MI = Hx+Hy-Hxy
    NMI = 2*MI/(Hx + Hy)
    return MI, NMI

# Example: 
# MI,NMI = mutual_information(x,y)
