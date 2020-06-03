#%% Imports
import numpy as np
from entropy import entropy, joint_entropy

# np.where() evalutes before condition
np.seterr(divide='ignore',invalid='ignore')

#%% Calculate the Mutual Information in terms of entropy

def mutual_information(x,y,nbins=20):
    Hx = entropy(x,nbins)
    Hy = entropy(y,nbins)
    Hxy = joint_entropy(x,y,nbins)
    MI = Hx+Hy-Hxy
    NMI = 2*MI/(Hx + Hy)
    return MI, NMI

#%% Calculate the Conditional Mutual Information in terms of entropy
# x independent variable
# y dependent variable (lagged)
# z is lagged x

def cond_mutual_information(x,y,z,nbins=20):
    Hxz = joint_entropy(x,z,nbins)
    Hyz = joint_entropy(y,z,nbins)
    Hz = entropy(z,nbins)

    # Mutual Information for 3 Variables
    count_xyz,edges = np.histogramdd(np.array([x,y,z]).T, bins=20)
    p_xyz = count_xyz/len(x)
    Hxyz = -np.sum(np.sum(np.sum(np.where(p_xyz>0, np.log2(p_xyz)*p_xyz,0))))
    
    # Conditional Mutual Information
    MIxy_z =  Hxz + Hyz - Hxyz - Hz
    NMIxy_z = 2*MIxy_z/(Hxz + Hyz)
    return MIxy_z, NMIxy_z
