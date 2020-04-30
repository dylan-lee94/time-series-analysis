import numpy as np
import math as m
import os
import pandas as pd

def histnorm(x,n_bins=20):
    count, bin_edges = np.histogram(x,n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])*0.5
    widths = np.diff(bin_edges)
    f = count/(count*widths).sum()
    return (f,bin_centers)

# Parametric VaR and CVaR
def VAR(ts,alpha=0.05):
    return -np.quantile(ts,alpha)

def CVAR(ts,alpha=0.05):
    k = m.ceil(len(ts)*alpha) # Ceil due to Zero-based numbering
    return -np.mean(np.sort(ts)[:k]) 

# Load Data
def load_data(name):
    os.chdir("../data")
    path = os.path.join(os.getcwd(),"dataset",'DASH_merged.txt')
    df = pd.read_csv(path, sep = ',')
    df.drop(labels='Unnamed: 0',axis=1,inplace=True)
    df.time = pd.to_datetime(df.time,unit='s')
    df.set_index('time',inplace=True)
    return df

