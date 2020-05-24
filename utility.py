#%% Imports
import os 
import pandas as pd
import numpy as np
#%% Load a singe Cryptocurrency

def load_currency(name,columns):
    path = "../data/dataset"
    df = pd.read_csv(os.path.join(path,name+'_merged.txt'), sep = ',', header=0)
    df.time = pd.to_datetime(df.time,unit='s')
    df = df.set_index('time')
    df = df[columns]
    return df

# # Example
# columns = ['close','logclose']
# df = load_currency(name = "BTC",columns=columns)



#%% Load several Cryptocurrencies

def load_all_currencies(names,columns):
    path = "../data/dataset"

    li = []

    for curr in names:
        df = pd.read_csv(os.path.join(path,curr+'_merged.txt'), sep = ',', header=0)
        df.time = pd.to_datetime(df.time,unit='s')
        df = df.set_index('time')
        df = df[columns]
        li.append(df)    


    frame = pd.concat(li,axis=1,keys=names,join='outer')

    # Pick common time window of all currencies
    frame = frame.dropna(axis=0)
    return frame

# # Example
# names = ["DASH", "ETC", 'ETH']
# columns = ['close','logclose']

# df = load_all_currencies(names,columns)


#%%
def histnorm(x,n_bins=20):
    count, bin_edges = np.histogram(x,n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])*0.5
    widths = np.diff(bin_edges)
    f = count/(count*widths).sum()
    return (f,bin_centers)