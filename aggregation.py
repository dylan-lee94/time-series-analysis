#%%
import sys
import numpy as np 

# sys.path.insert(0,'../')
from utility import load_currency

df = load_currency('DASH', columns=['close','open','high','low'])

#%% Resample prices

# Set the period
k = 24
ts = df['close']

# Generic resampling function for prices
def resample_prices(ts,k):
    if k <= 1:
        print('Please fill in an integer larger than 1')

    ts = np.asarray(ts)
    return [ts[i] for i in range(0,len(ts),k)]

# Pandas resampling
logic = {'open' : 'first',
         'close' : 'last',
         'high'  : 'max',
         'low'   : 'min',
         }

df.resample(rule='24H').apply(logic)

#%% Resample returns

# Set the period
k = 24
returns = df.close.pct_change().dropna()

# Generic resampling function for returns
def resample_returns(ts,k):
    ts = np.asarray(ts)
    r = (ts+1)
    return [np.prod(r[i:i+k])-1 for i in range(0,len(r)-k,k)]



# Pandas resampling
daily_r = (returns+1).resample(rule='24H').prod()-1


#%% Resample Log Returns

# Set the period
k = 24
log_returns = np.log(df.close).diff().dropna()

# Generic resampling function for log returns
def resample_logreturns(ts,k):
    ts = np.asarray(ts)
    return [np.sum(ts[i:i+k]) for i in range(0,len(ts)-k,k)]

# Pandas resampling
daily_logr = log_returns.resample(rule='24H').sum()

#%% Difference between aggregated and overlapping returns

# Set the period
ts = df.close.values
k = 24

# k-period log return from prices
k_return = [np.log(ts[k+i]/ts[i]) for i in range(0,len(ts)-k,k)]
print(len(k_return))

# k-period overlapping log return from prices
overlap_return = np.log(ts[k:]/ts[:-k])
print(len(overlap_return))
