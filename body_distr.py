
#%%
import os
import pandas as pd
import numpy as np
from scipy.stats import norm,t 
import math


import matplotlib.pyplot as plt

path = os.getcwd()
df = pd.read_csv(path+'/Data/dataset/DASH_merged.txt', sep = ',')
df.drop(labels='Unnamed: 0',axis=1,inplace=True)
df.time = pd.to_datetime(df.time,unit='s')
df.set_index('time',inplace=True)
df.head()


#%% Log Returns
r = df['close'].pct_change().dropna()
r = r.resample('D').sum()

fig, ax = plt.subplots(2)

ax[0].plot(r)
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Log-Returns')
ax[0].set_title('Log-Returns over Time Horizon')

ax[1].plot(df['close'])
ax[1].set_ylabel('Closing Price')
ax[1].set_title('Closing Price over Time Horizon')

#%% Compute and print the values of the first four moments

print(f'Number of Observations = {r.count()}')
print(f'Mean = {r.mean():.4f}')
print(f'Std. deviation = {r.std():.4f}')
print(f'Skewness = {r.skew():.4f}')
print(f'Excess kurtosis = {(r.kurt()-3):.4f}')


#%% Fit Distribution to the Data

# Define point grid between min and max return
nbins = 100
x = np.linspace(np.min(r),np.max(r),nbins) 

# MLE of Normal Distribuion
mean, std = norm.fit(r)
log_likelihood_norm = norm.logpdf(r,mean,std).sum()

# MLE of Student-t distribution
df, loc, scale  = t.fit(r)
log_likelihood_student = t.logpdf(r, df,loc,scale).sum()

print('Loglikelihood: ')
print(f'Normal Distribution = {log_likelihood_norm}')
print(f'Students t-Distribution = {log_likelihood_student}')


#%% Plot of empirical PDF vs Theoretical 
fig, ax = plt.subplots(2)
fig.suptitle('Normal Distribution fit')

ax[0].hist(r,nbins,density=True,label='Data')
ax[0].plot(x,norm.pdf(x,loc = mean, scale = std),'r',label ='Normal PDF')
ax[0].set_ylabel('Probability Density')
ax[0].set_xlabel('Hourly Log-Returns')
ax[0].legend()

ax[1].hist(r,nbins,density=True,label='Data')
ax[1].plot(x,norm.pdf(x,loc = mean, scale = std),'r',label ='Normal PDF')
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Hourly Log-Returns')
ax[1].set_yscale('log')
ax[1].legend()



fig, ax = plt.subplots(2)
fig.suptitle('Students t-distribution fit')

ax[0].hist(r,nbins,density=True,label='Data')
ax[0].plot(x,t.pdf(x,loc = loc, scale = scale,df=df),'r',label ='Student')
ax[0].set_ylabel('Probability Density')
ax[0].set_xlabel('Hourly Log-Returns')
ax[0].legend()


ax[1].hist(r,nbins,density=True,label='Data')
ax[1].plot(x,t.pdf(x,loc = loc, scale = scale,df=df),'r',label ='Student')
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Hourly Log-Returns')
ax[1].set_yscale('log')
ax[1].legend()

#%% Plot of empirical CCDF vs Gaussian
import statsmodels.api as sm 

fig,ax = plt.subplots()

fig,ax = plt.subplots(1,2)
sm.qqplot(r, dist='norm',fit=True, line ='s',ax=ax[0]) 
sm.qqplot(r, dist='t',fit=True, line ='s',ax=ax[1])
ax[0].set_title('QQ Plot Normal Distribution')
ax[1].set_title('QQ Plot Students t-Distribution')
plt.show()
