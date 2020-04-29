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

returns = df['logclose'].pct_change().dropna()





#%% Define the Test Window and Parameters
tau = 20
alpha = 0.05

#%% Value at Risk - NonParametric

#Preallocating Arrays
VaR_H = []

for i in range(0,len(returns)-tau):
  #Parameters
  ts = returns[0:tau+i]

  #VaR
  VaR_H.append(ts.quantile(0.05)) 
  
  N = len(ts)
  ts_sorted = ts.sort_values()

  # #ES95
  # k = math.ceil(N*0.05)
  # CVaR_H.append(np.mean(ts_sorted[:k])) 


# Plotting VaR & CVaR
fig,ax = plt.subplots()
ax.plot(returns[tau:])
ax.plot(returns[tau:].index,VaR_H, label='VaR')
ax.set_xlabel('Date')
ax.set_title('VaR  Estimation Using the Historical Simulation Method')
ax.legend()

#%% Value at Risk - Normal Distribution

#Preallocating Arrays
VaR_N = []

for i in range(0,len(returns)-tau):
  #Parameters
  ts = returns[0:tau+i]
  mu_norm, sig_norm = norm.fit(ts)

  #VaR & CVaR
  VaR_N.append(mu_norm+sig_norm*norm.ppf(1-alpha)) 


# Plotting VaR & CVaR
fig, ax = plt.subplots()
ax.plot(returns[tau:])
ax.plot(returns[tau:].index,-1*np.array(VaR_N), label='VaR')
ax.set_xlabel('Date')
ax.set_title('VaR  Estimation using the Normal Distribution Method')
ax.legend()

#%% Value at Risk - Student's t-distribution

#Preallocating Arrays
VaR_T = []

for i in range(0,len(returns)-tau):
  #Parameters
  ts = returns[0:tau+i]
  # mu_norm = np.mean(ts)
  # sig_norm = np.std(ts)
  mu_norm, sig_norm = norm.fit(ts)
  nu, mu_t, sig_t = t.fit(ts)

  sig_t = ((nu-2)/nu)**0.5 * sig_norm

  #VaR &CVaR
  VaR_T.append(mu_norm + sig_t* t.ppf(1-0.05,nu))
  # CVaR_N.append(mu+std*norm.pdf(norm.ppf(0.05))/0.05)


# Plotting VaR & CVaR
fig, ax = plt.subplots()
ax.plot(returns[tau:])
ax.plot(returns[tau:].index,-1*np.array(VaR_T), label='VaR')
ax.set_xlabel('Date')
ax.set_title('VaR Estimation using the Normal Distribution Method')
ax.legend()


#%% VAR & ES  - Parametric (EWMA) 
# Initiate the EWMA using a warm-up phase to set up the standard deviation.

tau=20

Lambda = 0.94
Sigma2     = [returns[0]**2]


for j in range(1,tau):
  Sigma2.append((1-Lambda) * returns[j-1]**2 + Lambda * Sigma2[j-1])


#Preallocate
VaR_EWM = []
CVaR_EWM = []

for j in range(tau,len(returns)):    
    Sigma2.append(Lambda* Sigma2[j-1]+(1-Lambda)*returns[j-1]**2)
    std = np.sqrt(Sigma2[j])
    
    #Parametrs
    # mu = returns[:t].ewm(alpha=0.94).mean()[-1]

    #VaR &CVaR
    VaR_EWM.append(std*norm.ppf(1-alpha)) 
    # CVaR_EWM.append(std*norm.pdf(norm.ppf(0.05))/0.05)

# Plotting VaR & CVaR
fig, ax = plt.subplots()
ax.plot(returns[tau:])
ax.plot(returns[tau:].index,-1*np.array(VaR_EWM), label='VaR')
ax.set_xlabel('Date')
ax.set_title('VaR & CVaR Estimation using the EWM Method')
ax.legend()


#%% Literature

# https://de.mathworks.com/help/risk/value-at-risk-estimation-and-backtesting-1.html
# https://mmquant.net/introduction-to-volatility-models-with-matlab-sma-ewma-cc-range-estimators/
# https://uk.mathworks.com/help/risk/overview-of-var-backtesting.html
# https://www.investopedia.com/articles/professionals/081215/backtesting-valueatrisk-var-basics.asp
