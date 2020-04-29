#%%
import os
import pandas as pd
import numpy as np
from scipy.stats import norm,t 
import math
import matplotlib.pyplot as plt

from utility import VAR,CVAR


path = os.getcwd()
df = pd.read_csv(path+'/Data/dataset/DASH_merged.txt', sep = ',')
df.drop(labels='Unnamed: 0',axis=1,inplace=True)
df.time = pd.to_datetime(df.time,unit='s')
df.set_index('time',inplace=True)
df.head()

returns = df['close'].pct_change().dropna()




#%% VaR Plotting

# Significance Level
alpha = 0.01  

# Defining Grid
nbins = 100
minv = np.min(returns)
maxv = np.max(returns)
x = np.linspace(minv,maxv,nbins)



# Formatting
fig, ax = plt.subplots()

ax.set(xlim=[minv,maxv], xlabel='Daily Log Return' )
ax.legend()

# Fit Empirical Distribution and compute non-parametric VaR & CVaR
grey = .77, .77, .77
ax.hist(returns,nbins,density=True,label='Sample Data',color=grey)
VaR_Historical = VAR(returns,alpha)
CVaR_Historical = CVAR(returns,alpha)

# Fit Normal Distribution and compute parametric VaR & CVaR
mu, sigma = norm.fit(returns)
VaR_Normal = mu + sigma*norm.ppf(1-alpha)
CVaR_Normal = mu+sigma*norm.pdf(norm.ppf(0.05))/0.05

ax.plot(x,norm.pdf(x,loc = mu, scale = sigma),'r',label ='Normal')
ax.axvline(-VaR_Normal,color='r', linestyle='--',label='Normal VaR')


# Fit Student's t-Distribution and compute VaR & CVaR
df, loc, scale  = t.fit(returns)
sigma_t = np.sqrt((df-2)/df) * sigma
VaR_Student = mu + sigma_t * t.ppf(1-alpha,df)
xanu = t.ppf(alpha, df) 
CVaR_Student = mu + sigma * -1/alpha * (1-df)**(-1) * (df-2+xanu**2) * t.pdf(xanu, df)

ax.plot(x,t.pdf(x,loc = loc, scale = scale,df=df),'b',label ='Student')
ax.axvline(-VaR_Student, color='b', linestyle='--', label="Student's t VaR")

# 

plt.tight_layout()

print(f"Historical VaR    = {VaR_Historical:.4f}")
print(f"Historical CVaR    = {CVaR_Historical:.4f}")
print(f"Student's t-VaR = {VaR_Student:.4f}")
print(f"Student's t-CVaR = {CVaR_Student:.4f}")
print(f"Normal VaR    = {VaR_Normal:.4f}")
print(f"Normal CVaR    = {CVaR_Normal:.4f}")


# http://www.quantatrisk.com/2015/12/02/student-t-distributed-linear-value-at-risk/
# http://www.quantatrisk.com/2016/12/08/conditional-value-at-risk-normal-student-t-var-model-python/

