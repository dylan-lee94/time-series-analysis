
#%%
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from utility import histnorm


# Load Data
path = os.getcwd()
df = pd.read_csv(path+'/Data/dataset/DASH_merged.txt', sep = ',')
df.drop(labels='Unnamed: 0',axis=1,inplace=True)
df.time = pd.to_datetime(df.time,unit='s')
df.set_index('time',inplace=True)

# Log Returns
r = df['logclose'].pct_change().dropna()
# r = r.resample('D').sum()

#%%
from scipy.special import erf
from scipy.stats import norm

# Fit normal Distribution
mean, std = norm.fit(r)

# Define point grid between min and max return
nbins = 100
x = np.linspace(np.min(r),np.max(r),nbins)  

# Rank frequency plot
r_sorted = r.sort_values(ascending=True)

r_sorted
#%%
y = np.array(range(len(r_sorted)))

N = len(r_sorted)
# Computing empirical CCDF
y = 1 - y/(N+1) 
#%%
# Computing Gaussian CCDF
c = 0.5*(1 - erf((r_sorted-mean)/(std*np.sqrt(2))))

log_loss = -r
loss_sorted = log_loss.sort_values()
y1 = np.array(range(len(loss_sorted)))


# Computing empirical CCDF
y1 = 1 - y1/(N+1) 

# Computing Gaussian CCDF
c1 = 0.5*(1 - erf((loss_sorted-mean)/(std*np.sqrt(2))))

fig,ax = plt.subplots()

ax.semilogy(r_sorted,c,label='Normal')
ax.semilogy(r_sorted,y,'g+', label='positive returns')
ax.semilogy(loss_sorted,y1,'r+', label='negative returns')
ax.set_title('complementary cumulative log return distribution')
ax.set_xlabel('log return')
ax.set_ylabel('complementary cumulative distribution')




#%% 
def power_law_fit(x, m='right', p=0.1):
    # p Defining tails as top p% of returns (both positive and negative)

    # Sorting Data
    x = np.sort(x)

    if m == 'right':
        # Selecting top p%  
        tail = x[round((1-p)*len(x)):] 
    else:
        # Selecting bottom p%
        tail = x[:round(p*len(x))]

        # Converting negative returns to positive numbers
        tail = abs(tail)

    # Number of returns selected as tail    
    N = len(tail)

    # MLE for tail exponent
    alpha = N/np.sum(np.log(tail/min(tail)))     
    return alpha, tail 
    
def plot_power_law(alpha,tail):

    # Define Grid
    x = np.linspace(min(tail),max(tail),100)

    # Power law distribution
    power_law = ((alpha-1)/min(tail))*(x/min(tail))**(-alpha)


    fig,ax = plt.subplots()
    f,bin_centers = histnorm(tail,n_bins= 20)
    ax.loglog(bin_centers,f,'bo',label="Data")


    ax.loglog(x,power_law,'r',label='Power Law fit')
    # ax.set_xlabel('Top 10% of hourly log-returns')
    # ax.set_xlabel('Bottom 10% of hourly log-returns')
    ax.set_ylabel('Probabiliy Density')
    ax.legend()

#%% Fitting the tails of the distribution

# Fitting the Right Tail of the distribution
alpha_right,right_tail = power_law_fit(r,'right')
print(f'Right tail exponent: {alpha_right}')
plot_power_law(alpha_right,right_tail)


# Fitting the Left
alpha_left,left_tail = power_law_fit(r,'left')
print(f'Left tail exponent: {alpha_left}')
plot_power_law(alpha_left,left_tail)

#%% Bootstrap tail


def bootstrap_tail(x, Nbts=2000,bts=0.8, p=0.9,pick='right'):
    # Nbts: Number of bootstrap samples
    # bts: Fraction of data to be retained in each bootstrap sample
    # p: Significance level


    # Collect bootstrap estimates for tail exponent
    alpha_bts = [] 

    #can push Nbts to a few million 
    for i in range(Nbts):

        # Random permutation of returns
        x_bts = np.random.permutation(x) 

        # Bootstrapping bts% of returns 
        x_bts = x_bts[1:round(bts*len(x_bts))] 

        # Computing alpha
        alpha,tail = power_law_fit(x_bts,pick)

        alpha_bts.append(alpha)
    return np.quantile(alpha_bts,(1-p)*0.5), np.quantile(alpha_bts, (1+p)*0.5),alpha_bts

CI_left, CI_right,alpha_bts = bootstrap_tail(r, Nbts=10000,bts=0.8, p=0.9,pick='right')

print(f'Right tail interval at CL: [{CI_left:.2f};{CI_right:.2f}]')

CI_left, CI_right,alpha_bts = bootstrap_tail(r, Nbts=10000,bts=0.8, p=0.9,pick='left')

print(f'Left tail interval at CL: [{CI_left:.2f};{CI_right:.2f}]')

#%%
from scipy.stats import norm
x = np.linspace(min(alpha_bts),max(alpha_bts),200)
mu,sigma = norm.fit(alpha_bts)

fig,ax = plt.subplots()
ax.hist(alpha_bts,100,density=True,label='Data')
ax.plot(x,norm.pdf(x,loc = mu, scale = sigma),'r',label ='Normal PDF')
ax.legend()
# f,bin_centers = histnorm(alpha_bts,n_bins=100)
# ax.bar(bin_centers,f,label="Data")



#%%


#Generalized Pareto Distribution
# paramEsts = gpfit(right_tail); % 95% confidence intervals for the parameter estimates.
# kHat      = paramEsts(1);   % Tail index parameter
# sigmaHat  = paramEsts(2);  % Scale parameter

# %GeneralizedExtremeValue
# ExtremeValue = fitdist(right_tail,'GeneralizedExtremeValue');

# %Exponential Distributions
# Exponential = fitdist(right_tail,'Exponential');
% 
# y1 = gppdf(x,kHat,sigmaHat); %Generalized Pareto
# y2 = pdf(ExtremeValue,x); %Generalized Extreme Value 
# y3 = pdf(Exponential,x); %Exponential Distribution

% figure(7)
% loglog(x,h,'ob','MarkerSize',8,'MarkerFaceColor','b')
% hold on
% loglog(x,y1,'r','LineWidth',2)
% set(gca,'FontSize',20)
% title('Right tail Pareto')
% 
% figure(8)
% loglog(x,h,'ob','MarkerSize',8,'MarkerFaceColor','b')
% hold on
% loglog(x,y2,'r','LineWidth',2)
% set(gca,'FontSize',20)
% title('Right tail Extreme Value')
% 
% figure(9)
% loglog(x,h,'ob','MarkerSize',8,'MarkerFaceColor','b')
% hold on
% loglog(x,y3,'r','LineWidth',2)
% set(gca,'FontSize',20)
% title('Right tail Exponential')

%https://www.mathworks.com/help/stats/generalized-pareto-distribution.html


%https://en.wikipedia.org/wiki/Heavy-tailed_distribution
%https://en.wikipedia.org/wiki/Power_law

%% Testing
%One-sample Kolmogorov-Smirnov test

[KST_N,p1] = kstest(r,'CDF',PD_n,'Alpha',0.01)%Testing the Normal Distribution
[KST_T,p2] = kstest(r,'CDF',PD_t,'Alpha',0.01)%Testing the t-student


%[h,p,ksstat,cv] = kstest(r,'CDF',PD_t,'Alpha',0.01) %detailed test of t-student

%Anderson-Darling test
[ADT_N,p] = adtest(r,'Distribution','norm','Alpha',0.01) 



%https://uk.mathworks.com/help/stats/kstest.html
%https://uk.mathworks.com/help/stats/adtest.html




%% Stationarity
[h,pValue,stat,cValue,reg] = kpsstest(y,Name,Value)
%% 
