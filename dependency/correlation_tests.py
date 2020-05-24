#%% Imports
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr
import sys
from tqdm import tqdm


# Local Imports
sys.path.insert(0, '../')
from utility import load_all_currencies

#%% Choose the Cryptocurrencies and the features to analyse
coins = ['DASH', 'ETC', 'ETH', 'XMR', 'BCN', 'ICX']
feature = ['logclose']
df = load_all_currencies(names=coins,columns=feature)

# Calculate log return
tmp = [df[item]['logclose'].diff().dropna() for item in coins]
df = pd.concat(tmp,axis=1,keys=coins)

# For Reproducibility
np.random.seed(420)

#%% Permutation Test

def permutation_test(df, item = 'pearson'):
    corr_matrix = df.corr(method=item)

    Nsamples = 1000

    nr_col = df.shape[1]
    nr_row = df.shape[0]

    pval = np.zeros((nr_col,nr_col))
    corr_dict = {'pearson': pearsonr, 'kendall': kendalltau, 'spearman': spearmanr}

    for i in tqdm(range(nr_col)):
        for j in range(i+1,nr_col):
            count = 0
            for _ in range(Nsamples):

                # Computing correlation on randomly reshuffled returns
                idx = list(range(nr_row))
                np.random.shuffle(idx)

                tmp = corr_dict[item](df.iloc[idx,i],df.iloc[:,j])[0]
                
                # Two Tailed Test (absolut)
                if abs(tmp) > abs(corr_matrix.iloc[i,j]):
                    count += 1 

            # Calculation of pvalue correlation
            pval[i,j] = count/Nsamples
    return pd.DataFrame(pval,index = df.columns, columns=df.columns)

pval = permutation_test(df)

print(pval)


#%% Bootstrapping

def bootstrap_test(df, alpha = 0.05, item = 'pearson'):
    Nsamples = 1000
    nr_col = df.shape[1]
    nr_row = df.shape[0]

    CI = np.zeros((nr_col,nr_col))
    corr_dict = {'pearson': pearsonr, 'kendall': kendalltau, 'spearman': spearmanr}

    for i in tqdm(range(nr_col)):
        for j in range(i+1,nr_col):
            tmp = []

            for _ in range(Nsamples):
                # Computing correlation on bootstrapped returns
                idx = np.random.randint(0,nr_row,nr_row)
                tmp.append(corr_dict[item](df.iloc[idx,i],df.iloc[idx,j])[0])
            
            CI[i,j]= np.quantile(tmp,1-alpha/2)
            CI[j,i]= np.quantile(tmp,alpha/2)
    
    return pd.DataFrame(CI,index = df.columns, columns=df.columns)

CI = bootstrap_test(df)

print(CI)