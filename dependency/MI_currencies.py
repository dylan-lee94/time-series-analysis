# In this script we analyze the mutual information with respect to Currency Pairs
# %% Imports
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Local Imports
sys.path.insert(0, '../')
from utility import load_all_currencies
from mutual_information import mutual_information

#%% Choose the Cryptocurrencies and the features to analyse
coins = ['DASH', 'ETC', 'ETH', 'XMR', 'BCN', 'ICX']
feature = ['logclose']
df = load_all_currencies(names=coins,columns=feature)

#%% Calculate log return

tmp = [df[item]['logclose'].diff().dropna() for item in coins]
df = pd.concat(tmp,axis=1,keys=coins)

# Resample log returns
# df = df.resample("D").sum()


#%% Calculate mutual information for all currency pairs

# Initialization Matricies
nr_col = df.shape[1]
MI_matrix = np.zeros((nr_col,nr_col))
NMI_matrix = np.zeros((nr_col,nr_col))

# MI for all pairs
for i in range(nr_col):
   for j in range(nr_col):
       if(j>i): 
           MI,NMI = mutual_information(df.iloc[:,i],df.iloc[:,j])
           MI_matrix[i,j] = MI
           NMI_matrix[i,j] = NMI

#%% Calculate the p-value by the permutation test:

# For reconstruction
np.random.seed(420)

# Number of runs of the permutation test
n_shuffle = 1000

# Initialization Matricies
MI_p_value = np.zeros((nr_col,nr_col))
NMI_p_value = np.zeros((nr_col,nr_col))

# Initialization Indicies for shuffling
indicies = list(range(df.shape[0]))

for _ in tqdm(range(n_shuffle)):

    # Shuffle the returns in terms of their point in time
    np.random.shuffle(indicies)
    df_shuffled = df.iloc[indicies,:]

    # Calculate the MI on the shuffled log-returns
    for i in range(nr_col):
        for j in range(nr_col):
            if (j>i):
                MI_rand,NMI_rand = mutual_information(df_shuffled.iloc[:,i],df_shuffled.iloc[:,j])

                # Lower triangle for p_value
                if MI_rand > MI_matrix[i,j]:
                    MI_p_value[j,i] += 1

                if NMI_rand > NMI_matrix[i,j]:
                    NMI_p_value[j,i] += 1

# Calculate the relative frequency
MI_p_value = MI_p_value/n_shuffle
NMI_p_value = NMI_p_value/n_shuffle

# %% Combine the results in one Matrix
MI_combined = np.triu(MI_matrix) + np.tril(MI_p_value)
NMI_combined = np.triu(NMI_matrix) + np.tril(NMI_p_value)

# DataFrame
MI_matrix = pd.DataFrame(MI_combined,index=df.columns,columns=df.columns)
NMI_matrix = pd.DataFrame(NMI_combined,index=df.columns,columns=df.columns)

# %%

print(f"Mututal Information with p-values in lower triangle:\n {MI_matrix}")

print(f"Normalized Mututal Information with p-values in lower triangle:\n {NMI_matrix}")
