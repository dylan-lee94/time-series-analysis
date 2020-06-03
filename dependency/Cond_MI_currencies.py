# In this script we analyze the mutual information with respect to Currency pairs
# %% Imports
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Local Imports
sys.path.insert(0, '../')
from utility import load_all_currencies,load_currency
from mutual_information import cond_mutual_information


#%% Choose the Cryptocurrencies and the features to analyse
coins = ['DASH', 'ETC', 'ETH', 'XMR', 'BCN', 'ICX']
feature = ['logclose']
df = load_all_currencies(names=coins,columns=feature)

#%% Calculate log return

tmp = [df[item]['logclose'].diff().dropna() for item in coins]
df = pd.concat(tmp,axis=1,keys=coins)

# Resample log returns
# df = df.resample("D").sum()


#%% Calculate conditional mutual information for all currency pairs

# CMI is very sensitive to number of bins
nbins = 40 
lag = 10

# Initialization Matricies
nr_col = df.shape[1]
CMI_matrix = np.zeros((nr_col,nr_col))
CNMI_matrix = np.zeros((nr_col,nr_col))

# CMI for all pairs
for i in range(nr_col):
   for j in range(nr_col):
           CMI,CNMI = cond_mutual_information(x=df.iloc[lag:,i],y=df.iloc[:-lag,j],z=df.iloc[:-lag,i],nbins=nbins)
           CMI_matrix[i,j] = CMI
           CNMI_matrix[i,j] = CNMI

#%% Calculate the p-value by the permutation test:

# For reconstruction
np.random.seed(420)

# Number of runs of the permutation test
n_shuffle = 1000

# Initialization Matricies
CMI_p_value = np.zeros((nr_col,nr_col))
CNMI_p_value = np.zeros((nr_col,nr_col))

# Initialization Indicies for shuffling
indicies = list(range(df.shape[0]))

for _ in tqdm(range(n_shuffle)):

    # Shuffle the returns in terms of their point in time
    np.random.shuffle(indicies)
    df_shuffled = df.iloc[indicies,:]

    # Calculate the MI on the shuffled log-returns
    for i in range(nr_col):
        for j in range(nr_col):
                CMI_rand,CNMI_rand = cond_mutual_information(x=df_shuffled.iloc[lag:,i],y=df_shuffled.iloc[:-lag,j],z=df_shuffled.iloc[:-lag,i],nbins=nbins)
                if CMI_rand > CMI_matrix[i,j]:
                    CMI_p_value[i,j] += 1
                    CNMI_p_value[i,j] += 1


# Calculate the relative frequency
CMI_p_value = CMI_p_value/n_shuffle
CNMI_p_value = CNMI_p_value/n_shuffle
                    
#%%

# Convert CMI to DataFrame
CMI_matrix = pd.DataFrame(CMI_matrix,index=df.columns,columns=df.columns)
CNMI_matrix = pd.DataFrame(CNMI_matrix,index=df.columns,columns=df.columns)

# Convert p-values to DataFrame
CMI_p_value = pd.DataFrame(CMI_p_value,index=df.columns,columns=df.columns)
CNMI_p_value = pd.DataFrame(CNMI_p_value,index=df.columns,columns=df.columns)


# %%

print(f"Conditional Mututal Information:\n {CMI_matrix}")
print(f"p-values for CMI:\n {CMI_p_value}")


print(f"Normalized Conditional Mututal Information:\n {CNMI_matrix}")
print(f"p-values for NCMI:\n {CNMI_p_value}")

