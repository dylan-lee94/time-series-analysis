# In this script we analyze the mutual information with respect to Currency and its twitter sentiment
# %% Imports
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Local Imports
sys.path.insert(0, '../')
from utility import load_currency
from mutual_information import mutual_information
# %% Choose the Cryptocurrency and the features to analyse
coin = 'DASH'
features = ['logclose','positive_score_en', 'negative_score_en', 'score_en', 'positive_volume_en', 'negative_volume_en', 'volume_ratio_en', 'total_volume_en']
df = load_currency(coin,columns=features)

# %% Calculate log return

df['logreturn'] = df.logclose.diff()
df = df.dropna()
logreturn = df.logreturn
df = df.drop(['logreturn','logclose'],axis=1)
# %% Calculate mutual information for all sentiment currency pairs

# Initialization Matricies
nr_col = df.shape[1]
MI_matrix = np.empty((1,nr_col))
NMI_matrix = np.empty((1,nr_col))

# MI for all pairs
for i in range(nr_col):
           MI,NMI = mutual_information(df.iloc[:,i],logreturn)
           MI_matrix[:,i] = MI
           NMI_matrix[:,i] = NMI
# %% Calculate the p-value by the permutation test:

# For reconstruction
np.random.seed(420)

# Number of runs of the permutation test
n_shuffle = 1000

# Initialization Matricies
MI_p_value = np.zeros((1,nr_col))
NMI_p_value = np.zeros((1,nr_col))

# Initialization Indicies for shuffling
indicies = list(range(df.shape[0]))

for _ in tqdm(range(n_shuffle)):

    # Shuffle the returns in terms of their point in time
    np.random.shuffle(indicies)
    df_shuffled = df.iloc[indicies,:]

    # Calculate the MI on the shuffled log-returns
    for i in range(nr_col):
                MI_rand,NMI_rand = mutual_information(df_shuffled.iloc[:,i],logreturn)

                # Lower triangle for p_value
                if MI_rand > MI_matrix[:,i]:
                    MI_p_value[:,i] += 1

                if NMI_rand > NMI_matrix[:,i]:
                    NMI_p_value[:,i] += 1


# Calculate the relative frequency
MI_p_value = MI_p_value/n_shuffle
NMI_p_value = NMI_p_value/n_shuffle
# %% Combine the results in one Matrix

# DataFrame
MI_matrix = pd.DataFrame(data=np.row_stack((MI_matrix,MI_p_value)),columns=df.columns,index=[coin,'p-values'])
NMI_matrix = pd.DataFrame(data=np.row_stack((NMI_matrix,NMI_p_value)),columns=df.columns,index=[coin,'p-values'])


# %%
print(f"Mututal Information with p-values in lower triangle:\n {MI_matrix}")

print(f"Normalized Mututal Information with p-values in lower triangle:\n {NMI_matrix}")
