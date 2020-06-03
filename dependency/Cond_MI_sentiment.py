# In this script we analyze the conditional mutual information with respect to Currency and its twitter sentiment
# %% Imports
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Local Imports
sys.path.insert(0, '../')
from utility import load_currency
from mutual_information import cond_mutual_information


# %% Choose the Cryptocurrency and the features to analyse
coin = 'DASH'
features = ['logclose','positive_score_en', 'negative_score_en', 'score_en', 'positive_volume_en', 'negative_volume_en', 'volume_ratio_en', 'total_volume_en']
df = load_currency(coin,columns=features)

# %% Calculate log return

df['logreturn'] = df.logclose.diff()
df = df.dropna()
logreturn = df.logreturn
df = df.drop(['logreturn','logclose'],axis=1)
# %% Calculate conditional mutual information for all sentiment currency pairs

# CMI is very sensitive to number of bins
nbins = 40 
lag = 10

# Initialization Matricies
nr_col = df.shape[1]
CMI_matrix = np.empty((1,nr_col))
CNMI_matrix = np.empty((1,nr_col))

# CMI for all pairs
for i in range(nr_col):
           CMI,CNMI = cond_mutual_information(x=df.iloc[lag:,i],y=logreturn.iloc[:-lag], z=df.iloc[:-lag,i],nbins=nbins)
           CMI_matrix[0,i] = CMI
           CNMI_matrix[0,i] = CNMI


# %% Calculate the p-value by the permutation test:

# For reconstruction
np.random.seed(420)

# Number of runs of the permutation test
n_shuffle = 1000

# Initialization Matricies
CMI_p_value = np.zeros((1,nr_col))
CNMI_p_value = np.zeros((1,nr_col))

# Initialization Indicies for shuffling
indicies = list(range(df.shape[0]))

for _ in tqdm(range(n_shuffle)):

    # Shuffle the returns in terms of their point in time
    np.random.shuffle(indicies)
    df_shuffled = df.iloc[indicies,:]

    # Calculate the CMI on the shuffled log-returns
    for i in range(nr_col):
                CMI_rand,CNMI_rand = cond_mutual_information(x=df_shuffled.iloc[lag:,i],y=logreturn.iloc[:-lag], z=df_shuffled.iloc[:-lag,i],nbins=nbins)

                # p_value values
                if CMI_rand > CMI_matrix[0,i]:
                    CMI_p_value[0,i] += 1

                if CNMI_rand > CNMI_matrix[0,i]:
                    CNMI_p_value[0,i] += 1


# Calculate the relative frequency
CMI_p_value = CMI_p_value/n_shuffle
CNMI_p_value = CNMI_p_value/n_shuffle
# %% Combine the results in one DataFrame

CMI_matrix = pd.DataFrame(data=np.row_stack((CMI_matrix,CMI_p_value)),columns=df.columns,index=[coin,'p-values'])
CNMI_matrix = pd.DataFrame(data=np.row_stack((CNMI_matrix,CNMI_p_value)),columns=df.columns,index=[coin,'p-values'])

# %%
print(f"Conditional Mututal Information with p-values:\n {CMI_matrix}")

print(f"Normalized Conditional Mututal Information with p-values:\n {CNMI_matrix}")

