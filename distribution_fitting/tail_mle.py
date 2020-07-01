import numpy as np

def power_law_fit(x, tail='right', q=0.1):
    """Function to fit the Power Law Distribution to the right or left tail of returns.

    Args:
        x: Returns.
        q (float):  Define tail as top or bottom q% of returns.
        tail (str): The tail to compute 'right' or 'left'
    Returns:
        alpha (float): The tail exponent.
        x: The tail values.
    """

    # Sort Data
    x = np.sort(x)

    if tail == 'right':
        # Select top q%  
        x = x[round((1-q)*len(x)):] 
    elif tail== 'left':
        # Select bottom q%
        x = x[:round(q*len(x))]

        # Convert negative returns to positive numbers
        x = abs(x)

    # MLE for tail exponent
    N = len(x)
    alpha = N/np.sum(np.log(x/min(x)))     
    return alpha, x 

def bootstrap_tail(x, q=0.1,Nbts=10000,bts=0.8, p=0.9,tail='right'):
    """Function to bootstrap the tail exponent for the right or left tail of returns.

    Args:
        x: Returns.
        Nbts (int):  Number of bootstrap samples.
        bts (float): Fraction of data to be retained in each bootstrap sample.
        p (float): Significance level.
        tail (str): The tail to compute 'right' or 'left'
        q (float): Define tail as top or bottom q% of returns.
    Returns:
        The return value. True for success, False otherwise.

    """

    # Collect bootstrap estimates for tail exponent
    alpha_bts = [] 

    for _ in range(Nbts):

        # Random permutation of returns
        x_bts = np.random.permutation(x) 

        # Bootstrapping bts% of returns 
        x_bts = x_bts[0:round(bts*len(x_bts))] 

        # Computing alpha
        result = power_law_fit(x_bts,tail,q)

        alpha_bts.append(result[0])
    CI_left = np.quantile(alpha_bts,(1-p)*0.5)
    CI_right =  np.quantile(alpha_bts, (1+p)*0.5)
    return CI_left, CI_right, alpha_bts
