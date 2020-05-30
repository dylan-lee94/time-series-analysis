#%% Variance-Ratio Test
import numpy as np
from collections.abc import Iterable 
from scipy.stats import norm
from typing import List, Optional, Union



def variance_ratio(ts: np.ndarray, lags: Union[int,List[int]]=2, alpha: float=0.05, debiased: Optional[bool]=True, IID: Optional[bool] =False, log: Optional[bool]=True):
    """
    Variance Ratio test for random walk.

    Args:
        ts (np.ndarray):                Vector of time-series price data.
        lags (Union[int,List[int]]):    The period q is sused to create overlapping return horizons, Scalar or vector of integers 2 < q < len(ts)/2.
        alpha(float):                   Significance levels for the tests.
        debiased (Optional[bool]):      If True the variance estimates are debiased.
        IID (Optional[bool]):           By default the robust test statistics is computed, which assumes homoscedasticity increments of the random walk, else homoscedasticity increments are assumed.
        log (Optioinal[bool]):          By default log-returns are used, otherwise simple returns.

    Returns:
        h:      Vector of Boolean decisions based on the test statistic, values of h = 1 indicate rejection of the H_0 := random walk.
        ratio:  Vector of variance ratios.
        stat:   Vector of test statistics. Statistics are asymptotically standard normal.
        p_val:  Vector of p-values of the test statistics. Values are standard normal probabilities.

    Notes:
    The variance ratio VR(q) is the variance of the q-period overlapping return horizon scaled by q and divided by variance of the return series. 
    The variance ratio test assesses in the null hypothesis that a univariate time series y is a random walk. 
    The null model is y(t) = c + y(t–1) + e(t), where c is a drift constant and e(t) are uncorrelated innovations with zero mean.
    When VR(q) = 1 serially uncorrleated (log)returns and homoskedastic are satisfied. Rejection of the null indicates the presence of positive serial correlation in the time series. 
    For VR(q) < 1 the time series is mean-reverting series and for VR(q) > 1 trending. 

    References: Lo, A. W., and A. C. MacKinlay. (1988), "Stock market prices do not follow randomwalk:  Evidence from a simple specification test"
                Lo, A. W., and A. C. MacKinlay. (1989), "The Size and Power of the Variance Ratio Test."
                Amélie Charles, Olivier Darné (2013) "Variance ratio tests of random walk: An overview"
                http://www.eviews.com/help/helpintro.html#page/content/advtimeser-Variance_Ratio_Test.html
                https://de.mathworks.com/help/econ/vratiotest.html
    """
    if not isinstance(lags, Iterable):
        lags = [lags]

    ts = np.asarray(ts)
    ratio = []
    Z_list = []

    ## Calculate the Variance Ratio and Test Statistic for q
    for q in lags:

        if log:
            # overlapping log k_th difference
            return_1 = np.log(ts[1:]/ts[:-1])
            return_k = np.log(ts[q:]/ts[:-q])
            mu = np.mean(return_1)
        else:
            # overlapping k_th difference
            return_1 = ts[1:]/ts[:-1]-1
            return_k = ts[q:]/ts[:-q]-1
            mu = np.mean(return_1)

        if debiased:
            nq = len(return_1)-1
            m = (nq-q+1)*(1-q/nq)
        else:
            nq = len(return_1)
            m = len(return_k) #(T-k+1)

        var_1 = np.sum(np.square(return_1-mu))/(nq)
        var_k = np.sum(np.square(return_k-mu*q))/(m*q)
        
        # Variance Ratio
        VR = var_k/var_1
        ratio.append(VR)

        ## Calculate the Test Statistic
        # Test statistic under the assumption of homoscedasticity
        if IID:
            asymptotic_var = (2*(2*q-1)*(q-1))/(3*q*nq) 

        # Compute Heteroscedasticity robust S.E.
        else:
            denominator =  sum((return_1 -mu)**2)**2
            asymptotic_var = 0.0
            for j in range(1,q):

                # numerator = sum(np.square(np.log(ts[j+1:]/ts[j:-1])-mu)*np.square(np.log(ts[1:-j]/ts[:-(j+1)])-mu)) 
                numerator = np.square(return_1[j:]) @ np.square(return_1[:-j])
                delta = numerator/denominator
                asymptotic_var += np.square(2*(q-j)/q)*delta
        
        # Compute Test Statistic
        Z = (VR-1)/np.sqrt(asymptotic_var)
        Z_list.append(Z)

    # two-tailed test
    p_val = norm.sf(np.abs(Z_list))*2

    # Rejection of H_0
    h = (p_val < alpha).astype(int)

    return h, p_val,Z_list,ratio
