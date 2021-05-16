# time-series-analysis
Projects based on statistical analysis of Cryptocurrencies

### Dependency
Quantifying the dependency between cryptocurrency returns and the effectof textual sentiment by computing the Mutual Information, Conditional Mutual Information and Correlation (linear & non-linear) of Cryptocurrency pairs and between Crytocurrency and Sentiment pairs. Analysis includes the calculation of p-values (permutation test) and confidence intervals (bootstrapping).

__To-Do List__
- [ ] Optimal number of bins when computing mutual information


### Scaling Behaviour
Examination of the time scaling properties, including stationarity, autocorrelation and generalized Hurst exponent


### Risk Measures
Calculation of VaR & CVaR with Historical Simulation, Normal Distribution, Student's t-Distribution. Computation of VaR for increasing Time Window with several methods (Parametric, Normal, Student's, EWMA). 

__To-Do List__
- [ ] VaR: Add Monte Carlo Simulation for Value at Risk
- [ ] VaR: Implement VaR Backtesting


### Distribution Fitting
Investigation of empirical probability distributions, including the parametric and non-parametric estimation of the Body & Tail PDF  of Cryptocurrencies for different time horizons. Includes the computation of Power Law Tail Exponents and the calculation of Confidence Intervals through Bootstrapping. 


__To-Do List__
- [ ] Kernel Density Estimation
- [ ] Tail: Fit Generalized Pareto distribution,  Generalized extreme value distribution,  Exponential Distributions
- [ ] Tail&Body: Statistical Test (Kolmogorow-Smirnow Test , Anderson–Darling Test)
