from montecarlo_functions import *
import numpy as np

def CVaR_Ret_Objective(weights, meanReturns, L, Z_fixed, initialPortfolio, lam):
    weights = np.array(weights)
    weights = np.clip(weights, 0, 1)
    weights = weights / np.sum(weights)

    terminal_values = np.zeros(Z_fixed.shape[0])

    for m in range(Z_fixed.shape[0]):
        dailyReturns = meanReturns.values + Z_fixed[m] @ L.T
        portfolio_path = np.cumprod(dailyReturns @ weights + 1) * initialPortfolio
        terminal_values[m] = portfolio_path[-1]

    portResults = pd.Series(terminal_values)

    expected_WT = portResults.mean()
    cvar_wealth = mcCVaR(portResults, alpha=5)
    cvar_loss = initialPortfolio - cvar_wealth

    return -(expected_WT - lam * cvar_loss)

def Sharpe_Objective(weights, meanReturns, L, Z_fixed, initialPortfolio, rf=0.04):
    """
    Monte Carlo Sharpe ratio objective.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    meanReturns : pd.Series or np.ndarray
        Expected per-period asset returns.
    L : np.ndarray
        Cholesky factor of covariance matrix.
    Z_fixed : np.ndarray
        Fixed random shocks with shape (mc_sims, T, n_assets).
    initialPortfolio : float
        Starting portfolio value.
    rf : float
        Risk-free return over the SAME horizon as the terminal return.
        For example, if terminal return is over T days, rf should also be over T days.

    Returns
    -------
    float
        Negative Sharpe ratio for minimization.
    """
    weights = np.array(weights, dtype=float)
    weights = np.clip(weights, 0, 1)
    weights = weights / np.sum(weights)

    mc_sims = Z_fixed.shape[0]
    terminal_values = np.zeros(mc_sims)

    mean_vec = meanReturns.values if hasattr(meanReturns, "values") else np.array(meanReturns)

    for m in range(mc_sims):
        dailyReturns = mean_vec + Z_fixed[m] @ L.T
        portfolio_path = np.cumprod(dailyReturns @ weights + 1) * initialPortfolio
        terminal_values[m] = portfolio_path[-1]

    terminal_returns = (terminal_values / initialPortfolio) - 1.0

    mean_ret = np.mean(terminal_returns)
    std_ret = np.std(terminal_returns, ddof=1)

    if std_ret < 1e-12:
        return 1e6

    sharpe = (mean_ret - rf) / std_ret
    return -sharpe

def mcVaR(returns, alpha = 5):
    """ Input: pandas series fo returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        return TypeError("Expected pd.Series, got {}".format(type(returns)))

def mcCVaR(returns, alpha = 5):
    """ Input: pandas series fo returns
        Output: CVaR or expected shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha =alpha)
        return returns[belowVaR].mean()
    else:
        return TypeError("Expected pd.Series, got {}".format(type(returns)))