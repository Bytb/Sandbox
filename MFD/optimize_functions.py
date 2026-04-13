from montecarlo_functions import *
import numpy as np

def simulate_terminal_values(weights, meanReturns, L, Z_fixed, initialPortfolio):
    """
    Simulate terminal portfolio values across all Monte Carlo paths.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    meanReturns : pd.Series or np.ndarray
        Expected per-period asset returns.
    L : np.ndarray
        Cholesky factor of covariance matrix.
    Z_fixed : np.ndarray
        Random shocks with shape (mc_sims, T, n_assets).
    initialPortfolio : float
        Starting portfolio value.

    Returns
    -------
    np.ndarray
        Terminal portfolio values of shape (mc_sims,).
    """
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    mean_vec = meanReturns.values if hasattr(meanReturns, "values") else np.array(meanReturns)

    # Shape: (mc_sims, T, n_assets)
    daily_returns = mean_vec[None, None, :] + Z_fixed @ L.T

    # Shape: (mc_sims, T)
    portfolio_returns = daily_returns @ weights

    # Shape: (mc_sims,)
    terminal_values = initialPortfolio * np.cumprod(1 + portfolio_returns, axis=1)[:, -1]

    return terminal_values


def CVaR_Ret_Objective(weights, meanReturns, L, Z_fixed, initialPortfolio, lam):
    """
    Objective that maximizes expected terminal wealth penalized by CVaR loss.
    Returned as negative for minimization.
    """
    terminal_values = simulate_terminal_values(
        weights, meanReturns, L, Z_fixed, initialPortfolio
    )

    portResults = pd.Series(terminal_values)

    expected_WT = portResults.mean()
    cvar_wealth = mcCVaR(portResults, alpha=5)
    cvar_loss = initialPortfolio - cvar_wealth

    return -(expected_WT - lam * cvar_loss)


def Sharpe_Objective(weights, meanReturns, L, Z_fixed, initialPortfolio, rf=0.04):
    """
    Monte Carlo Sharpe ratio objective.
    Returned as negative for minimization.
    """
    terminal_values = simulate_terminal_values(
        weights, meanReturns, L, Z_fixed, initialPortfolio
    )

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