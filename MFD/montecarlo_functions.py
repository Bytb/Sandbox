import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from scipy.optimize import minimize

from optimize_functions import *

def get_data(stocks, start, end):
  stockData = yf.download(stocks, start, end, auto_adjust=False)
  stockData = stockData['Close']
  returns = stockData.pct_change().dropna()
  meanReturns = returns.mean()
  covMatrix = returns.cov()
  return meanReturns, covMatrix

def print_portfolio_stats(portfolio_sims, initialPortfolio, alpha=5, rf=0.04):
    portResults = pd.Series(portfolio_sims[-1, :])

    VaR = initialPortfolio - mcVaR(portResults, alpha=alpha)
    CVaR = initialPortfolio - mcCVaR(portResults, alpha=alpha)
    percentile_line = np.percentile(portResults, alpha)
    mean_line = np.mean(portResults)
    std_line = np.std(portResults)

    total_return = mean_line - initialPortfolio
    percent_return = (total_return / initialPortfolio) * 100
    percent_profit = np.mean(portResults > initialPortfolio) * 100

    terminal_returns = (portResults / initialPortfolio) - 1

    mean_ret = np.mean(terminal_returns)
    std_ret = np.std(terminal_returns, ddof=1)

    sharpe = (mean_ret - rf) / std_ret if std_ret > 1e-12 else 0

    print('Sharpe Ratio {:.4f}'.format(sharpe))
    print('VaR ${}'.format(round(VaR, 2)))
    print('CVaR ${}'.format(round(CVaR, 2)))
    print('Mean ${}'.format(round(mean_line, 2)))
    print('Standard deviation ${}'.format(round(std_line, 2)))
    print('Expected Total Return ${}'.format(round(total_return, 2)))
    print('Expected Percent Return {}%'.format(round(percent_return, 2)))
    print('Expected Percent of Making a Profit {}%'.format(round(percent_profit, 2)))

    return {
        "portResults": portResults,
        "VaR": VaR,
        "CVaR": CVaR,
        "percentile_line": percentile_line,
        "mean_line": mean_line,
        "std_line": std_line,
        "total_return": total_return,
        "percent_return": percent_return,
        "percent_profit": percent_profit,
        "sharpe": sharpe
    }

def plot_portfolio_results(portfolio_sims, initialPortfolio, percentile_line, mean_line, portResults, alpha=5):
    # Path plot
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.axhline(percentile_line, linestyle='--', color='r',
                label=f'{alpha}th percentile final value = {percentile_line:.2f}')
    plt.axhline(mean_line, linestyle='--', color='b',
                label=f'Mean final value = {mean_line:.2f}')
    plt.title('MC Simulation of a Portfolio')
    plt.legend()
    plt.show()

    # Histogram
    final_returns = portResults - initialPortfolio
    mean_return = final_returns.mean()
    mc_sims = portfolio_sims.shape[1]

    plt.hist(final_returns, bins=int(math.sqrt(mc_sims)), edgecolor='black')
    plt.axvline(0, linestyle='--', label='Break-even')
    plt.axvline(mean_return, linestyle='--', color='r',
                label=f'Mean = {round(mean_return, 2)}')

    plt.xlabel("Final Return")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Monte Carlo Returns")
    plt.legend()
    plt.show()

def MonteCarlo(initial_portfolio, stock_tickers, weights, projection_len=365,
               t0=None, look_back=365, alpha=5, num_sims=1000,
               optimize=False, allow_short=False, use_CVaR=False):

    if t0 is None:
        t0 = dt.datetime.now()

    start_date = t0 - dt.timedelta(days=look_back)

    if weights == "random":
        weights = np.random.rand(len(stock_tickers))
    else:
        weights = np.array(weights, dtype=float) / 100.0

    weights = weights / np.sum(weights)

    portfolio_sims = np.full((projection_len, num_sims), 0.0)
    meanReturns, covMatrix = get_data(stock_tickers, start_date, t0)

    meanM = np.full((projection_len, len(weights)), meanReturns)
    meanM = meanM.T

    L = np.linalg.cholesky(covMatrix)

    if optimize:
        Z_fixed = np.random.normal(size=(num_sims, projection_len, len(weights)))
        lam = 1.0
        n_assets = len(stock_tickers)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if allow_short:
            bounds = [(-0.2, 0.4) for _ in range(n_assets)]
        else:
            bounds = [(0, 0.4) for _ in range(n_assets)]

        objective = CVaR_Ret_Objective if use_CVaR else Sharpe_Objective

        result = minimize(
            objective,
            weights,
            args=(meanReturns, L, Z_fixed, initial_portfolio, lam),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )



        if not result.success:
            print("Optimization failed:", result.message)
        else:
            weights = result.x / np.sum(result.x)
            print("\nOptimal Weights (%):")
            for stock, w in zip(stock_tickers, weights):
                print(f"{stock}: {w * 100:.2f}%")

    portfolio_sims = np.full((projection_len, num_sims), 0.0)

    for m in range(num_sims):
        Z = np.random.normal(size=(projection_len, len(weights)))
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initial_portfolio

    stats = print_portfolio_stats(portfolio_sims, initial_portfolio, alpha=alpha)
    plot_portfolio_results(
        portfolio_sims,
        initial_portfolio,
        stats['percentile_line'],
        stats['mean_line'],
        stats['portResults'],
        alpha=alpha
    )

    return portfolio_sims