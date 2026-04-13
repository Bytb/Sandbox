import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from scipy.optimize import minimize
from tqdm import tqdm
from optimize_functions import CVaR_Ret_Objective, Sharpe_Objective, mcVaR, mcCVaR

def get_data(stocks, start, end):
  stockData = yf.download(stocks, start, end, auto_adjust=False)
  stockData = stockData['Close']
  returns = stockData.pct_change().dropna()
  meanReturns = returns.mean()
  covMatrix = returns.cov()
  return meanReturns, covMatrix

def print_portfolio_stats(portfolio_sims, initialPortfolio, alpha=5, rf=0.04, print_stats=True):
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

    if print_stats:
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
               min_allocation=0, max_allocation=40,
               optimize=False, method="CVaR", show_stats=True,
               show_plots=True):

    if t0 is None:
        t0 = dt.datetime.now()

    opt_maxiter = 100 if optimize else 0
    total_steps = 6 + opt_maxiter

    pbar = tqdm(total=total_steps, desc="Monte Carlo", unit="step")

    # ---------------- SETUP ----------------
    pbar.set_postfix_str("setup")
    start_date = t0 - dt.timedelta(days=look_back)
    pbar.update(1)

    # ---------------- WEIGHTS ----------------
    pbar.set_postfix_str("weights")
    if weights == "random":
        weights = np.random.rand(len(stock_tickers))
    else:
        weights = np.array(weights, dtype=float) / 100.0
    weights = weights / np.sum(weights)
    pbar.update(1)

    # ---------------- DATA ----------------
    pbar.set_postfix_str("downloading data")
    meanReturns, covMatrix = get_data(stock_tickers, start_date, t0)
    pbar.update(1)

    # ---------------- PREP ----------------
    pbar.set_postfix_str("matrix prep")
    meanM = np.full((projection_len, len(weights)), meanReturns).T
    pbar.update(1)

    pbar.set_postfix_str("cholesky")
    L = np.linalg.cholesky(covMatrix)
    pbar.update(1)

    # ---------------- OPTIMIZATION ----------------
    if optimize:
        pbar.set_postfix_str("optimizing")

        Z_fixed = np.random.normal(size=(num_sims, projection_len, len(weights)))
        lam = 1.0
        n_assets = len(stock_tickers)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(min_allocation/100, max_allocation/100) for _ in range(n_assets)]

        if method == "CVaR":
            objective = CVaR_Ret_Objective
            args = (meanReturns, L, Z_fixed, initial_portfolio, lam)
        else:
            objective = Sharpe_Objective
            args = (meanReturns, L, Z_fixed, initial_portfolio, 0.04)

        opt_iter = {"count": 0}

        def callback(xk):
            if opt_iter["count"] < opt_maxiter:
                opt_iter["count"] += 1
                pbar.update(1)

        result = minimize(
            objective,
            weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options={"maxiter": opt_maxiter}
        )

        # Fill unused steps
        remaining = opt_maxiter - opt_iter["count"]
        if remaining > 0:
            pbar.update(remaining)

        if result.success:
            weights = result.x / np.sum(result.x)
        else:
            print("Optimization failed:", result.message)

    # ---------------- SIMULATION ----------------
    pbar.set_postfix_str("simulating")

    Z = np.random.normal(size=(num_sims, projection_len, len(weights)))
    daily_returns = meanReturns.values[None, None, :] + Z @ L.T
    portfolio_returns = daily_returns @ weights
    portfolio_sims = initial_portfolio * np.cumprod(1 + portfolio_returns, axis=1).T

    pbar.update(1)

    # ---------------- CLEAN OUTPUT ----------------
    if show_stats:
        # ---------------- STATS ----------------
        pbar.set_postfix_str("finalizing")
        stats = print_portfolio_stats(portfolio_sims, initial_portfolio, alpha=alpha, print_stats=False)
        pbar.update(1)
        print("\n" + "="*35)
        print("     OPTIMAL PORTFOLIO")
        print("="*35)

        for stock, w in zip(stock_tickers, weights):
            print(f"{stock:<6}: {w*100:>6.2f}%")

        print("\n" + "="*35)
        print("     PERFORMANCE METRICS")
        print("="*35)

        print(f"Sharpe Ratio:                {stats['sharpe']:.4f}")
        print(f"VaR:                         ${stats['VaR']:.2f}")
        print(f"CVaR:                        ${stats['CVaR']:.2f}")
        print(f"Mean Portfolio Value:        ${stats['mean_line']:.2f}")
        print(f"Std Dev:                     ${stats['std_line']:.2f}")
        print(f"Expected Total Return:       ${stats['total_return']:.2f}")
        print(f"Expected Percent Return:     {stats['percent_return']:.2f}%")
        print(f"Probability of Profit:       {stats['percent_profit']:.2f}%")

    # 🔥 CLOSE BAR BEFORE PRINTING
    pbar.close()
    if show_plots:
        plot_portfolio_results(
            portfolio_sims,
            initial_portfolio,
            stats['percentile_line'],
            stats['mean_line'],
            stats['portResults'],
            alpha=alpha
        )

    return portfolio_sims