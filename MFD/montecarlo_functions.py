import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from scipy.optimize import minimize
from optimize_functions import CVaR_Ret_Objective, Sharpe_Objective, mcVaR, mcCVaR

def get_data(stocks, start, end, print_stats=True):
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=False)

    close = stockData['Close']
    returns = close.pct_change().dropna()

    meanReturns = returns.mean()
    covMatrix = returns.cov()

    if print_stats:

        print("\n" + "=" * 75)
        print("                            DATA DIAGNOSTICS")
        print("=" * 75)

        # ================= RETURN STATS =================
        daily_mean = returns.mean()
        annual_mean = daily_mean * 252
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = annual_mean / annual_vol

        # ================= DIVIDENDS =================
        div_yield_yearly = []
        div_yield_quarterly = []

        for stock in stocks:
            ticker = yf.Ticker(stock)
            divs = ticker.dividends.copy()

            if len(divs) == 0:
                div_yield_yearly.append(0.0)
                div_yield_quarterly.append(0.0)
                continue

            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)

            divs = divs[(divs.index >= start) & (divs.index <= end)]

            price_series = close[stock].copy()
            price_series = price_series[(price_series.index >= start) & (price_series.index <= end)]

            if len(divs) == 0 or len(price_series) == 0:
                div_yield_yearly.append(0.0)
                div_yield_quarterly.append(0.0)
                continue

            # Yearly yield: average of yearly (dividends / avg yearly price)
            divs_y = divs.resample("YE").sum()
            prices_y = price_series.resample("YE").mean()
            yearly_yields = (divs_y / prices_y).replace([np.inf, -np.inf], np.nan).dropna()
            div_yield_yearly.append(yearly_yields.mean() if len(yearly_yields) > 0 else 0.0)

            # Quarterly yield: average of quarterly (dividends / avg quarterly price)
            divs_q = divs.resample("QE").sum()
            prices_q = price_series.resample("QE").mean()
            quarterly_yields = (divs_q / prices_q).replace([np.inf, -np.inf], np.nan).dropna()
            div_yield_quarterly.append(quarterly_yields.mean() if len(quarterly_yields) > 0 else 0.0)

        div_yield_yearly = pd.Series(div_yield_yearly, index=stocks)
        div_yield_quarterly = pd.Series(div_yield_quarterly, index=stocks)

        # ================= MAX DRAWDOWN =================
        max_dd_list = []

        for stock in stocks:
            price_series = close[stock]
            running_max = price_series.cummax()
            drawdown = (price_series - running_max) / running_max
            max_dd_list.append(drawdown.min())

        max_dd_series = pd.Series(max_dd_list, index=stocks)

        # ================= COMBINE =================
        full_df = pd.DataFrame({
            "Mean Daily": daily_mean,
            "Annual Ret": annual_mean,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "Div Yld (Y)": div_yield_yearly,
            "Div Yld (Q)": div_yield_quarterly,
            "Max DD": max_dd_series
        })

        # Optional: sort by Sharpe descending
        # full_df = full_df.sort_values("Sharpe", ascending=False)

        # ================= FORMAT =================
        full_df_print = full_df.copy()

        full_df_print["Mean Daily"] = full_df_print["Mean Daily"].map(lambda x: f"{x:.4%}")
        full_df_print["Annual Ret"] = full_df_print["Annual Ret"].map(lambda x: f"{x:.2%}")
        full_df_print["Volatility"] = full_df_print["Volatility"].map(lambda x: f"{x:.2%}")
        full_df_print["Sharpe"] = full_df_print["Sharpe"].map(lambda x: f"{x:.3f}")
        full_df_print["Div Yld (Y)"] = full_df_print["Div Yld (Y)"].map(lambda x: f"{x:.2%}")
        full_df_print["Div Yld (Q)"] = full_df_print["Div Yld (Q)"].map(lambda x: f"{x:.2%}")
        full_df_print["Max DD"] = full_df_print["Max DD"].map(lambda x: f"{x:.2%}")

        print("\n--- PORTFOLIO INPUT STATS ---")
        print(full_df_print.to_string(col_space=12))

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

def MonteCarlo(initial_portfolio, stock_tickers, weights='random', projection_len=365,
               t0=None, look_back=365, alpha=5, num_sims=1000,
               min_allocation=0, max_allocation=40,
               optimize=False, method="CVaR", show_stats=True,
               show_plots=True):

    if t0 is None:
        t0 = dt.datetime.now()

    # ---------------- SETUP ----------------
    start_date = t0 - dt.timedelta(days=look_back)

    # ---------------- WEIGHTS ----------------
    if weights == "random":
        weights = np.random.rand(len(stock_tickers))
    else:
        weights = np.array(weights, dtype=float) / 100.0
    weights = weights / np.sum(weights)

    # ---------------- DATA ----------------
    meanReturns, covMatrix = get_data(stock_tickers, start_date, t0, show_stats)

    # ---------------- PREP ----------------
    meanM = np.full((projection_len, len(weights)), meanReturns).T

    L = np.linalg.cholesky(covMatrix)

    # ---------------- OPTIMIZATION ----------------
    if optimize:
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

        result = minimize(
            objective,
            weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x / np.sum(result.x)
        else:
            print("Optimization failed:", result.message)

    # ---------------- SIMULATION ----------------
    Z = np.random.normal(size=(num_sims, projection_len, len(weights)))
    daily_returns = meanReturns.values[None, None, :] + Z @ L.T
    portfolio_returns = daily_returns @ weights
    portfolio_sims = initial_portfolio * np.cumprod(1 + portfolio_returns, axis=1).T

    # ---------------- CLEAN OUTPUT ----------------
    if show_stats:
        stats = print_portfolio_stats(portfolio_sims, initial_portfolio, alpha=alpha, print_stats=False)
        print("\n" + "="*35)
        print("     PORTFOLIO WEIGHTS")
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