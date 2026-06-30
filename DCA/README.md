# DCA Rebalancer

Version 1 of a simple contribution-only DCA rebalancer.

The app lets you enter:

- ticker
- current shares
- target allocation percentage
- optional asset label
- optional asset class
- new contribution amount

Then it fetches prices from `yfinance` and recommends how much of the new contribution to invest in each underweight asset.

## How to run

```bash
cd dca_rebalancer_app
pip install -r requirements.txt
streamlit run app.py
```

## Algorithm

This is a **buy-underweight-assets-only** algorithm.

It does not recommend sells.

For each asset:

```text
current_value = shares * price
future_total_value = current_portfolio_value + contribution
target_future_value = target_weight * future_total_value
dollar_gap = target_future_value - current_value
```

Only positive gaps are eligible for buys.

If total positive gaps exceed the contribution amount, the contribution is allocated proportionally across the underweight assets.

## Notes

- This is not connected to a brokerage.
- It does not place trades.
- It assumes fractional shares are allowed.
- Prices from `yfinance` may be delayed or approximate.
- This is not financial advice.
