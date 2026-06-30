"""
DCA Rebalancer - Version 1

A simple Streamlit app that:
1. Lets you enter current portfolio holdings.
2. Fetches latest prices with yfinance.
3. Takes a new cash contribution amount.
4. Allocates that contribution only toward underweight assets.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd
import streamlit as st
import yfinance as yf


DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["VOO", "QQQM", "BND", "GLD"],
        "Shares": [0.0, 0.0, 0.0, 0.0],
        "Target %": [50.0, 20.0, 20.0, 10.0],
        "Asset Label": ["S&P 500", "NASDAQ 100", "Total Bond Market", "Gold"],
        "Asset Class": ["Equity", "Equity", "Bond", "Commodity"],
    }
)


st.set_page_config(page_title="DCA Rebalancer", page_icon="📈", layout="wide")


def clean_ticker(ticker: str) -> str:
    """Normalize ticker input."""
    return str(ticker).strip().upper()


@st.cache_data(ttl=300)
def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch latest available prices from yfinance.

    Uses fast_info where possible and falls back to recent history.
    Cache refreshes every 5 minutes.
    """
    prices: Dict[str, float] = {}

    for ticker in tickers:
        ticker = clean_ticker(ticker)
        if not ticker:
            continue

        try:
            yf_ticker = yf.Ticker(ticker)

            price = None
            try:
                fast_info = yf_ticker.fast_info
                price = fast_info.get("last_price")
            except Exception:
                price = None

            if price is None or pd.isna(price):
                history = yf_ticker.history(period="5d", interval="1d")
                if not history.empty:
                    price = history["Close"].dropna().iloc[-1]

            if price is None or pd.isna(price) or float(price) <= 0:
                raise ValueError(f"No valid price found for {ticker}")

            prices[ticker] = float(price)
        except Exception as exc:
            raise RuntimeError(f"Could not fetch price for {ticker}: {exc}") from exc

    return prices


def validate_portfolio(df: pd.DataFrame, contribution: float) -> List[str]:
    """Return validation errors for the portfolio inputs."""
    errors: List[str] = []

    required_columns = {"Ticker", "Shares", "Target %"}
    missing = required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return errors

    working = df.copy()
    working["Ticker"] = working["Ticker"].apply(clean_ticker)
    working = working[working["Ticker"] != ""]

    if working.empty:
        errors.append("Add at least one ticker.")
        return errors

    if working["Ticker"].duplicated().any():
        duplicated = working.loc[working["Ticker"].duplicated(), "Ticker"].tolist()
        errors.append(f"Duplicate tickers found: {', '.join(duplicated)}")

    if (pd.to_numeric(working["Shares"], errors="coerce").isna()).any():
        errors.append("Shares must be numeric.")
    elif (working["Shares"].astype(float) < 0).any():
        errors.append("Shares must be greater than or equal to 0.")

    if (pd.to_numeric(working["Target %"], errors="coerce").isna()).any():
        errors.append("Target % values must be numeric.")
    else:
        target_sum = working["Target %"].astype(float).sum()
        if not math.isclose(target_sum, 100.0, abs_tol=0.01):
            errors.append(f"Target weights must sum to 100%. Current sum: {target_sum:.2f}%")
        if (working["Target %"].astype(float) < 0).any():
            errors.append("Target % values must be greater than or equal to 0.")

    if contribution <= 0:
        errors.append("Contribution amount must be greater than $0.")

    return errors


def compute_dca_rebalance(portfolio_df: pd.DataFrame, contribution: float, prices: Dict[str, float]) -> pd.DataFrame:
    """
    Contribution-only rebalance algorithm.

    This buys underweight assets only. It does not recommend sells.

    Steps:
    1. Calculate current market value by asset.
    2. Calculate future portfolio value after adding contribution.
    3. Calculate each asset's target future dollar value.
    4. Gap = target future value - current value.
    5. Only positive gaps are eligible for buys.
    6. Allocate contribution proportionally across positive gaps.
    """
    df = portfolio_df.copy()
    df["Ticker"] = df["Ticker"].apply(clean_ticker)
    df = df[df["Ticker"] != ""].reset_index(drop=True)

    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0.0).astype(float)
    df["Target %"] = pd.to_numeric(df["Target %"], errors="coerce").fillna(0.0).astype(float)
    df["Target Weight"] = df["Target %"] / 100.0
    df["Price"] = df["Ticker"].map(prices)
    df["Current Value"] = df["Shares"] * df["Price"]

    current_total = float(df["Current Value"].sum())
    future_total = current_total + float(contribution)

    df["Current Weight %"] = 0.0
    if current_total > 0:
        df["Current Weight %"] = df["Current Value"] / current_total * 100.0

    df["Target Future Value"] = df["Target Weight"] * future_total
    df["Dollar Gap"] = df["Target Future Value"] - df["Current Value"]
    df["Positive Gap"] = df["Dollar Gap"].clip(lower=0.0)

    total_positive_gap = float(df["Positive Gap"].sum())

    if total_positive_gap <= 0:
        df["Recommended Buy $"] = 0.0
    else:
        # Allocate cash to underweight assets in proportion to their positive gaps.
        df["Recommended Buy $"] = contribution * (df["Positive Gap"] / total_positive_gap)

    df["Estimated Shares to Buy"] = df["Recommended Buy $"] / df["Price"]
    df["Post-Buy Value"] = df["Current Value"] + df["Recommended Buy $"]
    df["Post-Buy Weight %"] = df["Post-Buy Value"] / future_total * 100.0
    df["Still Off Target %"] = df["Post-Buy Weight %"] - df["Target %"]

    display_columns = [
        "Ticker",
        "Asset Label",
        "Asset Class",
        "Shares",
        "Price",
        "Current Value",
        "Current Weight %",
        "Target %",
        "Dollar Gap",
        "Recommended Buy $",
        "Estimated Shares to Buy",
        "Post-Buy Weight %",
        "Still Off Target %",
    ]

    # Keep optional columns only if they exist.
    display_columns = [col for col in display_columns if col in df.columns]
    return df[display_columns]


def format_results(df: pd.DataFrame) -> pd.DataFrame:
    """Round output for display."""
    rounded = df.copy()
    money_cols = ["Price", "Current Value", "Dollar Gap", "Recommended Buy $"]
    pct_cols = ["Current Weight %", "Target %", "Post-Buy Weight %", "Still Off Target %"]

    for col in money_cols:
        if col in rounded.columns:
            rounded[col] = rounded[col].astype(float).round(2)

    for col in pct_cols:
        if col in rounded.columns:
            rounded[col] = rounded[col].astype(float).round(2)

    if "Shares" in rounded.columns:
        rounded["Shares"] = rounded["Shares"].astype(float).round(6)
    if "Estimated Shares to Buy" in rounded.columns:
        rounded["Estimated Shares to Buy"] = rounded["Estimated Shares to Buy"].astype(float).round(6)

    return rounded


def main() -> None:
    st.title("DCA Rebalancer")
    st.caption("Version 1: buy underweight assets only. No sells. Prices from yfinance.")

    with st.expander("How the algorithm works", expanded=False):
        st.write(
            "The app calculates where each asset should be after your new contribution. "
            "It then buys only assets that are below their target dollar value. "
            "If an asset is overweight, the app recommends buying $0 of that asset."
        )

    st.subheader("1. Enter portfolio")
    portfolio_df = st.data_editor(
        DEFAULT_PORTFOLIO,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=0.000001, format="%.6f"),
            "Target %": st.column_config.NumberColumn("Target %", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "Asset Label": st.column_config.TextColumn("Asset Label"),
            "Asset Class": st.column_config.TextColumn("Asset Class"),
        },
        key="portfolio_editor",
    )

    st.subheader("2. Enter new contribution")
    contribution = st.number_input("Contribution amount ($)", min_value=0.0, value=500.0, step=50.0)

    target_sum = pd.to_numeric(portfolio_df.get("Target %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    st.caption(f"Current target weight sum: {target_sum:.2f}%")

    if st.button("Rebalance", type="primary"):
        errors = validate_portfolio(portfolio_df, contribution)

        if errors:
            for error in errors:
                st.error(error)
            return

        clean_df = portfolio_df.copy()
        clean_df["Ticker"] = clean_df["Ticker"].apply(clean_ticker)
        clean_df = clean_df[clean_df["Ticker"] != ""]
        tickers = clean_df["Ticker"].tolist()

        with st.spinner("Fetching latest prices..."):
            try:
                prices = fetch_prices(tickers)
            except RuntimeError as exc:
                st.error(str(exc))
                return

        results = compute_dca_rebalance(clean_df, contribution, prices)
        formatted = format_results(results)

        st.subheader("Recommended buys")
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        total_current = results["Current Value"].sum()
        total_buy = results["Recommended Buy $"].sum()
        future_total = total_current + total_buy

        col1, col2, col3 = st.columns(3)
        col1.metric("Current portfolio value", f"${total_current:,.2f}")
        col2.metric("Contribution allocated", f"${total_buy:,.2f}")
        col3.metric("Post-buy portfolio value", f"${future_total:,.2f}")

        csv = formatted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="dca_rebalance_results.csv",
            mime="text/csv",
        )

        st.info(
            "This is not financial advice. Prices may be delayed or approximate. "
            "Review all trades before placing orders with a broker."
        )


if __name__ == "__main__":
    main()
