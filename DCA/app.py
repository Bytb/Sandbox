import copy
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="DCA Rebalancer", layout="wide")

CONFIG_DIR = Path("configs")
CONFIG_DIR.mkdir(exist_ok=True)

DEFAULT_GROUPS = {
    "Hedges": {
        "group_weight": 25.0,
        "max_reallocate": None,
        "securities": [
            {"ticker": "TLT", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
            {"ticker": "GLD", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
        ],
    },
    "Slow Growth": {
        "group_weight": 25.0,
        "max_reallocate": None,
        "securities": [
            {"ticker": "VXUS", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
            {"ticker": "VTI", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
        ],
    },
    "High Growth": {
        "group_weight": 25.0,
        "max_reallocate": None,
        "securities": [
            {"ticker": "QQQM", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
            {"ticker": "SPYG", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
        ],
    },
    "Equities": {
        "group_weight": 25.0,
        "max_reallocate": None,
        "securities": [
            {"ticker": "AMZN", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
            {"ticker": "NFLX", "shares": 0.0, "security_weight": 50.0, "max_reallocate": None},
        ],
    },
}


def init_state():
    defaults = {
        "groups": copy.deepcopy(DEFAULT_GROUPS),
        "loaded_config": None,
        "pending_contribution": None,
        "ui_version": 0,
        "last_rebalance_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_cap(value):
    if value is None:
        return None

    text = str(value).strip().upper()

    if text in ["", "NA", "N/A", "NONE", "NULL"]:
        return None

    try:
        parsed = float(text)
        return parsed
    except ValueError:
        return None


def cap_to_text(value):
    if value is None:
        return "NA"
    return str(value)


def get_config_names():
    return sorted([p.stem for p in CONFIG_DIR.glob("*.json")])


def safe_config_name(name):
    return name.strip().replace(" ", "_")


def save_config(name, groups, contribution):
    name = safe_config_name(name)

    if not name:
        st.warning("Config name cannot be blank.")
        return

    data = {
        "contribution": float(contribution),
        "groups": groups,
    }

    with open(CONFIG_DIR / f"{name}.json", "w") as f:
        json.dump(data, f, indent=4)

    st.session_state.loaded_config = name
    st.success(f"Saved config: {name}")


def load_config(name):
    with open(CONFIG_DIR / f"{name}.json", "r") as f:
        data = json.load(f)

    st.session_state.groups = data.get("groups", copy.deepcopy(DEFAULT_GROUPS))
    st.session_state.loaded_config = name
    st.session_state.pending_contribution = float(data.get("contribution", 500.0))
    st.session_state.last_rebalance_result = None
    st.session_state.ui_version += 1


def overwrite_loaded_config(contribution=0.0):
    if not st.session_state.loaded_config:
        st.error("Load or save a config before updating.")
        return

    save_config(st.session_state.loaded_config, st.session_state.groups, contribution)


def clear_current_config():
    st.session_state.groups = {}
    st.session_state.loaded_config = None
    st.session_state.pending_contribution = 500.0
    st.session_state.last_rebalance_result = None
    st.session_state.ui_version += 1


def reset_defaults():
    st.session_state.groups = copy.deepcopy(DEFAULT_GROUPS)
    st.session_state.loaded_config = None
    st.session_state.pending_contribution = 500.0
    st.session_state.last_rebalance_result = None
    st.session_state.ui_version += 1


def fetch_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if data.empty:
            return None
        return float(data["Close"].dropna().iloc[-1])
    except Exception:
        return None


def build_portfolio_rows(groups):
    rows = []

    for group_name, group_data in groups.items():
        group_weight = float(group_data.get("group_weight", 0.0))
        group_cap = parse_cap(group_data.get("max_reallocate", None))

        for sec in group_data.get("securities", []):
            ticker = str(sec.get("ticker", "")).upper().strip()
            shares = float(sec.get("shares", 0.0))
            security_weight = float(sec.get("security_weight", 0.0))
            security_cap = parse_cap(sec.get("max_reallocate", None))
            final_weight = group_weight * security_weight / 100

            rows.append(
                {
                    "Group": group_name,
                    "Group Target %": group_weight,
                    "Group Max Reallocate %": group_cap,
                    "Ticker": ticker,
                    "Shares": shares,
                    "Security Weight in Group %": security_weight,
                    "Ticker Max Reallocate %": security_cap,
                    "Final Target %": final_weight,
                }
            )

    return pd.DataFrame(rows)


def validate_portfolio(df):
    errors = []

    if df.empty:
        return ["Portfolio cannot be empty."]

    group_sum = df[["Group", "Group Target %"]].drop_duplicates()["Group Target %"].sum()

    if abs(group_sum - 100) > 0.01:
        errors.append(f"Group weights must sum to 100%. Current sum: {group_sum:.2f}%")

    for group, group_df in df.groupby("Group"):
        internal_sum = group_df["Security Weight in Group %"].sum()
        if abs(internal_sum - 100) > 0.01:
            errors.append(
                f"{group} security weights must sum to 100%. Current sum: {internal_sum:.2f}%"
            )

    if df["Ticker"].isna().any() or (df["Ticker"].str.strip() == "").any():
        errors.append("Every row must have a ticker.")

    if (df["Shares"] < 0).any():
        errors.append("Shares cannot be negative.")

    for col in ["Group Max Reallocate %", "Ticker Max Reallocate %"]:
        non_na = df[col].dropna()
        if ((non_na < 0) | (non_na > 100)).any():
            errors.append(f"{col} must be NA or between 0 and 100.")

    return errors


def add_ticker(group_name):
    st.session_state.groups[group_name]["securities"].append(
        {"ticker": "", "shares": 0.0, "security_weight": 0.0, "max_reallocate": None}
    )
    st.session_state.ui_version += 1


def remove_ticker(group_name, index):
    securities = st.session_state.groups[group_name]["securities"]
    if len(securities) > 1:
        securities.pop(index)
    st.session_state.ui_version += 1


def add_group(group_name):
    group_name = group_name.strip()

    if not group_name:
        st.warning("Group name cannot be blank.")
        return

    if group_name in st.session_state.groups:
        st.warning("That group already exists.")
        return

    st.session_state.groups[group_name] = {
        "group_weight": 0.0,
        "max_reallocate": None,
        "securities": [
            {"ticker": "", "shares": 0.0, "security_weight": 100.0, "max_reallocate": None}
        ],
    }
    st.session_state.ui_version += 1


def delete_group(group_name):
    if len(st.session_state.groups) > 1:
        del st.session_state.groups[group_name]
        st.session_state.ui_version += 1


def fetch_prices_for_df(df):
    return {ticker: fetch_price(ticker) for ticker in df["Ticker"].unique()}


def add_current_values(df, prices):
    df = df.copy()
    df["Price"] = df["Ticker"].map(prices)

    if df["Price"].isna().any():
        missing = df[df["Price"].isna()]["Ticker"].tolist()
        raise ValueError(f"Could not fetch prices for: {', '.join(missing)}")

    df["Current Value"] = df["Shares"] * df["Price"]

    current_total = df["Current Value"].sum()
    df["Current Weight %"] = 0.0

    if current_total > 0:
        df["Current Weight %"] = df["Current Value"] / current_total * 100

    return df


def cap_pct_to_dollars(cap_pct, future_total):
    if cap_pct is None or pd.isna(cap_pct):
        return math.inf
    return float(cap_pct) / 100 * future_total


def compute_rebalance(df, contribution, min_shares_to_buy=1.0):
    df = df.copy()

    current_total = df["Current Value"].sum()
    future_total = current_total + contribution

    df["Target Value After Contribution"] = df["Final Target %"] / 100 * future_total
    df["Dollar Gap"] = df["Target Value After Contribution"] - df["Current Value"]

    group_current = df.groupby("Group")["Current Value"].sum()
    group_targets = df[["Group", "Group Target %"]].drop_duplicates().set_index("Group")
    group_target_values = group_targets["Group Target %"] / 100 * future_total

    group_gaps = group_target_values - group_current
    df["Group Gap"] = df["Group"].map(group_gaps)

    df["Ticker Cap $"] = df["Ticker Max Reallocate %"].apply(
        lambda cap: cap_pct_to_dollars(cap, future_total)
    )

    group_caps = (
        df[["Group", "Group Max Reallocate %"]]
        .drop_duplicates()
        .set_index("Group")["Group Max Reallocate %"]
        .apply(lambda cap: cap_pct_to_dollars(cap, future_total))
    )

    df["Group Cap $"] = df["Group"].map(group_caps)

    df["Candidate Gap"] = df.apply(
        lambda row: min(max(row["Dollar Gap"], 0), row["Ticker Cap $"])
        if row["Group Gap"] > 0 and row["Dollar Gap"] > 0
        else 0.0,
        axis=1,
    )

    df["Capped Eligible Gap"] = 0.0

    for group, group_df in df.groupby("Group"):
        idx = group_df.index
        group_cap = group_df["Group Cap $"].iloc[0]
        candidate_sum = group_df["Candidate Gap"].sum()

        if candidate_sum <= 0:
            df.loc[idx, "Capped Eligible Gap"] = 0.0
        elif candidate_sum > group_cap:
            scale = group_cap / candidate_sum
            df.loc[idx, "Capped Eligible Gap"] = group_df["Candidate Gap"] * scale
        else:
            df.loc[idx, "Capped Eligible Gap"] = group_df["Candidate Gap"]

    # Enforce minimum share purchase:
    # If buy would be between 0 and min_shares_to_buy, set it to 0.
    df["Minimum Buy $"] = df["Price"] * min_shares_to_buy
    df["Capped Eligible Gap"] = df.apply(
        lambda row: row["Capped Eligible Gap"]
        if row["Capped Eligible Gap"] >= row["Minimum Buy $"]
        else 0.0,
        axis=1,
    )

    total_capped_gap = df["Capped Eligible Gap"].sum()
    spendable_contribution = min(contribution, total_capped_gap)

    if total_capped_gap > 0:
        df["Recommended Buy $"] = df["Capped Eligible Gap"] / total_capped_gap * spendable_contribution
    else:
        df["Recommended Buy $"] = 0.0

    df["Estimated Shares to Buy"] = df["Recommended Buy $"] / df["Price"]

    # Final cleanup: prevent tiny fractional recommendations after proportional scaling
    df.loc[df["Estimated Shares to Buy"] < min_shares_to_buy, "Recommended Buy $"] = 0.0
    df.loc[df["Estimated Shares to Buy"] < min_shares_to_buy, "Estimated Shares to Buy"] = 0.0

    total_after_filter = df["Recommended Buy $"].sum()

    if total_after_filter > 0:
        df["Weight After Rebalance %"] = (
            (df["Current Value"] + df["Recommended Buy $"]) / future_total * 100
        )
    else:
        df["Weight After Rebalance %"] = df["Current Weight %"]

    return df

def implement_rebalance(result_df):
    buys = result_df[["Ticker", "Estimated Shares to Buy"]].copy()

    for group_name, group_data in st.session_state.groups.items():
        for sec in group_data.get("securities", []):
            ticker = str(sec.get("ticker", "")).upper().strip()
            buy_rows = buys[buys["Ticker"] == ticker]

            if not buy_rows.empty:
                added_shares = float(buy_rows["Estimated Shares to Buy"].sum())
                sec["shares"] = float(sec.get("shares", 0.0)) + added_shares

    st.session_state.pending_contribution = 0.0
    st.session_state.last_rebalance_result = None
    overwrite_loaded_config(contribution=0.0)
    st.session_state.ui_version += 1
    st.success("Implemented rebalance and updated the loaded config.")


def plot_group_pie(df):
    group_df = (
        df.groupby("Group", as_index=False)["Current Value"]
        .sum()
        .query("`Current Value` > 0")
    )

    if group_df.empty:
        st.info("Enter shares above 0 to see group allocation.")
        return

    fig = px.pie(group_df, names="Group", values="Current Value", hole=0.45, title="By Group")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=330, margin=dict(l=5, r=5, t=45, b=5))
    st.plotly_chart(fig, use_container_width=True)


def plot_security_pie(df):
    security_df = df[df["Current Value"] > 0].copy()

    if security_df.empty:
        st.info("Enter shares above 0 to see security allocation.")
        return

    fig = px.pie(security_df, names="Ticker", values="Current Value", title="By Security")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=330, margin=dict(l=5, r=5, t=45, b=5))
    st.plotly_chart(fig, use_container_width=True)


init_state()

st.title("DCA Portfolio Rebalancer")
st.caption("Grouped buy-underweight-only DCA calculator.")

with st.sidebar:
    st.header("Contribution")

    contribution_default = st.session_state.pending_contribution
    if contribution_default is None:
        contribution_default = st.session_state.get("contribution", 500.0)

    contribution = st.number_input(
        "New money to invest",
        min_value=0.0,
        value=float(contribution_default),
        step=50.0,
        key=f"{st.session_state.ui_version}_contribution",
    )

    min_shares_to_buy = st.number_input(
        "Minimum shares per buy",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

    st.session_state.pending_contribution = None

    st.markdown("---")
    st.subheader("Configs")

    config_names = get_config_names()

    selected_config = st.selectbox(
        "Saved configs",
        options=config_names,
        index=0 if config_names else None,
        placeholder="No saved configs yet",
        key=f"{st.session_state.ui_version}_config_select",
    )

    if st.button("Load old config", use_container_width=True, disabled=not bool(config_names)):
        load_config(selected_config)
        st.rerun()

    save_name = st.text_input(
        "Save config as",
        value=st.session_state.loaded_config or "my_portfolio",
        key=f"{st.session_state.ui_version}_save_name",
    )

    if st.button("Save current config", use_container_width=True):
        save_config(save_name, st.session_state.groups, contribution)

    if st.button(
        "Update loaded config",
        use_container_width=True,
        disabled=st.session_state.loaded_config is None,
    ):
        overwrite_loaded_config(contribution=contribution)

    if st.button("Clear current config", use_container_width=True):
        clear_current_config()
        st.rerun()

    st.markdown("---")
    st.subheader("Add Group")

    new_group_name = st.text_input("New group name", key=f"{st.session_state.ui_version}_new_group")

    if st.button("Add group", use_container_width=True):
        add_group(new_group_name)
        st.rerun()

    if st.button("Reset defaults", use_container_width=True):
        reset_defaults()
        st.rerun()


st.header("Portfolio Editor")

group_names = list(st.session_state.groups.keys())
ui_version = st.session_state.ui_version

if st.session_state.loaded_config:
    st.info(f"Loaded config: {st.session_state.loaded_config}")
else:
    st.warning("No config loaded. Save or load a config before using Implement.")

if not group_names:
    st.warning("Current config is empty. Add a group in the sidebar or reset defaults.")

cols = st.columns(2)

for idx, group_name in enumerate(group_names):
    group_data = st.session_state.groups[group_name]

    with cols[idx % 2]:
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])

            with top_left:
                st.subheader(group_name)

            with top_right:
                if st.button("Delete group", key=f"{ui_version}_{group_name}_delete_group"):
                    delete_group(group_name)
                    st.rerun()

            group_col1, group_col2 = st.columns([1, 1])

            with group_col1:
                group_weight = st.number_input(
                    "Group target %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(group_data.get("group_weight", 0.0)),
                    step=1.0,
                    key=f"{ui_version}_{group_name}_group_weight",
                )

            with group_col2:
                group_max_reallocate_text = st.text_input(
                    "Group max realloc %",
                    value=cap_to_text(group_data.get("max_reallocate", None)),
                    placeholder="NA",
                    key=f"{ui_version}_{group_name}_max_reallocate",
                )

            st.session_state.groups[group_name]["group_weight"] = group_weight
            st.session_state.groups[group_name]["max_reallocate"] = parse_cap(group_max_reallocate_text)

            st.markdown("**Ticker | Shares | Percent | Max Realloc**")

            for i, sec in enumerate(list(group_data.get("securities", []))):
                c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.0, 1.0, 0.45])

                with c1:
                    ticker = st.text_input(
                        "Ticker",
                        value=sec.get("ticker", ""),
                        label_visibility="collapsed",
                        placeholder="Ticker",
                        key=f"{ui_version}_{group_name}_{i}_ticker",
                    )

                with c2:
                    shares = st.number_input(
                        "Shares",
                        min_value=0.0,
                        value=float(sec.get("shares", 0.0)),
                        step=0.01,
                        label_visibility="collapsed",
                        key=f"{ui_version}_{group_name}_{i}_shares",
                    )

                with c3:
                    security_weight = st.number_input(
                        "Percent",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(sec.get("security_weight", 0.0)),
                        step=1.0,
                        label_visibility="collapsed",
                        key=f"{ui_version}_{group_name}_{i}_security_weight",
                    )

                with c4:
                    ticker_max_reallocate_text = st.text_input(
                        "Max Realloc",
                        value=cap_to_text(sec.get("max_reallocate", None)),
                        label_visibility="collapsed",
                        placeholder="NA",
                        key=f"{ui_version}_{group_name}_{i}_max_reallocate",
                    )

                with c5:
                    if st.button("✕", key=f"{ui_version}_{group_name}_{i}_remove"):
                        remove_ticker(group_name, i)
                        st.rerun()

                st.session_state.groups[group_name]["securities"][i] = {
                    "ticker": ticker,
                    "shares": shares,
                    "security_weight": security_weight,
                    "max_reallocate": parse_cap(ticker_max_reallocate_text),
                }

            if st.button("+ Add ticker", key=f"{ui_version}_{group_name}_add_ticker"):
                add_ticker(group_name)
                st.rerun()


portfolio_df = build_portfolio_rows(st.session_state.groups) if group_names else pd.DataFrame()
errors = validate_portfolio(portfolio_df) if group_names else ["Portfolio cannot be empty."]

st.header("Summary")

summary_col1, summary_col2 = st.columns([1.2, 0.8])

with summary_col1:
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True, height=230)

with summary_col2:
    if errors:
        for error in errors:
            st.error(error)
    else:
        st.success("Portfolio inputs are valid.")

priced_df = None

if not errors:
    try:
        prices = fetch_prices_for_df(portfolio_df)
        priced_df = add_current_values(portfolio_df, prices)

        current_total = priced_df["Current Value"].sum()
        future_total = current_total + contribution

        m1, m2, m3 = st.columns(3)
        m1.metric("Current value", f"${current_total:,.2f}")
        m2.metric("Contribution", f"${contribution:,.2f}")
        m3.metric("After contribution", f"${future_total:,.2f}")

        st.header("Current Portfolio Allocation")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            plot_group_pie(priced_df)

        with chart_col2:
            plot_security_pie(priced_df)

    except Exception as e:
        st.warning(str(e))


st.header("Rebalance")

if st.button("Rebalance", type="primary"):
    if errors:
        st.error("Fix validation errors before rebalancing.")
    elif contribution <= 0:
        st.error("Contribution must be greater than $0.")
    else:
        try:
            if priced_df is None:
                prices = fetch_prices_for_df(portfolio_df)
                priced_df = add_current_values(portfolio_df, prices)

            result = compute_rebalance(priced_df, contribution, min_shares_to_buy)
            st.session_state.last_rebalance_result = result

        except Exception as e:
            st.error(str(e))

if st.session_state.last_rebalance_result is not None:
    result = st.session_state.last_rebalance_result
    result = result.sort_values(
        by="Estimated Shares to Buy",
        ascending=False
    ).reset_index(drop=True)

    display_cols = [
        "Group",
        "Ticker",
        "Estimated Shares to Buy",
        "Recommended Buy $",
        "Current Weight %",
        "Weight After Rebalance %",
        "Final Target %",
        "Shares",
        "Price",
        "Current Value",
        "Dollar Gap",
        "Group Gap",
        "Group Max Reallocate %",
        "Ticker Max Reallocate %",
        "Group Cap $",
        "Ticker Cap $",
    ]

    st.subheader("Recommended Buys")

    styled = result[display_cols].style.format(
        {
            "Estimated Shares to Buy": "{:.6f}",
            "Current Weight %": "{:.2f}%",
            "Weight After Rebalance %": "{:.2f}%",
            "Final Target %": "{:.2f}%",
            "Shares": "{:.4f}",
            "Price": "${:,.2f}",
            "Current Value": "${:,.2f}",
            "Dollar Gap": "${:,.2f}",
            "Group Gap": "${:,.2f}",
            "Recommended Buy $": "${:,.2f}",
            "Group Max Reallocate %": "{:.2f}%",
            "Ticker Max Reallocate %": "{:.2f}%",
            "Group Cap $": "${:,.2f}",
            "Ticker Cap $": "${:,.2f}",
        },
        na_rep="NA",
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=330,
    )

    total_buy = result["Recommended Buy $"].sum()
    unallocated = contribution - total_buy

    st.metric("Total allocated", f"${total_buy:,.2f}")

    if unallocated > 0.01:
        st.warning(
            f"${unallocated:,.2f} was not allocated because no eligible underweight assets were found or caps were reached."
        )

    can_implement = st.session_state.loaded_config is not None and total_buy > 0

    if st.button("Implement", type="secondary", disabled=not can_implement):
        implement_rebalance(result)
        st.rerun()

    if st.session_state.loaded_config is None:
        st.info("Save or load a config before using Implement.")