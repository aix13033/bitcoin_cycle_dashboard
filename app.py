"""
Bitcoin Cycle Top Indicator Dashboard
===================================

This Streamlit application aggregates on‑chain, exchange and macro data
to help identify bitcoin market cycle tops.  Metrics are fetched from
several third‑party APIs (BGeometrics, Coingecko, Binance, DefiLlama
and Yahoo Finance) and combined with simple technical indicators such
as the Pi Cycle moving averages.  When at least three of the five
primary indicators align—MVRV Z‑Score, Long Term Holder SOPR,
exchange inflows, Reserve Risk and Pi‑Cycle status—a “high
confidence” signal is raised.  Additional escalation levels and
Coinglass’s optional bull‑market peak indicators provide further
context.

This code is intended to run under Streamlit.  You can launch the
dashboard locally with

    streamlit run app.py

No API key is required for the on‑chain metrics; however, the
Coinglass signals require your own API key.  The upstream APIs are
rate‑limited, so if data appears missing (`N/A`) please wait a few
minutes before refreshing.
"""

import os
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import yfinance as yf

###############################################################################
# Coinglass Bull Market Peak Signals
###############################################################################

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_coinglass_peak_indicators(api_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch bull market peak indicators from the Coinglass API.

    This endpoint returns a list of indicator objects with fields such as
    ``name``, ``value``, ``targetValue`` and ``hit`` (whether the target is
    met).  Passing a blank API key returns ``None``.

    Parameters
    ----------
    api_key : str
        Your Coinglass API key.  Must be passed via the ``CG-API-KEY`` header.

    Returns
    -------
    DataFrame or None
        DataFrame containing indicator data or ``None`` if the request fails
        or the key is missing.
    """
    if not api_key:
        return None
    url = "https://open-api-v3.coinglass.com/api/bull-market-peak-indicator"
    headers = {"CG-API-KEY": api_key, "accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # The response structure may vary; attempt to extract the list of items
        items: Optional[list] = None
        if isinstance(data, dict):
            # v3/v4 API may wrap data under 'data', 'dataList' or 'result'
            for key in ["data", "dataList", "result", "resultList"]:
                val = data.get(key)
                if isinstance(val, list):
                    items = val
                    break
            # sometimes the list is nested under data->list
            if items is None and isinstance(data.get("data"), dict):
                for subkey in ["list", "items"]:
                    val = data["data"].get(subkey)
                    if isinstance(val, list):
                        items = val
                        break
        if not items:
            return None
        df = pd.DataFrame(items)
        return df
    except Exception:
        return None

###############################################################################
# Helpers for remote data fetching
###############################################################################

def _bgeometrics_get(endpoint: str, value_field: str, days: int | None = None) -> Optional[pd.DataFrame]:
    """
    Fetch a timeseries metric from the free BGeometrics on‑chain API.

    Parameters
    ----------
    endpoint : str
        The metric endpoint path under ``/api/v1``.  For example,
        ``'mvrv-zscore'`` or ``'lth-sopr'``.
    value_field : str
        Name of the JSON field that contains the metric value (e.g. 'mvrvZscore').
    days : int, optional
        If provided, returns only the last ``days`` observations.

    Returns
    -------
    DataFrame or None
        DataFrame indexed by datetime with a single ``value`` column, or
        ``None`` if the request fails or the field is missing.

    Notes
    -----
    The Bitcoin Data API from BGeometrics provides on‑chain metrics
    without requiring an API key.  Data is updated daily and rate‑limited.
    """
    base_url = f"https://bitcoin-data.com/api/v1/{endpoint}"
    try:
        resp = requests.get(base_url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        # Use 'unixTs' if present to build datetime index; fallback to 'd' (string date)
        if 'unixTs' in df.columns:
            df['t'] = pd.to_datetime(df['unixTs'], unit='s')
        elif 'd' in df.columns:
            df['t'] = pd.to_datetime(df['d'])
        else:
            return None
        df.set_index('t', inplace=True)
        if value_field not in df.columns:
            return None
        # Convert numeric values; some early rows may contain nulls
        df['value'] = pd.to_numeric(df[value_field], errors='coerce')
        df = df[['value']].dropna()
        if days is not None and len(df) > days:
            return df.tail(days)
        return df
    except Exception:
        return None


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mvrv_zscore(api_key: str = "", days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch MVRV Z‑Score from the BGeometrics API (last ``days`` observations)."""
    # endpoint returns full history; value field is 'mvrvZscore'
    return _bgeometrics_get("mvrv-zscore", "mvrvZscore", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_lth_sopr(api_key: str = "", days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch Long Term Holder SOPR from the BGeometrics API."""
    return _bgeometrics_get("lth-sopr", "lthSopr", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_reserve_risk(api_key: str = "", days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch Reserve Risk from the BGeometrics API."""
    return _bgeometrics_get("reserve-risk", "reserveRisk", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_exchange_inflows(api_key: str = "", days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch a proxy for exchange inflows using ETF BTC flows from the BGeometrics API.

    The free API does not provide total exchange inflow volume.  As a proxy,
    we use the daily BTC flows into exchange‑traded funds (ETFs) which
    aggregates flows from major spot Bitcoin ETFs.  This metric still
    reflects institutional demand and distribution patterns.

    Returns
    -------
    DataFrame or None
        DataFrame of daily ETF flows (BTC) with datetime index and ``value``
        column.  Only the last ``days`` observations are returned.
    """
    return _bgeometrics_get("etf-flow-btc", "etfFlow", days=days)


def fetch_btc_dominance() -> Optional[float]:
    """
    Fetch Bitcoin market dominance percentage from Coingecko global data.

    Returns
    -------
    float or None
        BTC dominance (0–100) or ``None`` if unavailable.
    """
    url = "https://api.coingecko.com/api/v3/global"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        dominance = (
            data.get('data', {})
            .get('market_cap_percentage', {})
            .get('btc')
        )
        # Fallback to uppercase key if necessary
        if dominance is None:
            dominance = (
                data.get('data', {})
                .get('market_cap_percentage', {})
                .get('BTC')
            )
        return float(dominance) if dominance is not None else None
    except Exception:
        return None


def fetch_funding_rate() -> Optional[float]:
    """
    Fetch the most recent funding rate for BTC perpetual futures from Binance.

    Funding rates are expressed as decimal fractions (e.g., 0.01 for 1%).
    Returns ``None`` if unavailable.
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": "BTCUSDT", "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            rate = float(data[0].get('fundingRate', 0))
            return rate
        return None
    except Exception:
        return None


def fetch_defi_tvl_growth(chain: str = "Ethereum", lookback_days: int = 30) -> Optional[float]:
    """
    Compute DeFi Total Value Locked (TVL) growth over a period.

    The function retrieves historical TVL data for a given chain from
    DefiLlama's API.  It then calculates the percentage growth over
    ``lookback_days``.  The API used to return a list of records but now
    sometimes returns a dictionary keyed by ``chainData``; both formats are
    supported.

    Parameters
    ----------
    chain : str
        Name of the chain.  Defaults to ``'Ethereum'``.
    lookback_days : int
        Number of days to look back.  Growth is computed from the difference
        between the most recent TVL and the value ``lookback_days`` earlier.

    Returns
    -------
    float or None
        Percentage growth over the period (0.25 means 25%).  Returns
        ``None`` if data cannot be fetched or the structure is unexpected.
    """
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # The API may return a list of records or a dict containing chainData
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Prefer 'chainData' key; fallback to 'data' or values()
            if isinstance(data.get('chainData'), list):
                records = data['chainData']
            elif isinstance(data.get('data'), list):
                records = data['data']
            else:
                # Some responses wrap lists under 'tvl' or other keys – try first list value
                first_list = None
                for val in data.values():
                    if isinstance(val, list):
                        first_list = val
                        break
                records = first_list if first_list is not None else None
        else:
            records = None
        if not isinstance(records, list) or not records:
            return None
        df = pd.DataFrame(records)
        if 'date' not in df.columns or 'tvl' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        # ensure enough data
        if len(df) < lookback_days + 1:
            return None
        latest = df['tvl'].iloc[-1]
        past = df['tvl'].iloc[-lookback_days - 1]
        if past == 0:
            return None
        growth = (latest - past) / past
        return float(growth)
    except Exception:
        return None


def fetch_treasury_yield() -> Optional[float]:
    """Fetch the current 10‑year US Treasury yield using yfinance."""
    try:
        df = yf.download("^TNX", period="5d", interval="1d", progress=False)
        if df.empty or 'Close' not in df.columns:
            return None
        latest = df['Close'].iloc[-1]
        yield_percent = float(latest) / 10.0
        return yield_percent
    except Exception:
        return None


def fetch_price_history(days: int | None = None) -> Optional[pd.DataFrame]:
    """
    Fetch historical Bitcoin price data from Coingecko, with yfinance fallback.

    Parameters
    ----------
    days : int or None
        Number of days to return.  If ``None``, returns the full history.

    Returns
    -------
    DataFrame or None
        Daily price history with datetime index and 'price' column.
    """
    # Attempt to fetch historical price data from Coingecko.
    range_param = days if days is not None else 'max'
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": range_param, "interval": "daily"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            prices = data.get('prices')
            if prices:
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
    except Exception:
        pass
    # Fallback to yfinance
    try:
        if days is None:
            period = 'max'
        else:
            # yfinance accepts strings like '1y', '2y', '5y'
            years = max(1, int(days // 365 + 1))
            period = f"{years}y"
        yf_df = yf.download('BTC-USD', period=period, interval='1d', progress=False)
        if yf_df.empty:
            return None
        yf_df = yf_df[['Close']].rename(columns={'Close': 'price'})
        yf_df.index = pd.to_datetime(yf_df.index)
        # If days is specified, slice the last ``days`` observations
        if days is not None and len(yf_df) > days:
            yf_df = yf_df.tail(days)
        return yf_df
    except Exception:
        return None


def compute_pi_cycle_cross(price_df: pd.DataFrame) -> Tuple[bool, bool, pd.DataFrame]:
    """
    Compute the Pi Cycle moving averages and detect crossovers.

    The Pi Cycle Top indicator compares a 111‑day simple moving average (SMA)
    of price with twice the 350‑day SMA.  When the shorter 111‑day SMA
    crosses above the 2×350‑day SMA, it historically marks major cycle tops.
    This function computes the SMAs and returns whether a cross has
    occurred, whether it is approaching (111SMA close to 2×350SMA),
    and a dataframe containing the moving averages for plotting.

    Parameters
    ----------
    price_df : DataFrame
        DataFrame with datetime index and 'price' column.

    Returns
    -------
    cross_occurred : bool
        True if the most recent 111SMA is above 2×350SMA.
    approaching : bool
        True if the difference is within 2% but not yet crossed.
    ma_df : DataFrame
        Dataframe containing columns '111sma' and '2x350sma' for plotting.
    """
    ma_df = pd.DataFrame(index=price_df.index)
    ma_df['111sma'] = price_df['price'].rolling(window=111).mean()
    ma_df['350sma'] = price_df['price'].rolling(window=350).mean()
    ma_df['2x350sma'] = ma_df['350sma'] * 2.0
    # latest values (drop rows with NaNs)
    valid = ma_df.dropna()
    if valid.empty:
        return False, False, ma_df[['111sma', '2x350sma']]
    latest = valid.iloc[-1]
    cross = bool(latest['111sma'] > latest['2x350sma'])
    # approaching: within 2% margin but not crossed
    approaching = False
    if not cross and latest['2x350sma'] != 0:
        pct_diff = abs(latest['2x350sma'] - latest['111sma']) / abs(latest['2x350sma'])
        approaching = pct_diff < 0.02
    return cross, approaching, ma_df[['111sma', '2x350sma']]


###############################################################################
# Dashboard display logic
###############################################################################

def display_metric_card(
    col: st.container,
    title: str,
    value: Optional[float],
    warning_threshold: Optional[float] = None,
    caution_threshold: Optional[float] = None,
    unit: str = '',
    higher_is_warning: bool = True,
) -> None:
    """
    Render a coloured metric card with optional warning and caution thresholds.

    Parameters
    ----------
    col : st.columns element
        Streamlit column in which to render the metric.
    title : str
        Title of the metric.
    value : float or None
        Current value of the metric.  Displays 'N/A' if None.
    warning_threshold : float, optional
        Value at which the metric triggers a high‑risk (red) warning.
    caution_threshold : float, optional
        Value at which the metric triggers a caution (yellow) state.
        Only applied if ``warning_threshold`` is also provided.  When
        ``higher_is_warning`` is True, values greater than or equal to
        ``warning_threshold`` are red, those between ``caution_threshold``
        and ``warning_threshold`` are yellow, and lower values are green.
        When ``higher_is_warning`` is False, the logic is inverted.
    unit : str
        Unit suffix to append to the value (e.g., '%').
    higher_is_warning : bool
        If True, values above the thresholds are considered risky; if False,
        values below the thresholds are considered risky.
    """
    if value is None or isinstance(value, (float, int)) and np.isnan(value):
        col.metric(title, "N/A")
        return
    # Format value
    try:
        val_float = float(value)
    except Exception:
        col.metric(title, "N/A")
        return
    display_val = f"{val_float:.3f}{unit}" if abs(val_float) >= 1e-3 else f"{val_float:.3e}{unit}"
    # Determine colour based on thresholds
    color = "#e5ffd5"  # default green tint
    if warning_threshold is not None:
        if higher_is_warning:
            # High‑risk if value >= warning_threshold
            if val_float >= warning_threshold:
                color = "#ffd5d5"  # red
            elif caution_threshold is not None and val_float >= caution_threshold:
                color = "#fff5cc"  # yellow
        else:
            # High‑risk if value <= warning_threshold
            if val_float <= warning_threshold:
                color = "#ffd5d5"
            elif caution_threshold is not None and val_float <= caution_threshold:
                color = "#fff5cc"
    # Render card using HTML
    col.markdown(
        f"<div style='padding:10px;border-radius:5px;background-color:{color};text-align:center'>"
        f"<b>{title}</b><br><span style='font-size:24px'>{display_val}</span></div>",
        unsafe_allow_html=True
    )


def main() -> None:
    st.set_page_config(page_title="Bitcoin Cycle Top Indicator", layout="wide")
    st.title("🚀 Bitcoin Cycle Top Indicator Dashboard")
    st.markdown(
        """
        This dashboard synthesises on‑chain metrics, exchange flows and
        macro indicators to highlight when bitcoin may be approaching
        a major cycle top.  Select your parameters in the sidebar and
        monitor the gauges below.  A **high confidence signal** occurs
        when at least three of the five primary indicators align.  The
        escalation levels summarise broader market conditions and help
        manage risk.
        """
    )

    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    st.sidebar.markdown(
        "<small>On‑chain metrics are sourced from the free BGeometrics API (no key required).</small>",
        unsafe_allow_html=True
    )
    api_key = ""  # placeholder for compatibility; ignored by fetch functions
    chain = st.sidebar.selectbox("DeFi chain for TVL", options=["Ethereum", "BSC", "Arbitrum", "Polygon"], index=0)
    lookback_days = st.sidebar.number_input("TVL growth lookback days", 7, 90, 30)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>Data sources: BGeometrics, Coingecko, Binance, DefiLlama, yfinance.</small>",
        unsafe_allow_html=True
    )

    # Coinglass API Key input for bull market peak signals
    coinglass_env = os.environ.get("COINGLASS_API_KEY", "")
    coinglass_key = st.sidebar.text_input(
        "Coinglass API Key",
        value=coinglass_env,
        type="password",
        help="Enter your Coinglass API key to fetch bull market peak indicators (optional).",
    )

    # Fetch data
    with st.spinner("Fetching data…"):
        mvrv_df = fetch_mvrv_zscore(api_key, days=730)
        lth_df = fetch_lth_sopr(api_key, days=730)
        reserve_df = fetch_reserve_risk(api_key, days=730)
        inflow_df = fetch_exchange_inflows(api_key, days=90)
        btc_dom = fetch_btc_dominance()
        funding_rate = fetch_funding_rate()
        defi_growth = fetch_defi_tvl_growth(chain, int(lookback_days))
        treasury_yield = fetch_treasury_yield()
        # Fetch ~500 days of price data to compute Pi Cycle moving averages (350‑day and 111‑day).
        price_df = fetch_price_history(days=500)
        # Coinglass bull market peak indicators
        coinglass_df = fetch_coinglass_peak_indicators(coinglass_key)
        # Compute Pi cycle
        pi_cross = (False, False, None)
        if price_df is not None and len(price_df) >= 350:
            pi_cross = compute_pi_cycle_cross(price_df)

    # Compute monthly ETF flows.  We return BTC flows by default and convert
    # to USD only if price history is available.
    monthly_inflow_usd: Optional[float] = None
    monthly_inflow_btc: Optional[float] = None
    if inflow_df is not None:
        if 'value' in inflow_df.columns:
            monthly_inflow_btc = float(inflow_df['value'].tail(30).sum())
        # Convert to USD if price data available
        if price_df is not None and not price_df.empty and monthly_inflow_btc is not None:
            # Resample price to daily and forward fill.  Convert the resulting Series into a DataFrame
            price_daily = price_df['price'].resample('1D').last().ffill().to_frame(name='price')
            # Join the inflow series with the daily price on the date index
            combined = inflow_df.join(price_daily, how='inner')
            if {'value', 'price'}.issubset(combined.columns) and not combined.empty:
                combined['usd'] = combined['value'] * combined['price']
                monthly_inflow_usd = float(combined['usd'].tail(30).sum())

    # Latest values for metrics
    latest_mvrv = None
    if mvrv_df is not None and not mvrv_df.empty and 'value' in mvrv_df.columns:
        latest_mvrv = float(mvrv_df['value'].iloc[-1])
    latest_lth = None
    if lth_df is not None and not lth_df.empty and 'value' in lth_df.columns:
        latest_lth = float(lth_df['value'].iloc[-1])
    latest_reserve = None
    if reserve_df is not None and not reserve_df.empty and 'value' in reserve_df.columns:
        latest_reserve = float(reserve_df['value'].iloc[-1])
    # If USD conversion is available, display in billions USD; else display in BTC (divided by 1e3 to show thousands)
    if monthly_inflow_usd is not None:
        latest_inflow = monthly_inflow_usd / 1e9  # billions USD
        inflow_unit = "B"
        inflow_warning, inflow_caution = 10.0, 5.0
    elif monthly_inflow_btc is not None:
        # Show BTC flows scaled to thousands to keep numbers manageable
        latest_inflow = monthly_inflow_btc / 1e3  # thousands BTC
        inflow_unit = "k BTC"
        # Approximate caution at ~165k BTC (5B/30k ≈ 166k)
        inflow_warning, inflow_caution = 330.0, 165.0
    else:
        latest_inflow = None
        inflow_unit = ""
        inflow_warning = None
        inflow_caution = None
    # Pi cycle cross results
    pi_cross_occurred, pi_approaching, ma_df = pi_cross

    # Determine high confidence alignment
    signals: list[str] = []
    if latest_mvrv is not None and latest_mvrv > 6.0:
        signals.append("MVRV Z > 6")
    if latest_lth is not None and latest_lth > 8.0:
        signals.append("LTH‑SOPR > 8")
    if pi_cross_occurred or pi_approaching:
        signals.append("Pi Cycle approaching/cross")
    if latest_inflow is not None and latest_inflow > 10.0:
        signals.append("Exchange inflows > $10B")
    if latest_reserve is not None and latest_reserve > 0.015:
        signals.append("Reserve Risk > 0.015")
    high_confidence = len(signals) >= 3

    # Layout: metrics row
    cols = st.columns(5)
    # MVRV Z‑Score: caution at >5, high risk at >6 (extreme)
    display_metric_card(cols[0], "MVRV Z‑Score", latest_mvrv, warning_threshold=6.0, caution_threshold=5.0, unit="", higher_is_warning=True)
    # LTH‑SOPR: caution at >5, high risk at >8
    display_metric_card(cols[1], "LTH‑SOPR", latest_lth, warning_threshold=8.0, caution_threshold=5.0, unit="", higher_is_warning=True)
    # Pi Cycle status: represent the ratio of difference to 2×350SMA.  Positive implies cross occurred, negative implies not yet.
    pi_status_val = None
    if price_df is not None and ma_df is not None and not ma_df.empty:
        last_2x = ma_df['2x350sma'].iloc[-1]
        last_111 = ma_df['111sma'].iloc[-1]
        if last_2x != 0:
            pi_status_val = (last_111 - last_2x) / abs(last_2x)
    # caution when approaching cross (within -2% to 0), high risk when cross occurs (>=0)
    display_metric_card(
        cols[2],
        "Pi Cycle Status",
        pi_status_val,
        warning_threshold=0.0,
        caution_threshold=-0.02,
        unit="",
        higher_is_warning=True,
    )
    # Exchange inflow: thresholds determined above
    display_metric_card(
        cols[3],
        "Exchange inflow (30d)",
        latest_inflow,
        warning_threshold=inflow_warning,
        caution_threshold=inflow_caution,
        unit=inflow_unit,
        higher_is_warning=True,
    )
    # Reserve Risk: caution at >0.01, high risk at >0.015
    display_metric_card(cols[4], "Reserve Risk", latest_reserve, warning_threshold=0.015, caution_threshold=0.01, unit="", higher_is_warning=True)

    # Additional metrics row
    cols2 = st.columns(4)
    # BTC dominance: caution if >60%, high risk if >65%
    display_metric_card(cols2[0], "BTC dominance", btc_dom, warning_threshold=65.0, caution_threshold=60.0, unit="%", higher_is_warning=True)
    # Funding rate: caution at >0.05%, high risk at >0.1%
    display_metric_card(cols2[1], "Funding rate", funding_rate, warning_threshold=0.1, caution_threshold=0.05, unit="", higher_is_warning=True)
    # DeFi TVL growth: caution at >20%, high risk at >25%
    display_metric_card(cols2[2], f"DeFi TVL {chain} growth {int(lookback_days)}d", defi_growth, warning_threshold=0.25, caution_threshold=0.20, unit="", higher_is_warning=True)
    # 10Y Treasury yield: caution at >4.0%, high risk at >4.5%
    display_metric_card(cols2[3], "10Y Treasury yield", treasury_yield, warning_threshold=4.5, caution_threshold=4.0, unit="%", higher_is_warning=True)

    # High confidence signal display
    if high_confidence:
        st.success(f"High Confidence Signal: {len(signals)}/5 metrics align → {', '.join(signals)}")
    else:
        st.info(f"Signals aligning: {len(signals)}/5 → {', '.join(signals) if signals else 'None'}")

    # Escalation levels based on approximate price and metrics
    current_price = None
    if price_df is not None and not price_df.empty:
        current_price = float(price_df['price'].iloc[-1])
    st.markdown("---")
    st.subheader("Escalation Levels")
    level: Optional[int] = None
    if current_price is not None:
        # Level 4: price >135k and cross and MVRV >7 and funding rate >1%
        cond4 = (
            current_price > 135_000 and
            latest_mvrv is not None and latest_mvrv > 7.0 and
            pi_cross_occurred and
            funding_rate is not None and funding_rate > 1.0
        )
        # Level 3: price >130k and MVRV >6 and btc dom <60 and funding rate >0.1 and at least two signals
        cond3 = (
            current_price > 130_000 and
            latest_mvrv is not None and latest_mvrv > 6.0 and
            btc_dom is not None and btc_dom < 60.0 and
            funding_rate is not None and funding_rate > 0.1 and
            len(signals) >= 2
        )
        # Level 2: price >120k and MVRV >5 and BTC dom declining (proxy: btc_dom < 65) and funding rate >0.01
        cond2 = (
            current_price > 120_000 and
            latest_mvrv is not None and latest_mvrv > 5.0 and
            btc_dom is not None and btc_dom < 65.0 and
            funding_rate is not None and funding_rate > 0.01
        )
        # Level 1: price >110k and BTC dom >65 and funding >0.05 and defi growth >25% and treasury yield >4.5
        cond1 = (
            current_price > 110_000 and
            btc_dom is not None and btc_dom > 65.0 and
            funding_rate is not None and funding_rate > 0.05 and
            defi_growth is not None and defi_growth > 0.25 and
            treasury_yield is not None and treasury_yield > 4.5
        )
        if cond4:
            level = 4
        elif cond3:
            level = 3
        elif cond2:
            level = 2
        elif cond1:
            level = 1

    level_descriptions: dict[int, str] = {
        1: (
            "**Level 1 – Early Warning ($110K–$120K)**\n\n"
            "• BTC dominance peaks >65%\n\n"
            "• Funding rates >0.05%\n\n"
            "• DeFi TVL growth >25% monthly\n\n"
            "• 10Y Treasury yield approaching 4.5%\n"
        ),
        2: (
            "**Level 2 – Caution Zone ($120K–$130K)**\n\n"
            "• MVRV Z‑Score >5.0\n\n"
            "• BTC dominance declining from peak\n\n"
            "• Hash rate plateau signals\n\n"
            "• Put/call ratios <0.5\n"
        ),
        3: (
            "**Level 3 – High Risk ($130K–$135K)**\n\n"
            "• MVRV Z‑Score >6.0\n\n"
            "• BTC dominance <60%\n\n"
            "• Funding rates >0.1%\n\n"
            "• Multiple indicators firing\n"
        ),
        4: (
            "**Level 4 – Extreme Danger (>$135K)**\n\n"
            "• Price >$135K and Pi Cycle cross\n\n"
            "• MVRV Z‑Score >7.0\n\n"
            "• Funding rates >1%\n\n"
            "• Escalated sell‑off risk\n"
        ),
    }
    if level is not None:
        st.warning(level_descriptions.get(level, ""), icon="⚠️")
    else:
        st.info("No escalation level triggered.")

    # Plotting MVRV, LTH SOPR and Reserve Risk
    st.markdown("---")
    st.subheader("On‑chain Metrics Over Time")
    chart_df = pd.DataFrame()
    if mvrv_df is not None and 'value' in mvrv_df.columns:
        chart_df['MVRV Z'] = mvrv_df['value']
    if lth_df is not None and 'value' in lth_df.columns:
        chart_df['LTH SOPR'] = lth_df['value']
    if reserve_df is not None and 'value' in reserve_df.columns:
        chart_df['Reserve Risk'] = reserve_df['value']
    if not chart_df.empty:
        fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, labels={"value": "Value", "index": "Date"}, title="On‑chain Indicators")
        fig.update_layout(legend_title_text='Metric', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("On‑chain data not available. This may occur due to API rate limits or format changes.")

    # Plot Pi cycle moving averages if available
    if price_df is not None and ma_df is not None and not ma_df.dropna().empty:
        st.subheader("Pi Cycle Moving Averages")
        plot_df = ma_df.dropna()
        fig2 = px.line(plot_df, x=plot_df.index, y=['111sma', '2x350sma'], labels={"value": "Price", "index": "Date"}, title="111‑day SMA vs 2×350‑day SMA")
        fig2.update_layout(legend_title_text='MA', height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Display Coinglass Bull Market Peak Signals if available
    if coinglass_df is not None and not coinglass_df.empty:
        st.markdown("---")
        st.subheader("Coinglass Bull Market Peak Signals")
        df = coinglass_df.copy()
        # Determine column names for indicator name and hit flags (case‑insensitive search)
        name_col = None
        hit_col = None
        for col_name in df.columns:
            lower = col_name.lower()
            if lower in {"name", "indicname", "indicator"}:
                name_col = col_name
            if lower in {"hit", "triggered", "ishit"}:
                hit_col = col_name
        if name_col is None:
            name_col = df.columns[0]
        if hit_col is None and 'hit' in df.columns:
            hit_col = 'hit'
        # Count triggered signals
        triggered = None
        if hit_col is not None:
            # Convert to boolean if values are strings
            triggered = df[hit_col].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
            st.write(f"Indicators triggered: {triggered}/{len(df)}")
        st.dataframe(df)

    # Show raw signals for reference
    st.markdown("---")
    st.subheader("Critical Success Factors")
    st.markdown(
        """
        * **Never rely on a single indicator:** require at least three signals (MVRV Z, LTH‑SOPR, Pi Cycle, exchange inflows, reserve risk) for high confidence.
        * **Monitor daily:** track MVRV progression, exchange flows and funding rates each day to catch rapid changes.
        * **Historical timing:** prior cycle tops have occurred ~1,060 days from the cycle low, suggesting the next major top could emerge around **September 2025** (although the ETF era may alter patterns).
        * **Risk management:** reduce positions as the escalation levels increase; levels 3 and 4 warrant aggressive de‑risking.
        * **Market evolution:** recognise that ETF inflows and changing macro conditions can shift the behaviour of these metrics.
        """
    )


if __name__ == "__main__":
    main()