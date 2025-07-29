"""
Bitcoin Cycle Top Indicator Dashboard
===================================

This Streamlit application aggregates on‑chain, exchange, macro and
technical data to help identify bitcoin market cycle tops. It fetches
metrics from several third‑party APIs, computes trend indicators and
displays them on an interactive dashboard. The on‑chain metrics
(MVRV Z‑Score, Long Term Holder SOPR and Reserve Risk) are obtained
from the free **BGeometrics Bitcoin Data API**, which does not
require an API key and is rate‑limited. Other data are pulled from
public sources like Coingecko (Bitcoin dominance and price), Binance
(funding rate), DefiLlama (DeFi TVL), and yfinance (10‑year Treasury
yield).

High confidence signals for cycle tops emerge when at least three
conditions align: an elevated MVRV Z‑Score (>6), surging long‑term
holder SOPR (>8), a looming Pi Cycle MA cross, high exchange inflows
($10B+ monthly) and rising reserve risk (>0.015). Escalation levels
capture broader market context such as BTC dominance, funding rates,
DeFi capital flows and macro yields. When multiple warning levels
activate, the dashboard highlights the increased probability of a
cycle peak.

This code is intended to run in a cloud environment. Ensure that
`streamlit run app.py` is executed from the project root. No API key
is required for on‑chain metrics because they come from the free
BGeometrics API. Other data are pulled from public sources.
"""

import os
import time
import datetime as dt
from typing import Optional, Tuple, Dict

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
    ``name``, ``value``, ``targetValue`` and ``hit`` (whether the target is met).

    Parameters
    ----------
    api_key : str
        Your Coinglass API key. Must be passed via the ``CG-API-KEY`` header.

    Returns
    -------
    DataFrame or None
        DataFrame containing indicator data or None if the request fails or
        the key is missing.
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
        # The response structure may vary; attempt to extract the list of indicators
        items = None
        if isinstance(data, dict):
            # v3/v4 API may wrap data under 'data', 'dataList' or 'result'
            for key in ["data", "dataList", "result", "resultList"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            # sometimes the list is nested under data->list
            if items is None and 'data' in data and isinstance(data['data'], dict):
                for subkey in ["list", "items"]:
                    if subkey in data['data'] and isinstance(data['data'][subkey], list):
                        items = data['data'][subkey]
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

def _bgeometrics_get(endpoint: str, value_field: str, days: int = None) -> Optional[pd.DataFrame]:
    """
    Fetch a timeseries metric from the free BGeometrics on‑chain API.

    Parameters
    ----------
    endpoint : str
        The metric endpoint path under ``/api/v1``. For example,
        ``'mvrv-zscore'`` or ``'lth-sopr'``.
    value_field : str
        Name of the JSON field that contains the metric value (e.g. 'mvrvZscore').
    days : int, optional
        If provided, returns only the last `days` observations.

    Returns
    -------
    DataFrame or None
        DataFrame indexed by datetime with a single 'value' column, or
        None if the request fails.

    Notes
    -----
    The Bitcoin Data API from BGeometrics provides on‑chain metrics
    without requiring an API key. Data is updated daily and rate‑limited.
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
    """
    Fetch MVRV Z‑Score from the BGeometrics API.

    The MVRV Z‑Score compares Bitcoin's market value to its realized value
    and normalizes by the standard deviation of market value, identifying
    periods where the price is extremely overvalued or undervalued【475926368150572†L216-L244】.

    Returns the last `days` observations.
    """
    # endpoint returns full history; value field is 'mvrvZscore'
    return _bgeometrics_get("mvrv-zscore", "mvrvZscore", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_lth_sopr(api_key: str = "", days: int = 730) -> Optional[pd.DataFrame]:
    """
    Fetch Long Term Holder SOPR (spent output profit ratio) from the BGeometrics API.

    LTH‑SOPR filters UTXOs older than 155 days and measures realised profit
    and loss only for coins moved on‑chain that have a lifespan more than
    155 days【474942229923549†L79-L83】. Values above 1 indicate holders selling at a profit.
    """
    return _bgeometrics_get("lth-sopr", "lthSopr", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_reserve_risk(api_key: str = "", days: int = 730) -> Optional[pd.DataFrame]:
    """
    Fetch Reserve Risk from the BGeometrics API【474942229923549†L90-L92】.

    Reserve Risk measures the conviction of long‑term holders relative to
    the price; high values (>0.015) indicate declining confidence and
    potential for market tops【118412254918152†L219-L248】.
    """
    return _bgeometrics_get("reserve-risk", "reserveRisk", days=days)


# Cache for a full day to stay well within the free BGeometrics rate limit
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_exchange_inflows(api_key: str = "", days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch proxy for exchange inflows using ETF BTC flow from the BGeometrics API.

    The free API does not provide total exchange inflow volume. As a proxy,
    we use the daily BTC flows into exchange‑traded funds (ETFs) which
    aggregates flows from major spot Bitcoin ETFs. This metric still
    reflects institutional demand and distribution patterns.

    Returns
    -------
    DataFrame or None
        DataFrame of daily ETF flows (BTC) with datetime index and 'value'
        column. Only the last `days` observations are returned.
    """
    return _bgeometrics_get("etf-flow-btc", "etfFlow", days=days)


def fetch_btc_dominance() -> Optional[float]:
    """Fetch Bitcoin market dominance percentage from Coingecko global data【644653073643528†L40-L43】.
    Returns a float representing BTC dominance (0‑100) or None if unavailable.
    """
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc')
        if dominance is None:
            return None
        return float(dominance)
    except Exception:
        return None


def fetch_funding_rate() -> Optional[float]:
    """Fetch the most recent funding rate for BTC perpetual futures from Binance【789894824174857†L93-L136】.

    Funding rates are expressed as decimal fractions (e.g., 0.01 for 1%).
    Returns None if unavailable.
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
    """Compute DeFi Total Value Locked (TVL) growth over a period.

    The function retrieves historical TVL data for a given chain from
    DefiLlama's v2 API. It then calculates the percentage growth over
    `lookback_days`. The API returns daily TVL values measured in USD.

    Parameters
    ----------
    chain : str
        Name of the chain. Defaults to ``'Ethereum'``.
    lookback_days : int
        Number of days to look back. Growth is computed from the
        difference between the most recent TVL and the value
        `lookback_days` earlier.

    Returns
    -------
    float or None
        Percentage growth over the period (0.25 means 25%). Returns
        None if data cannot be fetched.
    """
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        if 'date' not in df.columns or 'tvl' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        # ensure enough data
        if len(df) < lookback_days + 1:
            return None
        latest = df.iloc[-1]['tvl']
        past = df.iloc[-lookback_days - 1]['tvl']
        if past == 0:
            return None
        growth = (latest - past) / past
        return float(growth)
    except Exception:
        return None


def fetch_treasury_yield() -> Optional[float]:
    """Fetch the current 10‑year US Treasury yield using yfinance.

    The ^TNX ticker on Yahoo Finance quotes the CBOE 10‑Year Treasury Note
    Yield Index in terms of percentage points times 10. Therefore, a
    closing price of 45 represents a yield of 4.5%. Returns the
    latest closing yield as a percentage (e.g., 4.5).
    """
    try:
        df = yf.download("^TNX", period="5d", interval="1d", progress=False)
        if df.empty or 'Close' not in df.columns:
            return None
        latest = df['Close'].iloc[-1]
        yield_percent = float(latest) / 10.0
        return yield_percent
    except Exception:
        return None


def fetch_price_history(days: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Fetch historical Bitcoin price data from Coingecko.

    Parameters
    ----------
    days : int or None
        Number of days to return. If None, returns the full history.

    Returns
    -------
    DataFrame or None
        Daily price history with datetime index and 'price' column.
    """
    # Attempt to fetch historical price data from Coingecko. If it fails,
    # fallback to yfinance (BTC‑USD). The `days` parameter controls how many
    # days of history to return when using Coingecko; for yfinance we
    # download the full history and slice.
    range_param = days if days is not None else 'max'
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
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
        # Determine lookback period for yfinance
        period = None
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
        # If days is specified, slice the last `days` observations
        if days is not None and len(yf_df) > days:
            yf_df = yf_df.tail(days)
        return yf_df
    except Exception:
        return None


def compute_pi_cycle_cross(price_df: pd.DataFrame) -> Tuple[bool, bool, pd.DataFrame]:
    """Compute the Pi Cycle moving averages and detect crossovers.

    The Pi Cycle Top indicator compares a 111‑day simple moving average (SMA)
    of price with twice the 350‑day SMA. When the shorter 111‑day SMA
    crosses above the 2×350‑day SMA, it historically marks major cycle
    tops. This function computes the SMAs and returns whether a cross
    has occurred, whether it is approaching (111SMA close to 2×350SMA),
    and a dataframe containing the moving averages for plotting.

    Parameters
    ----------
    price_df : DataFrame
        DataFrame with datetime index and 'price' column.

    Returns
    -------
    (bool, bool, DataFrame)
        A tuple of (cross_occurred, approaching, ma_df). The boolean
        `cross_occurred` is True if the most recent 111SMA is above
        2×350SMA. `approaching` is True if the difference is small but
        not yet crossed. `ma_df` contains columns '111sma' and '2x350sma'.
    """
    ma_df = pd.DataFrame(index=price_df.index)
    ma_df['111sma'] = price_df['price'].rolling(window=111).mean()
    ma_df['350sma'] = price_df['price'].rolling(window=350).mean()
    ma_df['2x350sma'] = ma_df['350sma'] * 2.0
    # latest values
    latest = ma_df.dropna().iloc[-1]
    cross = bool(latest['111sma'] > latest['2x350sma']) if not np.isnan(latest['111sma']) and not np.isnan(latest['2x350sma']) else False
    # approaching: within 1% margin but not crossed
    approaching = False
    if not cross and not np.isnan(latest['111sma']) and not np.isnan(latest['2x350sma']):
        diff = latest['2x350sma'] - latest['111sma']
        if latest['2x350sma'] != 0:
            pct_diff = abs(diff) / latest['2x350sma']
            approaching = pct_diff < 0.02  # within 2%
    return cross, approaching, ma_df[['111sma', '2x350sma']]


###############################################################################
# Dashboard display logic
###############################################################################

def display_metric_card(
    col,
    title: str,
    value: Optional[float],
    warning_threshold: Optional[float] = None,
    caution_threshold: Optional[float] = None,
    unit: str = '',
    higher_is_warning: bool = True,
):
    """
    Render a coloured metric card with optional warning and caution thresholds.

    Parameters
    ----------
    col : st.columns element
        Streamlit column in which to render the metric.
    title : str
        Title of the metric.
    value : float or None
        Current value of the metric. Displays 'N/A' if None.
    warning_threshold : float, optional
        Value at which the metric triggers a high‑risk (red) warning.
    caution_threshold : float, optional
        Value at which the metric triggers a caution (yellow) state. Only
        applied if ``warning_threshold`` is also provided. When
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
    if value is None:
        col.metric(title, "N/A")
        return
    display_val = f"{value:.3f}{unit}" if abs(value) >= 1e-3 else f"{value:.3e}{unit}"
    # Determine colour based on thresholds
    color = "#e5ffd5"  # default green tint
    if warning_threshold is not None:
        if higher_is_warning:
            # High‑risk if value >= warning_threshold
            if value >= warning_threshold:
                color = "#ffd5d5"  # red
            elif caution_threshold is not None and value >= caution_threshold:
                color = "#fff5cc"  # yellow
        else:
            # High‑risk if value <= warning_threshold
            if value <= warning_threshold:
                color = "#ffd5d5"
            elif caution_threshold is not None and value <= caution_threshold:
                color = "#fff5cc"
    # Render card
    col.markdown(
        f"<div style='padding:10px;border-radius:5px;background-color:{color};text-align:center'>"
        f"<b>{title}</b><br><span style='font-size:24px'>{display_val}</span></div>",
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Bitcoin Cycle Top Indicator", layout="wide")
    st.title("🚀 Bitcoin Cycle Top Indicator Dashboard")
    st.markdown(
        """
        This dashboard synthesises on‑chain metrics, exchange flows and
        macro indicators to highlight when bitcoin may be approaching
        a major cycle top. Select your parameters in the sidebar and
        monitor the gauges below. A **high confidence signal** occurs
        when at least three of the five primary indicators align. The
        escalation levels summarise broader market conditions and help
        manage risk.
        """
    )

    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    # On‑chain metrics from BGeometrics do not require an API key, so we hide
    # the Glassnode API input. If a GLASSNODE_API_KEY environment
    # variable is set, it is ignored.
    st.sidebar.markdown("<small>On‑chain metrics are sourced from the free BGeometrics API (no key required).</small>", unsafe_allow_html=True)
    api_key = ""  # placeholder for compatibility; ignored by fetch functions
    chain = st.sidebar.selectbox("DeFi chain for TVL", options=["Ethereum", "BSC", "Arbitrum", "Polygon"], index=0)
    lookback_days = st.sidebar.number_input("TVL growth lookback days", 7, 90, 30)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<small>Data sources: BGeometrics, Coingecko, Binance, DefiLlama, yfinance.</small>", unsafe_allow_html=True)

    # Coinglass API Key input for bull market peak signals
    coinglass_env = os.environ.get("COINGLASS_API_KEY", "")
    coinglass_key = st.sidebar.text_input(
        "Coinglass API Key", value=coinglass_env, type="password",
        help="Enter your Coinglass API key to fetch bull market peak indicators (optional)."
    )

    # Fetch data
    with st.spinner("Fetching data…"):
        mvrv_df = fetch_mvrv_zscore(api_key, days=730)
        lth_df = fetch_lth_sopr(api_key, days=730)
        reserve_df = fetch_reserve_risk(api_key, days=730)
        inflow_df = fetch_exchange_inflows(api_key, days=90)
        btc_dom = fetch_btc_dominance()
        funding_rate = fetch_funding_rate()
        defi_growth = fetch_defi_tvl_growth(chain, lookback_days)
        treasury_yield = fetch_treasury_yield()
        # Fetch ~500 days of price data to compute Pi Cycle moving averages (350‑day and 111‑day).
        price_df = fetch_price_history(days=500)
        # Coinglass bull market peak indicators
        coinglass_df = fetch_coinglass_peak_indicators(coinglass_key)

        # compute Pi cycle
        pi_cross = (False, False, None)
        if price_df is not None and len(price_df) >= 350:
            pi_cross = compute_pi_cycle_cross(price_df)

    # Compute monthly ETF flows. We return BTC flows by default and convert
    # to USD only if price history is available.
    monthly_inflow_usd = None
    monthly_inflow_btc = None
    if inflow_df is not None:
        # Sum the last 30 days of ETF flows in BTC
        monthly_inflow_btc = inflow_df['value'].tail(30).sum()
        if price_df is not None and not price_df.empty:
            # Resample price to daily and forward fill. Convert the resulting
            # Series into a DataFrame so it retains a column name when joined.
            price_daily = (
                price_df['price']
                .resample('1D')
                .last()
                .ffill()
                .to_frame(name='price')
            )
            # Join the inflow series with the daily price on the date index
            combined = inflow_df.join(price_daily, how='inner')
            # Only perform multiplication if both required columns are present.
            # Without this guard, combined['price'] may not exist and raise a
            # KeyError/ValueError.
            if {'value', 'price'}.issubset(combined.columns):
                combined['usd'] = combined['value'] * combined['price']
                monthly_inflow_usd = combined['usd'].tail(30).sum()

    # Latest values for metrics
    latest_mvrv = mvrv_df.iloc[-1]['value'] if mvrv_df is not None and not mvrv_df.empty else None
    latest_lth = lth_df.iloc[-1]['value'] if lth_df is not None and not lth_df.empty else None
    latest_reserve = reserve_df.iloc[-1]['value'] if reserve_df is not None and not reserve_df.empty else None
    # If USD conversion is available, display in billions USD; else display in BTC (divided by 1e3 to show thousands)
    if monthly_inflow_usd is not None:
        latest_inflow = monthly_inflow_usd / 1e9  # billions USD
    elif monthly_inflow_btc is not None:
        # Show BTC flows scaled to thousands to keep numbers manageable
        latest_inflow = monthly_inflow_btc / 1e3  # thousands BTC
    else:
        latest_inflow = None
    # Pi cycle cross results
    pi_cross_occured, pi_approaching, ma_df = pi_cross

    # Determine high confidence alignment
    signals = []
    if latest_mvrv is not None and latest_mvrv > 6.0:
        signals.append("MVRV Z > 6")
    if latest_lth is not None and latest_lth > 8.0:
        signals.append("LTH‑SOPR > 8")
    if pi_cross_occured or pi_approaching:
        signals.append("Pi Cycle approaching/cross")
    if latest_inflow is not None and latest_inflow > 10.0:
        signals.append("Exchange inflows > $10B")
    if latest_reserve is not None and latest_reserve > 0.015:
        signals.append("Reserve Risk > 0.015")

    high_confidence = len(signals) >= 3

    # Layout: metrics row
    cols = st.columns(5)
    # Conservative thresholds: caution (yellow) and warning (red)
    # MVRV Z‑Score: caution at >5, high risk at >6 (extreme)
    display_metric_card(cols[0], "MVRV Z‑Score", latest_mvrv, warning_threshold=6.0, caution_threshold=5.0, unit="", higher_is_warning=True)
    # LTH‑SOPR: caution at >5, high risk at >8
    display_metric_card(cols[1], "LTH‑SOPR", latest_lth, warning_threshold=8.0, caution_threshold=5.0, unit="", higher_is_warning=True)
    # For Pi cycle: show difference or status as numeric (1 for cross, 0 for approaching, negative for far)
    pi_status_val = None
    if price_df is not None and ma_df is not None:
        diff = ma_df.iloc[-1]['111sma'] - ma_df.iloc[-1]['2x350sma']
        # Represent status: positive implies cross occurred, negative implies not yet
        pi_status_val = diff / ma_df.iloc[-1]['2x350sma'] if ma_df.iloc[-1]['2x350sma'] != 0 else None
    # Pi Cycle status: caution when approaching cross (within -2% to 0), high risk when cross occurs (>=0)
    # Represent status ratio; invert: higher values mean cross occurred; negative values indicate below cross.
    display_metric_card(cols[2], "Pi Cycle Status", pi_status_val, warning_threshold=0.0, caution_threshold=-0.02, unit="", higher_is_warning=True)
    # Determine threshold and unit based on whether we have USD or BTC flows
    if monthly_inflow_usd is not None:
        inflow_warning = 10.0  # $10B
        inflow_caution = 5.0   # $5B
        inflow_unit = "B"
    elif monthly_inflow_btc is not None:
        # Approximate caution at ~165k BTC (5B/30k = 166k) → 165 (thousand)
        inflow_warning = 330.0  # thousands BTC for $10B
        inflow_caution = 165.0  # thousands BTC for $5B
        inflow_unit = "k BTC"
    else:
        inflow_warning = None
        inflow_caution = None
        inflow_unit = ""
    display_metric_card(cols[3], "Exchange inflow (30d)", latest_inflow, warning_threshold=inflow_warning, caution_threshold=inflow_caution, unit=inflow_unit, higher_is_warning=True)
    # Reserve Risk: caution at >0.01, high risk at >0.015
    display_metric_card(cols[4], "Reserve Risk", latest_reserve, warning_threshold=0.015, caution_threshold=0.01, unit="", higher_is_warning=True)

    # Additional metrics row
    cols2 = st.columns(4)
    # BTC dominance: caution if >60%, high risk if >65%
    display_metric_card(cols2[0], "BTC dominance", btc_dom, warning_threshold=65.0, caution_threshold=60.0, unit="%", higher_is_warning=True)
    # Funding rate: caution at >0.05%, high risk at >0.1%
    display_metric_card(cols2[1], "Funding rate", funding_rate, warning_threshold=0.1, caution_threshold=0.05, unit="", higher_is_warning=True)
    # DeFi TVL growth: caution at >20%, high risk at >25%
    display_metric_card(cols2[2], f"DeFi TVL {chain} growth {lookback_days}d", defi_growth, warning_threshold=0.25, caution_threshold=0.20, unit="", higher_is_warning=True)
    # 10Y Treasury yield: caution at >4.0%, high risk at >4.5%
    display_metric_card(cols2[3], "10Y Treasury yield", treasury_yield, warning_threshold=4.5, caution_threshold=4.0, unit="%", higher_is_warning=True)

    # High confidence signal display
    if high_confidence:
        st.success(f"High Confidence Signal: {len(signals)}/5 metrics align → {', '.join(signals)}")
    else:
        st.info(f"Signals aligning: {len(signals)}/5 → {', '.join(signals) if signals else 'None'}")

    # Escalation levels based on approximate price and metrics
    # Determine current price from price_df
    current_price = price_df.iloc[-1]['price'] if price_df is not None else None
    st.markdown("---")
    st.subheader("Escalation Levels")
    level = None
    # Define conditions for each level
    if current_price is not None:
        # Level 4: price >135k and cross and MVRV >7 and funding rate >1
        cond4 = (current_price > 135000) and (latest_mvrv is not None and latest_mvrv > 7.0) and (pi_cross_occured) and (funding_rate is not None and funding_rate > 1.0)
        # Level 3: price >130k and MVRV >6 and btc dom <60 and funding rate >0.1 and at least two signals
        cond3 = (current_price > 130000) and (latest_mvrv is not None and latest_mvrv > 6.0) and (btc_dom is not None and btc_dom < 60.0) and (funding_rate is not None and funding_rate > 0.1) and (len(signals) >= 2)
        # Level 2: price >120k and MVRV >5 and BTC dom declining (proxy: btc_dom < 65) and funding rate < 0.05 or call ratio not available
        cond2 = (current_price > 120000) and (latest_mvrv is not None and latest_mvrv > 5.0) and (btc_dom is not None and btc_dom < 65.0) and (funding_rate is not None and funding_rate > 0.01)
        # Level 1: price >110k and BTC dom >65 and funding >0.05 and defi growth >25% and treasury yield >4.5
        cond1 = (current_price > 110000) and (btc_dom is not None and btc_dom > 65.0) and (funding_rate is not None and funding_rate > 0.05) and (defi_growth is not None and defi_growth > 0.25) and (treasury_yield is not None and treasury_yield > 4.5)
        if cond4:
            level = 4
        elif cond3:
            level = 3
        elif cond2:
            level = 2
        elif cond1:
            level = 1

    # Describe escalation levels
    level_descriptions = {
        1: "**Level 1 – Early Warning ($110K–$120K)**\n\n"
           "• BTC dominance peaks >65%\n\n"
           "• Funding rates >0.05%\n\n"
           "• DeFi TVL growth >25% monthly\n\n"
           "• 10Y Treasury yield approaching 4.5%\n",
        2: "**Level 2 – Caution Zone ($120K–$130K)**\n\n"
           "• MVRV Z‑Score >5.0\n\n"
           "• BTC dominance declining from peak\n\n"
           "• Hash rate plateau signals\n\n"
           "• Put/call ratios <0.5\n",
        3: "**Level 3 – High Risk ($130K–$135K)**\n\n"
           "• MVRV Z‑Score >6.0\n\n"
           "• BTC dominance <60%\n\n"
           "• Funding rates >0.1%\n\n"
           "• Multiple indicators aligning\n",
        4: "**Level 4 – Extreme Danger (>$135K)**\n\n"
           "• MVRV Z‑Score >7.0\n\n"
           "• Pi Cycle Top cross confirmed\n\n"
           "• Funding rates >1.0%\n\n"
           "• All systems flashing red\n",
    }

    if level:
        st.error(f"Escalation Level {level} activated")
        st.markdown(level_descriptions[level])
    else:
        st.success("Escalation: No immediate danger detected.")
        st.markdown("\n".join([desc for desc in level_descriptions.values()]))

    # Plotting MVRV, LTH SOPR and Reserve Risk
    st.markdown("---")
    st.subheader("On‑chain Metrics Over Time")
    chart_df = pd.DataFrame()
    if mvrv_df is not None:
        chart_df['MVRV Z'] = mvrv_df['value']
    if lth_df is not None:
        chart_df['LTH SOPR'] = lth_df['value']
    if reserve_df is not None:
        chart_df['Reserve Risk'] = reserve_df['value']
    if not chart_df.empty:
        fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, labels={"value": "Value", "index": "Date"},
                      title="On‑chain Indicators")
        fig.update_layout(legend_title_text='Metric', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("On‑chain data not available. This may occur due to API rate limits.")

    # Plot Pi cycle moving averages if available
    if price_df is not None and ma_df is not None:
        st.subheader("Pi Cycle Moving Averages")
        fig2 = px.line(ma_df.dropna(), x=ma_df.dropna().index, y=['111sma', '2x350sma'], labels={"value": "Price", "index": "Date"},
                       title="111‑day SMA vs 2×350‑day SMA")
        fig2.update_layout(legend_title_text='MA', height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Display Coinglass Bull Market Peak Signals if available
    if coinglass_df is not None and not coinglass_df.empty:
        st.markdown("---")
        st.subheader("Coinglass Bull Market Peak Signals")
        # Show number of indicators triggered (hit == True) and list
        # Normalise column names if needed
        df = coinglass_df.copy()
        # Determine column names for indicator name and hit
        name_col = None
        hit_col = None
        for col in df.columns:
            if col.lower() in ["name", "indicname", "indicator", "indicname"]:
                name_col = col
            # Some API responses use a camelCase 'isHit' field, which becomes
            # 'ishit' when lowered. Include that variant in our match list so
            # we pick up the correct column regardless of case.
            if col.lower() in ["hit", "triggered", "ishit"]:
                hit_col = col
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
        * **Never rely on a single indicator:** require at least three signals
          (MVRV Z, LTH‑SOPR, Pi Cycle, exchange inflows, reserve risk) for
          high confidence.
        * **Monitor daily:** track MVRV progression, exchange flows and funding
          rates each day to catch rapid changes.
        * **Historical timing:** prior cycle tops have occurred ~1,060 days from the
          cycle low, suggesting the next major top could emerge around
          **September 2025** (although the ETF era may alter patterns).
        * **Risk management:** reduce positions as the escalation levels
          increase; levels 3 and 4 warrant aggressive de‑risking.
        * **Market evolution:** recognise that ETF inflows and changing
          macro conditions can shift the behaviour of these metrics.
        """
    )


if __name__ == "__main__":
    main()