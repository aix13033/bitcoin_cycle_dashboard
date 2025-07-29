# Bitcoin Cycle Top Indicator Dashboard

This repository contains a Streamlit application that aggregates
on‑chain, exchange and macro data to help identify when Bitcoin may
be approaching a cycle top.  The dashboard tracks key metrics such as
MVRV Z‑Score, Long‑Term Holder SOPR, exchange inflows (approximated
by ETF flows), Reserve Risk, Pi Cycle moving averages, Bitcoin
dominance, funding rates, DeFi TVL growth and the 10‑year US Treasury
yield.  When at least three of the five primary indicators align, the
app flags a “high confidence” signal.  It also summarises escalation
levels based on broader market conditions.

## Features

* **On‑chain metrics** sourced from the free **BGeometrics Bitcoin Data
  API**: MVRV Z‑Score, Long Term Holder SOPR and Reserve Risk.  The
  API requires no key, though it is rate‑limited.  Exchange inflow
  volumes are proxied using ETF flows from the same API.  When price
  data is unavailable, ETF flows are displayed in BTC (thousands)
  rather than converted to USD.
* **Technical indicator**: Pi Cycle Top (111‑day vs 2×350‑day SMAs).
* **Exchange data**: Bitcoin dominance (Coingecko) and funding
  rates (Binance perpetual futures).
* **Macro data**: DeFi Total Value Locked (DefiLlama) and 10‑year
  Treasury yield (Yahoo Finance via `yfinance`).
* **Escalation levels** describing early warning, caution, high risk
  and extreme danger scenarios.

* **Bull Market Peak Signals (optional)**: The dashboard can pull the
  latest bull market peak indicators from Coinglass’s API.  These
  signals include metrics like AHR999, Pi Cycle Top, Stock‑to‑Flow and
  others.  You must supply your own **Coinglass API key** via the
  `COINGLASS_API_KEY` environment variable or the sidebar input.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

No API key is required for on‑chain metrics because the app uses
BGeometrics’ free Bitcoin Data API.  However, the API is rate‑limited
to approximately 30 requests per hour.  Other data sources (Coingecko,
Binance, DefiLlama, yfinance) do not require keys.

To display Coinglass bull market peak signals, you must supply your
own API key.  The dashboard automatically reads the key from one of
the following locations (in order of precedence):

1. The `COINGLASS_API_KEY` environment variable.
2. A `.streamlit/secrets.toml` file with either a top‑level
   `coinglass_api_key` entry or a `[coinglass]` section containing
   `api_key`.  For example:

   ```toml
   # .streamlit/secrets.toml
   coinglass_api_key = "your_coinglass_key"

   # or as a section
   [coinglass]
   api_key = "your_coinglass_key"
   ```

If neither of these is set, you can still enter the key in the
sidebar at runtime, but storing it in `secrets.toml` or an environment
variable avoids having to re‑type it each time.

## Running the App

Run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The app will launch in your default web browser.  The configuration
sidebar has been removed, so the dashboard uses sensible defaults
internally (DeFi chain = **Ethereum** and TVL lookback = **30 days**).
If you need to change these values, edit the corresponding variables
(`chain` and `lookback_days`) near the beginning of the `main()` function in
`app.py`.  The dashboard will then display the latest metrics, highlight
high confidence signals and show escalation levels without requiring
any runtime configuration.

The interface groups related metrics into categories (on‑chain, technical/exchange,
macro) and uses tabs to organise charts, signals, a comparison table and
guidance.  The comparison tab summarises how your framework’s thresholds differ
from CoinGlass thresholds and displays the current values of each indicator.
The guidance tab consolidates escalation levels and provides critical success
factors and multi‑domain insights.

## Notes

* BGeometrics’ free API is rate‑limited.  If on‑chain data fails to
  load or appears as `N/A`, wait a few minutes before refreshing.
* The DefiLlama API may occasionally block server‑side requests or
  change response formats.  If the TVL data fails to load, try again
  later or choose a different chain.
* The Pi Cycle indicator is computed using Coingecko price data and
  simple moving averages.  It is meant to provide directional
  guidance rather than exact price targets.
