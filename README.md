# Bitcoin Cycle Top Indicator Dashboard

This repository contains a Streamlit application that aggregates
on‑chain, exchange and macro data to help identify when Bitcoin may
be approaching a cycle top. The dashboard tracks key metrics such as
MVRV Z‑Score, Long‑Term Holder SOPR, exchange inflows (approximated
by ETF flows), Reserve Risk, Pi Cycle moving averages, Bitcoin
dominance, funding rates, DeFi TVL growth and the 10‑year US Treasury
yield. When at least three of the five primary indicators align, the
app flags a “high confidence” signal. It also summarises escalation
levels based on broader market conditions.

## Features

- **On‑chain metrics** sourced from the free **BGeometrics Bitcoin Data
  API**: MVRV Z‑Score, Long Term Holder SOPR and Reserve Risk. The
  API requires no key, though it is rate‑limited. Exchange inflow
  volumes are proxied using ETF flows from the same API.
- **Technical indicator**: Pi Cycle Top (111‑day vs 2×350‑day SMAs).
- **Exchange data**: Bitcoin dominance (Coingecko) and funding
  rates (Binance perpetual futures).
- **Macro data**: DeFi Total Value Locked (DefiLlama) and 10‑year
  Treasury yield (Yahoo Finance via `yfinance`).
- **Escalation levels** describing early warning, caution, high risk
  and extreme danger scenarios.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

No API key is required for on‑chain metrics because the app uses
BGeometrics’ free Bitcoin Data API. However, the API is rate‑limited
to approximately 30 requests per hour. Other data sources (Coingecko,
Binance, DefiLlama, yfinance) do not require keys.

## Running the App

Run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The app will launch in your default web browser. Adjust parameters
such as the DeFi chain and TVL lookback window in the sidebar. The
dashboard will then display the latest metrics, highlight high
confidence signals and show escalation levels.

## Notes

- BGeometrics’ free API is rate‑limited. If on‑chain data fails to
  load or appears as `N/A`, wait a few minutes before refreshing.
- The DeFiLlama API may occasionally block server‑side requests. If
  the TVL data fails to load, try again later or choose a different
  chain.
- The Pi Cycle indicator is computed using Coingecko price data and
  simple moving averages. It is meant to provide directional
  guidance rather than exact price targets.

## Sources

- Glassnode metric definitions for **MVRV Z‑Score**, **LTH‑SOPR** and
  **Reserve Risk**【475926368150572†L216-L244】【118412254918152†L219-L248】.
- BGeometrics Bitcoin Data API provides free on‑chain metrics and ETF
  flows (no citation available since API docs cannot be scraped).
- Coingecko global data for Bitcoin dominance【644653073643528†L40-L43】.
- Binance funding rate documentation【789894824174857†L93-L136】.
