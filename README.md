# Bitcoin Cycle Top Indicator Dashboard

This repository contains a Streamlit application that aggregates
on‑chain, exchange and macro data to help identify when Bitcoin may
be approaching a cycle top. The dashboard tracks key metrics such as
MVRV Z‑Score, Long‑Term Holder SOPR, exchange inflows, Reserve Risk,
Pi Cycle moving averages, Bitcoin dominance, funding rates, DeFi TVL
growth and the 10‑year US Treasury yield. When at least three of the
five primary indicators align, the app flags a “high confidence”
signal. It also summarises escalation levels based on broader market
conditions.

## Features

- **On‑chain metrics** sourced from Glassnode: MVRV Z‑Score, Long
  Term Holder SOPR, Reserve Risk and exchange inflow volumes.
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

You need a valid **Glassnode API key** with an
advanced subscription to access some on‑chain metrics. Provide the key
via the environment variable `GLASSNODE_API_KEY` or enter it into the
sidebar when running the app.

## Running the App

Run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The app will launch in your default web browser. Enter your Glassnode
API key in the sidebar and adjust parameters such as the DeFi chain
and TVL lookback window. The dashboard will then display the latest
metrics, highlight high confidence signals and show escalation levels.

## Notes

- Some Glassnode endpoints require an *advanced* plan. If you use a
  free API key, these metrics will return `N/A`.
- The DeFiLlama API may occasionally block server‑side requests. If
  the TVL data fails to load, try again later or choose a different
  chain.
- The Pi Cycle indicator is computed using Coingecko price data and
  simple moving averages. It is meant to provide directional
  guidance rather than exact price targets.

## Sources

- Glassnode metric definitions for **MVRV Z‑Score**, **LTH‑SOPR** and
  **Reserve Risk**【475926368150572†L216-L244】【118412254918152†L219-L248】.
- Glassnode API endpoint mappings from the open‑source script listing
  metrics【474942229923549†L79-L83】【474942229923549†L154-L160】.
- Coingecko global data for Bitcoin dominance【644653073643528†L40-L43】.
- Binance funding rate documentation【789894824174857†L93-L136】.
