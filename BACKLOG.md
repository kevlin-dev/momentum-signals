# Momentum Signals — Backlog

## Next Evolution: Real-Time Streaming (Next.js)

**Why**: Streamlit can't do real-time charts. It refreshes the whole page or uses fragments — the chart line never moves live. Day trading needs tick-by-tick visual feedback.

**Tech stack**:
- **Frontend**: Next.js + [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) (open source, by TradingView — built for exactly this)
- **Data feed**: WebSocket connection to Finnhub (they have a WebSocket API) or Alpaca
- **How it works**: WebSocket keeps a persistent connection open. Each new price tick pushes one data point to the chart — line extends smoothly, no page refresh. This is what TradingView, thinkorswim, and every real trading platform uses.

**When**: Worth building when Kevin gets serious about day trading. Swing trading works fine with the current Streamlit version + fragment refresh.

## Current Version (Streamlit) — Open Items

- [ ] Design pass — Streamlit defaults work but limited
- [ ] Multi-ticker watchlist — scan 3-5 stocks at once
- [ ] Connect to portfolio tracker data (show signals for Kevin's positions)
- [ ] Trade journal feature — log entries/exits, track P&L per swing
- [ ] Intraday candles from Finnhub (currently yfinance with delay for historical)
- [ ] Alpha Vantage API key — Kevin has one, store when needed
