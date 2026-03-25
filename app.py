"""
Momentum Trading Signal App — MVP
"Right action, right price, right moment"

For beginners who can't read charts. The engine does the analysis.
You get a card that says what to do.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time as _time
import requests as _requests

# --- CONFIG ---
st.set_page_config(
    page_title="Momentum Signals",
    page_icon="--",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_REFRESH = 15  # seconds
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")


def get_finnhub_quote(symbol):
    """Get real-time quote from Finnhub. Returns dict with c (current), h (high), l (low), o (open), pc (prev close)."""
    if not FINNHUB_KEY:
        return None
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
        resp = _requests.get(url, timeout=5)
        data = resp.json()
        if data.get("c", 0) > 0:
            return data
    except Exception:
        pass
    return None

# --- CALCULATIONS ---

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_support_resistance(df, window=20):
    recent = df.tail(window)
    resistance = recent["High"].max()
    support = recent["Low"].min()
    return support, resistance


def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return lower, sma, upper


def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_trailing_stop(df, atr_series, multiplier=2.0):
    closes = df["Close"].values
    highs = df["High"].values
    atr_vals = atr_series.values
    n = len(closes)
    stop = np.full(n, np.nan)
    in_trade = np.full(n, False)
    entry_signals = np.full(n, False)
    exit_signals = np.full(n, False)

    start = 0
    for i in range(n):
        if not np.isnan(atr_vals[i]):
            start = i
            break

    highest = highs[start]
    stop[start] = highest - (multiplier * atr_vals[start])
    in_trade[start] = closes[start] > stop[start]

    for i in range(start + 1, n):
        if np.isnan(atr_vals[i]):
            stop[i] = stop[i - 1]
            in_trade[i] = in_trade[i - 1]
            continue

        if in_trade[i - 1]:
            highest = max(highest, highs[i])
            new_stop = highest - (multiplier * atr_vals[i])
            stop[i] = max(stop[i - 1], new_stop)
            if closes[i] < stop[i]:
                in_trade[i] = False
                exit_signals[i] = True
                highest = highs[i]
            else:
                in_trade[i] = True
        else:
            highest = highs[i]
            stop[i] = highest - (multiplier * atr_vals[i])
            if closes[i] > stop[i] and closes[i] > closes[i - 1]:
                in_trade[i] = True
                entry_signals[i] = True

    return stop, in_trade, entry_signals, exit_signals


def get_signal(price, rsi, macd_hist, sma_20, sma_50, volume_ratio, support, resistance, in_trade_now):
    score = 0
    reasons_bull = []
    reasons_bear = []

    if in_trade_now:
        score += 1
        reasons_bull.append("Above trailing stop (in trade)")
    else:
        score -= 1
        reasons_bear.append("Below trailing stop (no trade)")

    if rsi > 70:
        score -= 2
        reasons_bear.append("RSI overbought (%.0f)" % rsi)
    elif rsi < 30:
        score += 2
        reasons_bull.append("RSI oversold (%.0f)" % rsi)
    elif rsi > 55:
        score += 1
        reasons_bull.append("RSI bullish (%.0f)" % rsi)
    elif rsi < 45:
        score -= 1
        reasons_bear.append("RSI bearish (%.0f)" % rsi)

    if macd_hist > 0:
        score += 1
        reasons_bull.append("MACD positive")
    else:
        score -= 1
        reasons_bear.append("MACD negative")

    if price > sma_20 and price > sma_50:
        score += 2
        reasons_bull.append("Above 20 & 50 day averages")
    elif price < sma_20 and price < sma_50:
        score -= 2
        reasons_bear.append("Below 20 & 50 day averages")
    elif price > sma_20:
        score += 1
        reasons_bull.append("Above 20-day average")

    if volume_ratio > 1.5:
        score += 1
        reasons_bull.append("High volume (%.1fx avg)" % volume_ratio)
    elif volume_ratio < 0.5:
        score -= 1
        reasons_bear.append("Low volume (%.1fx avg)" % volume_ratio)

    range_size = resistance - support
    if range_size > 0:
        position = (price - support) / range_size
        if position < 0.2:
            score += 1
            reasons_bull.append("Near support ($%.2f)" % support)
        elif position > 0.8:
            score -= 1
            reasons_bear.append("Near resistance ($%.2f)" % resistance)

    if score >= 3:
        signal = "ENTER"
        color = "#22c55e"
    elif score <= -3:
        signal = "EXIT"
        color = "#ef4444"
    else:
        signal = "WAIT"
        color = "#f59e0b"

    return signal, score, color, reasons_bull, reasons_bear


def compute_stop_and_targets(price, support, resistance, atr):
    stop_loss = max(support, price - (2 * atr))
    target_1 = price + (1.5 * atr)
    target_2 = max(resistance * 1.02, price + (3 * atr))
    risk = price - stop_loss
    reward_1 = target_1 - price
    rr_ratio = reward_1 / risk if risk > 0 else 0
    return stop_loss, target_1, target_2, risk, rr_ratio


def compute_position_size(capital, risk_pct, price, stop_loss):
    risk_per_share = price - stop_loss
    if risk_per_share <= 0:
        return 0, 0
    risk_amount = capital * (risk_pct / 100)
    shares_by_risk = int(risk_amount / risk_per_share)
    max_shares_by_capital = int(capital / price)
    shares = min(shares_by_risk, max_shares_by_capital)
    position_value = shares * price
    return shares, position_value


# --- CHART BUILDER ---

def build_chart(df, trailing_stop, in_trade, entry_signals, exit_signals,
                sma_20_series, sma_50_series, support, resistance, stop_loss,
                target_1, target_2, current_price, rsi_series, volume_series,
                avg_volume, atr, ticker, lookback=60, level=1):
    """Build chart with progressive complexity levels."""

    plot_df = df.tail(lookback).copy()
    plot_stop = trailing_stop[-lookback:]
    plot_in_trade = in_trade[-lookback:]
    plot_entry = entry_signals[-lookback:]
    plot_exit = exit_signals[-lookback:]
    dates = plot_df.index

    # Determine subplot layout based on level
    if level >= 3:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.6, 0.2, 0.2],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # =====================
    # LEVEL 1: Price + Trailing Stop (the essentials)
    # =====================

    # Blue price line — ALWAYS visible in all modes
    fig.add_trace(go.Scatter(
        x=dates, y=plot_df["Close"],
        mode="lines",
        name="Price",
        line=dict(color="#60a5fa", width=3 if level < 3 else 2),
        hovertemplate="Price: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Candlesticks — only in Pro mode, layered behind the price line
    if level >= 3:
        fig.add_trace(go.Candlestick(
            x=dates,
            open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"],
            name="Candles",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            increasing_fillcolor="rgba(34,197,94,0.4)", decreasing_fillcolor="rgba(239,68,68,0.4)",
            opacity=0.6,
        ), row=1, col=1)

    # Trailing stop — THE key visual
    # Color it based on trade state
    in_trade_colors = []
    for i, t in enumerate(plot_in_trade):
        in_trade_colors.append("#22c55e" if t else "#ef4444")

    fig.add_trace(go.Scatter(
        x=dates, y=plot_stop,
        mode="lines",
        name="Safety Line (trailing stop)",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate=(
            "<b>Safety Line: $%{y:.2f}</b><br>"
            "If price drops below this → SELL<br>"
            "This line only moves UP, never down<extra></extra>"
        ),
    ), row=1, col=1)

    # Shade the zone between price and stop when in trade
    for i in range(len(dates) - 1):
        if plot_in_trade[i]:
            fig.add_shape(
                type="rect",
                x0=dates[i], x1=dates[min(i + 1, len(dates) - 1)],
                y0=plot_stop[i], y1=plot_df["Close"].iloc[i],
                fillcolor="rgba(34, 197, 94, 0.18)",
                line_width=0,
                layer="below",
                row=1, col=1,
            )

    # Entry / Exit markers (always visible — these are the signals)
    entry_dates = [d for d, e in zip(dates, plot_entry) if e]
    entry_prices = [plot_df.loc[d, "Close"] for d in entry_dates]
    exit_dates = [d for d, e in zip(dates, plot_exit) if e]
    exit_prices = [plot_df.loc[d, "Close"] for d in exit_dates]

    # Offset arrows by ATR so they stay close to price on any timeframe
    price_range = plot_df["Close"].max() - plot_df["Close"].min()
    arrow_offset = max(price_range * 0.08, 0.05)  # 8% of visible range, min $0.05

    if entry_dates:
        fig.add_trace(go.Scatter(
            x=entry_dates, y=[p - arrow_offset for p in entry_prices],
            mode="markers+text",
            name="BUY signal",
            marker=dict(symbol="triangle-up", size=20, color="#16a34a",
                       line=dict(width=2, color="#166534")),
            text=["BUY"] * len(entry_dates),
            textposition="bottom center",
            textfont=dict(color="#16a34a", size=12, family="Arial Black"),
            hovertemplate="<b>BUY SIGNAL</b><br>Price recovered above safety line<br>OK to enter<extra></extra>",
        ), row=1, col=1)

    if exit_dates:
        fig.add_trace(go.Scatter(
            x=exit_dates, y=[p + arrow_offset for p in exit_prices],
            mode="markers+text",
            name="SELL signal",
            marker=dict(symbol="triangle-down", size=20, color="#dc2626",
                       line=dict(width=2, color="#991b1b")),
            text=["SELL"] * len(exit_dates),
            textposition="top center",
            textfont=dict(color="#dc2626", size=12, family="Arial Black"),
            hovertemplate="<b>SELL SIGNAL</b><br>Price dropped below safety line<br>Exit the trade<extra></extra>",
        ), row=1, col=1)

    # =====================
    # LEVEL 2: Add zones + your trade levels
    # =====================

    if level >= 2:
        # Support zone (green band)
        fig.add_hrect(
            y0=support * 0.99, y1=support * 1.01,
            fillcolor="rgba(34, 197, 94, 0.15)", line_width=0,
            annotation_text=f"Support ${support:.2f} — price tends to bounce here",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="#22c55e"),
            row=1, col=1,
        )

        # Resistance zone (red band)
        fig.add_hrect(
            y0=resistance * 0.99, y1=resistance * 1.01,
            fillcolor="rgba(239, 68, 68, 0.15)", line_width=0,
            annotation_text=f"Resistance ${resistance:.2f} — price tends to stall here",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#ef4444"),
            row=1, col=1,
        )

        # Your stop loss
        fig.add_hline(
            y=stop_loss, line_dash="dot", line_color="#ef4444", line_width=2,
            annotation_text=f"Your stop loss ${stop_loss:.2f}",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#ef4444"),
            row=1, col=1,
        )

        # Your targets
        fig.add_hline(
            y=target_1, line_dash="dot", line_color="#22c55e", line_width=1,
            annotation_text=f"Target 1 ${target_1:.2f} (sell half)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#22c55e"),
            row=1, col=1,
        )
        fig.add_hline(
            y=target_2, line_dash="dot", line_color="#16a34a", line_width=1,
            annotation_text=f"Target 2 ${target_2:.2f} (sell rest)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#16a34a"),
            row=1, col=1,
        )

        # Moving averages
        fig.add_trace(go.Scatter(
            x=dates, y=sma_20_series.tail(lookback),
            mode="lines", name=f"{fast_ma}-period trend",
            line=dict(color="#60a5fa", width=1, dash="dash"),
            hovertemplate=f"{fast_ma}-period trend: $%{{y:.2f}} — short-term direction<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=sma_50_series.tail(lookback),
            mode="lines", name=f"{slow_ma}-period trend",
            line=dict(color="#a78bfa", width=1, dash="dash"),
            hovertemplate=f"{slow_ma}-period trend: $%{{y:.2f}} — medium-term direction<extra></extra>",
        ), row=1, col=1)

    # =====================
    # LEVEL 3: RSI + Volume subplots
    # =====================

    if level >= 3:
        plot_rsi = rsi_series.tail(lookback)
        plot_vol = volume_series.tail(lookback)

        # RSI
        fig.add_trace(go.Scatter(
            x=dates, y=plot_rsi,
            mode="lines", name="RSI (momentum)",
            line=dict(color="#f59e0b", width=1.5),
            hovertemplate=(
                "RSI: %{y:.0f}<br>"
                "Above 70 = stock ran too hot, may pull back<br>"
                "Below 30 = stock dropped too hard, may bounce<extra></extra>"
            ),
        ), row=2, col=1)

        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.1)",
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.1)",
                      line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)", row=2, col=1)

        # Volume
        vol_colors = ["#22c55e" if c >= o else "#ef4444"
                      for c, o in zip(plot_df["Close"], plot_df["Open"])]
        fig.add_trace(go.Bar(
            x=dates, y=plot_vol, name="Volume",
            marker_color=vol_colors, opacity=0.5,
            hovertemplate=(
                "Volume: %{y:,.0f} shares traded<br>"
                "High volume = real conviction behind the move<br>"
                "Low volume = the move might be a fake-out<extra></extra>"
            ),
        ), row=3, col=1)
        fig.add_hline(
            y=avg_volume, line_dash="dash", line_color="rgba(255,255,255,0.3)",
            annotation_text="Average volume",
            annotation_font=dict(size=9, color="rgba(255,255,255,0.5)"),
            row=3, col=1,
        )

    # =====================
    # LAYOUT
    # =====================

    height = 350 if level == 1 else 400 if level == 2 else 600

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=height,
        margin=dict(l=50, r=20, t=20, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=12),
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True, row=1, col=1)

    if level >= 3:
        fig.update_yaxes(row=2, col=1, title_text="RSI", range=[0, 100],
                        gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(row=3, col=1, title_text="Volume",
                        gridcolor="rgba(255,255,255,0.05)")

    return fig


# --- UI ---

# Minimal CSS — only what Streamlit can't do natively
st.markdown("""
<style>
    .signal-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .signal-text {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: 4px;
    }
    .price-text {
        font-size: 18px;
        margin-top: 4px;
    }
    .trade-status {
        font-size: 13px;
        margin-top: 6px;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_title, col_input, col_mode, col_level = st.columns([2, 1.5, 1, 1])

with col_title:
    st.markdown("## Momentum Signals")

with col_input:
    ticker = st.text_input("Ticker", value="ONDS", label_visibility="collapsed",
                           placeholder="Ticker (e.g. ONDS)")

with col_mode:
    mode = st.radio("Mode", ["Swing", "Day"], horizontal=True, label_visibility="collapsed",
                    help="Swing = daily candles, hold for days. Day = 5-min candles, in and out same day.")

with col_level:
    level = st.select_slider(
        "Chart detail",
        options=[1, 2, 3],
        value=1,
        format_func=lambda x: {1: "Easy", 2: "Normal", 3: "Pro"}[x],
    )

# Mode config
if mode == "Swing":
    data_period = "6mo"
    data_interval = "1d"
    default_lookback = 60
    mode_label = "Swing Trading — daily candles, hold for days"
    refresh_interval = 60
else:
    data_period = "5d"
    data_interval = "5m"
    default_lookback = 78
    mode_label = "Day Trading — 5-min candles, in and out today"
    refresh_interval = 15

# Always auto-refresh
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = _time.time()
elapsed = _time.time() - st.session_state.last_refresh
if elapsed >= refresh_interval:
    st.session_state.last_refresh = _time.time()
    st.rerun()

# --- MAIN ---
if ticker:
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=data_period, interval=data_interval)

        if df.empty:
            st.error("No data found for %s" % ticker.upper())
            st.stop()

        # Real-time price from Finnhub, historical candles from yfinance
        finnhub_quote = get_finnhub_quote(ticker.upper())
        if finnhub_quote:
            current_price = finnhub_quote["c"]
            prev_close = finnhub_quote["pc"]
            daily_change = finnhub_quote["dp"]
            price_source = "Finnhub (real-time)"
        else:
            current_price = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2]
            daily_change = ((current_price - prev_close) / prev_close) * 100
            price_source = "Yahoo Finance (~15 min delay)"

        rsi_series = compute_rsi(df["Close"])
        rsi = rsi_series.iloc[-1]
        macd_line, signal_line_macd, macd_hist = compute_macd(df["Close"])

        # Moving averages — shorter periods for intraday
        fast_ma = 9 if mode == "Day" else 20
        slow_ma = 21 if mode == "Day" else 50
        sma_20_series = df["Close"].rolling(fast_ma).mean()
        sma_50_series = df["Close"].rolling(slow_ma).mean()
        sma_20 = sma_20_series.iloc[-1]
        sma_50 = sma_50_series.iloc[-1]
        avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
        current_volume = df["Volume"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        atr_series = compute_atr(df)
        atr = atr_series.iloc[-1]
        support, resistance = find_support_resistance(df)

        # ATR multiplier: wider for intraday (5-min noise needs more room)
        stop_multiplier = 3.5 if mode == "Day" else 2.0
        trailing_stop, in_trade, entry_signals, exit_signals = compute_trailing_stop(
            df, atr_series, multiplier=stop_multiplier
        )
        in_trade_now = in_trade[-1]
        current_trailing_stop = trailing_stop[-1]

        signal, score, color, reasons_bull, reasons_bear = get_signal(
            current_price, rsi, macd_hist.iloc[-1], sma_20, sma_50,
            volume_ratio, support, resistance, in_trade_now
        )

        stop_loss, target_1, target_2, risk, rr_ratio = compute_stop_and_targets(
            current_price, support, resistance, atr
        )

        # =====================
        # LAYOUT — Card + Chart side by side, rest below
        # =====================

        col_card, col_chart = st.columns([1, 3])

        with col_card:
            # --- Signal card ---
            trade_status = "Above safety line" if in_trade_now else "Below safety line"
            change_color = '#22c55e' if daily_change >= 0 else '#ef4444'
            change_sign = '+' if daily_change >= 0 else ''
            st.markdown(f"""
            <div style="background: #111827; border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center; margin: 8px 0;">
                <div style="font-size: 48px; font-weight: 800; letter-spacing: 4px; color: {color};">{signal}</div>
                <div style="font-size: 18px; color: #e5e7eb; margin-top: 4px;">
                    {ticker.upper()} &mdash; ${current_price:.2f}
                    <span style="color: {change_color};">({change_sign}{daily_change:.1f}%)</span>
                </div>
                <div style="font-size: 13px; color: #9ca3af; margin-top: 6px;">{trade_status}</div>
            </div>
            """, unsafe_allow_html=True)

            # --- Key numbers ---
            st.metric("Safety Line", f"${current_trailing_stop:.2f}",
                      help="Trailing stop. Only moves up. If price drops below → sell.")

            if level >= 2:
                st.metric("Stop Loss", f"${stop_loss:.2f}",
                          delta=f"-{((current_price - stop_loss) / current_price * 100):.1f}%",
                          delta_color="inverse",
                          help="Your absolute floor. Never move it lower.")
                st.metric("Target 1", f"${target_1:.2f}",
                          delta=f"+{((target_1 - current_price) / current_price * 100):.1f}%",
                          help="Sell half here. Move stop to breakeven.")
                st.metric("Target 2", f"${target_2:.2f}",
                          delta=f"+{((target_2 - current_price) / current_price * 100):.1f}%",
                          help="Sell the rest here.")
                if rr_ratio > 0:
                    rr_label = "Good" if rr_ratio >= 1.5 else "OK" if rr_ratio >= 1 else "Skip"
                    st.metric("Risk/Reward", f"{rr_ratio:.1f}:1", delta=rr_label,
                              delta_color="normal" if rr_ratio >= 1.5 else "off",
                              help="Above 1.5:1 = good trade. Below 1:1 = don't bother.")

        with col_chart:
            # --- Chart ---
            level_desc = {
                1: "Price + Safety Line. That's all you need.",
                2: "Adds support/resistance, targets, and trends.",
                3: "Full dashboard: candlesticks, RSI, volume.",
            }
            st.caption(f"_{mode_label}_ | {level_desc[level]}")

            fig = build_chart(
                df, trailing_stop, in_trade, entry_signals, exit_signals,
                sma_20_series, sma_50_series, support, resistance,
                stop_loss, target_1, target_2, current_price,
                rsi_series, df["Volume"], avg_volume, atr, ticker.upper(),
                lookback=default_lookback, level=level,
            )
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": "hover",
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
                "scrollZoom": True,
            })
            st.caption("Drag to zoom | Scroll to zoom | Double-click to reset")

        # --- 3. THE RULES (know before you act) ---
        st.markdown("---")
        col_play, col_never = st.columns(2)
        with col_play:
            if mode == "Swing":
                st.success(
                    "**How to play (Swing):**\n"
                    "1. Wait for a catalyst (earnings, news) — watch Day 1\n"
                    "2. Enter on Day 2 if same direction + strong volume\n"
                    "3. Set your stop loss immediately\n"
                    "4. At Target 1 → sell half, move stop to breakeven\n"
                    "5. Ride the rest → safety line protects you\n"
                    "6. 3 days of nothing → close and move on\n"
                    "7. Check once in the morning, once at close. That's it."
                )
            else:
                st.success(
                    "**How to play (Day):**\n"
                    "1. Wait for a breakout with volume — don't chase\n"
                    "2. Enter only when price is above safety line AND moving up\n"
                    "3. Set your stop loss immediately\n"
                    "4. At Target 1 → sell half, move stop to breakeven\n"
                    "5. Ride the rest with the safety line\n"
                    "6. No movement for 30 min → close the trade\n"
                    "7. Always exit before market close — never hold overnight"
                )
        with col_never:
            if mode == "Swing":
                st.error(
                    "**Never do this (Swing):**\n"
                    "- Buy without setting a stop loss\n"
                    "- Move your stop loss lower (\"give it more room\")\n"
                    "- Buy more when it's going down (\"it's cheaper now\")\n"
                    "- Hold a trade that isn't working (\"it'll come back\")\n"
                    "- Ignore overnight gaps — check pre-market before open\n"
                    "- Turn a swing trade into an \"investment\""
                )
            else:
                st.error(
                    "**Never do this (Day):**\n"
                    "- Buy without setting a stop loss\n"
                    "- Move your stop loss lower\n"
                    "- Hold a losing position hoping it reverses\n"
                    "- Hold overnight without a plan — gaps can destroy you\n"
                    "- Trade the first 15 min of market open (too chaotic)\n"
                    "- Take more than 3 trades in a day (overtrading)"
                )

        # --- 4. POSITION SIZE (how much to risk) ---
        st.markdown("---")
        st.markdown("### How Much to Buy")
        col_cap, col_risk, col_result = st.columns([1, 1, 2])
        with col_cap:
            capital = st.number_input("Money for this trade ($)", value=10000, step=1000,
                                      help="The amount you're allocating to this trade — not your total portfolio")
        with col_risk:
            risk_pct = st.slider("Max risk (%)", min_value=1, max_value=5, value=2,
                                 help="1% = conservative (beginners start here). 2% = standard. 3%+ = aggressive.")
        with col_result:
            shares, position_value = compute_position_size(capital, risk_pct, current_price, stop_loss)
            if shares > 0:
                max_loss = shares * (current_price - stop_loss)
                st.info(f"Buy {shares} shares (${position_value:,.0f})")
                st.caption(
                    f"Worst case if stopped out: you lose ${max_loss:,.0f} "
                    f"({risk_pct}% of ${capital:,}). That's the MOST you can lose."
                )
            else:
                st.warning("No valid trade setup at this price.")

        # --- 5. LEARN MORE (expandable) ---
        st.markdown("---")

        with st.expander("Why this signal?"):
            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.markdown("**Working for it:**")
                for r in reasons_bull:
                    st.markdown(f"- {r}")
                if not reasons_bull:
                    st.markdown("- *Nothing right now*")
            with col_bear:
                st.markdown("**Working against it:**")
                for r in reasons_bear:
                    st.markdown(f"- {r}")
                if not reasons_bear:
                    st.markdown("- *Nothing right now*")
            st.caption(f"Score: {score} | 3+ = ENTER, -3 or less = EXIT, between = WAIT")

        with st.expander("What am I looking at?"):
            timeframe = "each day" if mode == "Swing" else "every 5 minutes"
            trend_fast = "20-day" if mode == "Swing" else "9-period"
            trend_slow = "50-day" if mode == "Swing" else "21-period"

            if level == 1:
                st.markdown(
                    f"**The blue line** is the stock price (updated {timeframe}).\n\n"
                    "**The yellow line** is your safety net (trailing stop). "
                    "It follows the price up but never goes back down — like a ratchet.\n\n"
                    "- Price ABOVE yellow line → safe to hold\n"
                    "- Price drops BELOW yellow line → sell immediately\n\n"
                    "**Green triangles** = good time to buy | "
                    "**Red triangles** = time to sell\n\n"
                    "When you're comfortable with this, switch to Normal mode."
                )
                if mode == "Day":
                    st.caption(
                        "In Day mode, the safety line is wider to avoid false signals "
                        "from short-term noise. It won't trigger on every small dip."
                    )
            elif level == 2:
                st.markdown(
                    "Everything from Easy mode, plus:\n\n"
                    "**Green band** = support zone (price tends to bounce here)\n"
                    "**Red band** = resistance zone (price tends to stall here)\n"
                    "**Dotted red line** = your stop loss (never move it lower)\n"
                    "**Dotted green lines** = your profit targets\n"
                    f"**Blue dashed** = {trend_fast} trend | **Purple dashed** = {trend_slow} trend\n\n"
                    "Price above both trend lines → uptrend (bullish). "
                    "Below both → downtrend (stay out)."
                )
            else:
                candle_desc = "each day" if mode == "Swing" else "each 5-minute window"
                st.markdown(
                    "Everything from Normal mode, plus:\n\n"
                    f"**Candlesticks** — Green = price went up that {candle_desc}. "
                    "Red = went down. Wicks show the full range.\n\n"
                    "**RSI (middle chart)** — Momentum thermometer 0-100. "
                    "Above 70 = overbought (may pull back). Below 30 = oversold (may bounce).\n\n"
                    "**Volume (bottom chart)** — How many shares traded. "
                    "Tall bars = real conviction. Short bars = possible fake-out. "
                    "Dashed line = average.\n\n"
                    "A strong signal needs BOTH: price moving AND volume confirming."
                )

        st.markdown("---")
        st.caption("Not financial advice. A learning tool. Do your own research. "
                  "Never risk money you can't afford to lose. "
                  f"Data: {price_source} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.caption("Check the ticker symbol and try again.")
