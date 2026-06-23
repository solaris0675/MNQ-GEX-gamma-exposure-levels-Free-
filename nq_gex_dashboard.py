# =============================================================================
#  NQ / MNQ GEX DASHBOARD  (Google Colab Edition)
# =============================================================================
#  What this does:
#    - Pulls LIVE QQQ options data (free, via yfinance) every time you run it
#    - Computes Gamma Exposure (GEX) per strike using Black-Scholes gamma
#    - Scales QQQ strikes -> NQ-equivalent price levels (QQQ tracks Nasdaq-100,
#      NQ/MNQ futures track the same index, so we convert using the live
#      QQQ-to-NQ ratio)
#    - Finds Gamma Flip, Call Wall, Put Wall
#    - Shows ONE combined summary table per timeframe: Flip + Walls + Top 7
#      strikes by absolute GEX (mixed positive/negative) = 10 lines total
#    - Plots a dark, professional dashboard: histogram of GEX by strike
#      (in NQ terms) with walls/flip marked as labeled vertical lines
#    - Runs ALL timeframes automatically every time: 0DTE, 1DTE, 7DTE,
#      14DTE, 30DTE, and ALL expirations combined
#
#  IMPORTANT — read this:
#    NQ/MNQ futures options are NOT available on free data sources.
#    There is no way around this without a paid CME/broker data feed.
#    The standard free-data workaround (used by most retail GEX tools) is
#    to use QQQ options -- since QQQ tracks the Nasdaq-100 almost exactly --
#    and convert the strikes into NQ-equivalent price levels using the
#    current QQQ:NQ ratio. That's what this script does. Treat the NQ
#    levels as a close approximation, NOT an exact 1:1 NQ options market.
#
#  WHY THE CHART LOOKS "SPIKY" / NOT A SMOOTH CURVE:
#    Real open interest is NOT evenly spread across strikes. Traders pile
#    into round numbers and popular strikes, so GEX per strike is naturally
#    jagged -- spikes and dips next to each other, with calls and puts both
#    contributing at strikes near the money. That's expected and correct,
#    not a bug.
#
#  HOW TO RUN IN COLAB:
#    1. New Colab notebook -> paste this whole script into one cell
#    2. Run the cell (Shift+Enter)
#    3. That's it. No file uploads, no API keys needed.
#    4. Re-run the cell anytime for a fresh, live snapshot.
# =============================================================================

# ---- Install dependencies (Colab-safe, only installs if missing) ----------
import subprocess, sys

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

for _p in ["yfinance", "pandas", "numpy", "scipy", "matplotlib"]:
    _ensure(_p)

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timezone

# =============================================================================
#  SETTINGS — tweak these if you want
# =============================================================================
RISK_FREE_RATE   = 0.045   # annualized risk-free rate used in Black-Scholes
VOLUME_WEIGHT    = 0.40    # how much same-day volume counts vs open interest
TOP_N_LEVELS     = 7       # top strikes (by |GEX|) shown in the combined summary
CONTRACT_MULT    = 100     # standard equity option multiplier (shares/contract)

# Which timeframes to run automatically, every single run, in this order.
# "max_dte" = include every expiration with DTE <= this number.
# "ALL" is handled separately (every expiration, no cutoff).
DTE_MODES = [
    ("0DTE",  0),
    ("1DTE",  1),
    ("7DTE",  7),
    ("14DTE", 14),
    ("30DTE", 30),
]

# NQ / MNQ futures contract multipliers (dollars per index point)
NQ_MULTIPLIER  = 20        # E-mini Nasdaq-100 (NQ)  = $20 / point
MNQ_MULTIPLIER = 2         # Micro E-mini Nasdaq-100 (MNQ) = $2 / point

# --- Candlestick + GEX-level overlay chart settings (GexView-style) --------
CANDLE_PERIOD       = "5d"   # how much price history to pull (yfinance limit for 5m bars)
CANDLE_INTERVAL     = "5m"   # candle size
N_LEVEL_LINES       = 14     # how many top GEX levels to draw as horizontal lines
MIN_LINE_FRAC       = 0.10   # weakest shown level draws a line at least this long (10% of max)


# =============================================================================
#  STEP 1 — Pull live spot prices (QQQ + NQ futures) and build the scale factor
# =============================================================================
def get_spots():
    qqq = yf.Ticker("QQQ")
    qqq_spot = float(qqq.history(period="1d")["Close"].iloc[-1])

    # NQ=F is the continuous front-month Nasdaq-100 futures contract on Yahoo
    nq = yf.Ticker("NQ=F")
    nq_hist = nq.history(period="1d")["Close"]
    if len(nq_hist) == 0:
        # fallback: NQ=F can be briefly unavailable around session changes
        nq_spot = qqq_spot * 41.0   # rough fallback ratio, rarely used
    else:
        nq_spot = float(nq_hist.iloc[-1])

    scale = nq_spot / qqq_spot   # multiply any QQQ strike by this to get NQ-equivalent
    return qqq, qqq_spot, nq_spot, scale


# =============================================================================
#  STEP 1b — Pull recent QQQ candles for the price-chart backdrop, converted
#  to NQ-equivalent price using the same scale factor
# =============================================================================
def get_recent_candles(scale):
    qqq = yf.Ticker("QQQ")
    hist = qqq.history(period=CANDLE_PERIOD, interval=CANDLE_INTERVAL)
    if hist is None or len(hist) == 0:
        return None

    hist = hist.reset_index()
    time_col = "Datetime" if "Datetime" in hist.columns else hist.columns[0]
    hist = hist.rename(columns={time_col: "time"})

    # Convert OHLC to NQ-equivalent price terms
    for col in ["Open", "High", "Low", "Close"]:
        hist[col] = hist[col] * scale

    return hist[["time", "Open", "High", "Low", "Close"]]


# =============================================================================
#  STEP 2 — Black-Scholes Gamma
# =============================================================================
def gamma_bs(S, K, T, r, sigma):
    """Black-Scholes gamma. Returns 0 on bad/missing inputs instead of crashing."""
    try:
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            return 0.0
        if T <= 0:
            T = 1 / 365
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception:
        return 0.0


# =============================================================================
#  STEP 3 — Pull option chain(s) and compute GEX per strike
#  (chains are cached per-expiration so each expiry is only fetched once,
#   even though 0/1/7/14/30/ALL modes overlap heavily)
# =============================================================================
def select_expirations(expirations, today, max_dte=None):
    """max_dte=None means ALL (no cutoff). Otherwise include DTE 0..max_dte."""
    if max_dte is None:
        return list(expirations)
    out = []
    for e in expirations:
        dte = (pd.Timestamp(e) - today).days
        if 0 <= dte <= max_dte:
            out.append(e)
    return out


def fetch_gex_data(ticker, spot, exp_list, today, chain_cache):
    """
    exp_list: list of expiration date strings to include
    chain_cache: dict shared across calls so we don't re-download the same
                 expiration's option chain multiple times
    Returns a DataFrame with columns: strike, type, effective_oi, gex, expiration
    or None if there's nothing to show.
    """
    if not exp_list:
        return None

    rows = []
    for exp in exp_list:
        try:
            if exp not in chain_cache:
                chain_cache[exp] = ticker.option_chain(exp)
            chain = chain_cache[exp]

            T = max((pd.Timestamp(exp) - today).days, 0) / 365
            T = max(T, 1 / 365)  # avoid T=0 blowing up the BS formula

            for opt_type, df, sign in [("CALL", chain.calls, 1), ("PUT", chain.puts, -1)]:
                for _, row in df.iterrows():
                    strike = row["strike"]
                    oi  = row.get("openInterest", 0) or 0
                    vol = row.get("volume", 0) or 0
                    iv  = row.get("impliedVolatility", np.nan)

                    effective_oi = oi + vol * VOLUME_WEIGHT
                    if effective_oi <= 0:
                        continue

                    gam = gamma_bs(spot, strike, T, RISK_FREE_RATE, iv)
                    # Dealer-convention GEX: calls = positive (long gamma dealer),
                    # puts = negative. Scaled to $ exposure per 1% move in spot.
                    gex = sign * gam * effective_oi * CONTRACT_MULT * spot ** 2 * 0.01

                    rows.append([strike, opt_type, effective_oi, gex, exp])
        except Exception:
            continue

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["strike", "type", "effective_oi", "gex", "expiration"])


# =============================================================================
#  STEP 4 — Walls, Gamma Flip
# =============================================================================
def analyze(df):
    """Takes the raw rows DataFrame, returns (levels_df, gamma_flip, call_wall, put_wall)."""
    levels = df.groupby("strike")["gex"].sum().reset_index().sort_values("strike")
    levels["cum_gex"] = levels["gex"].cumsum()

    gamma_flip = None
    for i in range(1, len(levels)):
        prev, curr = levels.iloc[i - 1]["cum_gex"], levels.iloc[i]["cum_gex"]
        if prev < 0 <= curr:
            # linear interpolation between the two strikes for a cleaner flip point
            s0, s1 = levels.iloc[i - 1]["strike"], levels.iloc[i]["strike"]
            frac = -prev / (curr - prev) if (curr - prev) != 0 else 0
            gamma_flip = s0 + frac * (s1 - s0)
            break

    call_oi = df[df["type"] == "CALL"].groupby("strike")["effective_oi"].sum()
    put_oi  = df[df["type"] == "PUT"].groupby("strike")["effective_oi"].sum()

    call_wall = call_oi.idxmax() if len(call_oi) else None
    put_wall  = put_oi.idxmax()  if len(put_oi)  else None

    return levels, gamma_flip, call_wall, put_wall


# =============================================================================
#  STEP 5 — The dashboard plot
# =============================================================================
def plot_dashboard(levels, gamma_flip, call_wall, put_wall, scale, mode_name,
                    qqq_spot, nq_spot, expiry_label):

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Convert strikes to NQ-equivalent price levels
    nq_strikes = levels["strike"] * scale
    colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in levels["gex"]]

    bar_width = max((nq_strikes.max() - nq_strikes.min()) / max(len(nq_strikes), 1) * 0.9, 1)
    ax.bar(nq_strikes, levels["gex"], color=colors, width=bar_width, alpha=0.9, zorder=2)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # Stack labels at different heights so they never overlap, regardless of
    # how close together spot / flip / call wall / put wall happen to land.
    label_heights = [ymax - yrange * 0.04, ymax - yrange * 0.14,
                      ymax - yrange * 0.24, ymax - yrange * 0.34]

    def _label(x, y, text, color):
        ax.text(x, y, text, color=color, fontsize=9, fontweight="bold",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0e1117", ec=color, lw=1, alpha=0.9),
                zorder=5)

    # Spot price line
    ax.axvline(nq_spot, color="white", linewidth=1.6, linestyle="-", zorder=3)
    _label(nq_spot, label_heights[0], f"Spot: {nq_spot:,.0f}", "white")

    # Gamma Flip
    if gamma_flip is not None:
        flip_nq = gamma_flip * scale
        ax.axvline(flip_nq, color="#f1c40f", linewidth=1.8, linestyle="--", zorder=3)
        _label(flip_nq, label_heights[1], f"Gamma Flip: {flip_nq:,.0f}", "#f1c40f")

    # Call Wall
    if call_wall is not None:
        cw_nq = call_wall * scale
        ax.axvline(cw_nq, color="#3498db", linewidth=1.8, linestyle=":", zorder=3)
        _label(cw_nq, label_heights[2], f"Call Wall: {cw_nq:,.0f}", "#3498db")

    # Put Wall
    if put_wall is not None:
        pw_nq = put_wall * scale
        ax.axvline(pw_nq, color="#e67e22", linewidth=1.8, linestyle=":", zorder=3)
        _label(pw_nq, label_heights[3], f"Put Wall: {pw_nq:,.0f}", "#e67e22")

    ax.set_title(
        f"NQ / MNQ GEX Dashboard  —  {mode_name}\n"
        f"{expiry_label}   |   QQQ spot: {qqq_spot:,.2f}   →   NQ-equiv scale ×{scale:,.2f}",
        color="white", fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel("NQ-Equivalent Price Level", color="white", fontsize=11)
    ax.set_ylabel("Gamma Exposure ($ per 1% move)", color="white", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e6:,.0f}M"))
    ax.tick_params(colors="white")
    ax.axhline(0, color="gray", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="gray", alpha=0.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(facecolor="#2ecc71", label="Positive GEX (support / resistance, low vol)"),
        Patch(facecolor="#e74c3c", label="Negative GEX (amplifies moves, high vol)"),
        Line2D([0], [0], color="white", lw=1.6, label="Spot Price"),
        Line2D([0], [0], color="#f1c40f", lw=1.8, ls="--", label="Gamma Flip"),
        Line2D([0], [0], color="#3498db", lw=1.8, ls=":", label="Call Wall"),
        Line2D([0], [0], color="#e67e22", lw=1.8, ls=":", label="Put Wall"),
    ]
    ax.legend(handles=legend_items, loc="upper left", facecolor="#1a1d24",
              edgecolor="#333333", labelcolor="white", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  STEP 5b — GexView-style chart: candlesticks with horizontal GEX level
#  lines drawn directly on the price chart, length = strength of that level
# =============================================================================
def _draw_candles(ax, candles):
    """Draw simple up/down candlesticks. candles: DataFrame with time, Open, High, Low, Close."""
    x = np.arange(len(candles))
    for i, row in enumerate(candles.itertuples(index=False)):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        color = "#26a69a" if c >= o else "#ef5350"  # teal up / red down, classic candle colors
        ax.plot([i, i], [l, h], color=color, linewidth=0.7, zorder=2)
        body_low, body_high = min(o, c), max(o, c)
        ax.add_patch(plt.Rectangle((i - 0.3, body_low), 0.6, max(body_high - body_low, 1e-6),
                                    facecolor=color, edgecolor=color, linewidth=0, zorder=3))
    return x


def plot_levels_on_price(levels, candles, scale, mode_name, nq_spot, expiry_label):
    """
    GexView-style chart: live NQ-equivalent candles with the strongest GEX
    levels drawn as horizontal lines starting from the right edge, extending
    left -- length is proportional to that level's GEX strength.
    """
    if candles is None or len(candles) == 0:
        print("⚠️  No recent price history available for the candle overlay chart.")
        return

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    x = _draw_candles(ax, candles)
    n_candles = len(candles)

    # Pick the strongest N levels by |GEX|, convert to NQ price
    top = levels.copy()
    top["nq_level"] = top["strike"] * scale
    top["abs_gex"] = top["gex"].abs()
    top = top.sort_values("abs_gex", ascending=False).head(N_LEVEL_LINES)

    max_abs = top["abs_gex"].max() if len(top) else 1.0

    # Line length scales with strength: weakest shown level gets MIN_LINE_FRAC
    # of the chart width, strongest gets close to full width, drawn from the
    # right edge backward (matches the reference image's style)
    full_span = n_candles * 0.9
    for _, r in top.iterrows():
        frac = MIN_LINE_FRAC + (1 - MIN_LINE_FRAC) * (r["abs_gex"] / max_abs)
        line_len = full_span * frac
        x_start = max(n_candles - line_len, 0)
        color = "#00e676" if r["gex"] >= 0 else "#ff2e6b"  # bright green / pink, like reference
        ax.plot([x_start, n_candles - 1 + n_candles * 0.04], [r["nq_level"], r["nq_level"]],
                color=color, linewidth=2.2, alpha=0.9, zorder=4, solid_capstyle="butt")

    # Spot price marker on the right edge
    ax.axhline(nq_spot, color="white", linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)
    ax.text(n_candles - 1 + n_candles * 0.045, nq_spot, f" {nq_spot:,.0f}",
            color="black", fontsize=9, fontweight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ff5252", ec="none"), zorder=6)

    # X-axis: show real dates/times at evenly spaced ticks
    n_ticks = min(8, n_candles)
    tick_idx = np.linspace(0, n_candles - 1, n_ticks).astype(int)
    tick_labels = [pd.Timestamp(candles["time"].iloc[i]).strftime("%b %d\n%H:%M")
                   for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8.5)

    ax.set_xlim(-2, n_candles * 1.08)

    y_lo, y_hi = ax.get_ylim()
    y_pad = (y_hi - y_lo) * 0.05
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_title(
        f"NQ / MNQ — Live GEX Levels on Price  —  {mode_name}\n"
        f"{expiry_label}   |   green = positive GEX (support/resistance)   "
        f"pink = negative GEX (volatility zones)",
        color="white", fontsize=12.5, fontweight="bold", pad=14
    )
    ax.set_ylabel("NQ-Equivalent Price", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.grid(axis="y", color="gray", alpha=0.15, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="#00e676", lw=2.2, label="Positive GEX level"),
        Line2D([0], [0], color="#ff2e6b", lw=2.2, label="Negative GEX level"),
        Line2D([0], [0], color="white", lw=0.8, label="Current Spot"),
    ]
    ax.legend(handles=legend_items, loc="upper left", facecolor="#1a1d24",
              edgecolor="#333333", labelcolor="white", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.show()
#  Spot, Gamma Flip, Call Wall, Put Wall + Top 7 strikes by |GEX| (mixed +/-)
#  = 10 lines total, beginner friendly
# =============================================================================
def print_combined_summary(levels, gamma_flip, call_wall, put_wall, scale,
                            nq_spot, mode_name):
    print(f"\n📋 SUMMARY — {mode_name}  (10-line readout)")
    print("-" * 50)
    print(f"{'Spot':<14}{nq_spot:>12,.0f}")
    print(f"{'Gamma Flip':<14}{(gamma_flip*scale if gamma_flip is not None else float('nan')):>12,.0f}")
    print(f"{'Call Wall':<14}{(call_wall*scale if call_wall is not None else float('nan')):>12,.0f}")
    print(f"{'Put Wall':<14}{(put_wall*scale if put_wall is not None else float('nan')):>12,.0f}")

    top = levels.copy()
    top["abs_gex"] = top["gex"].abs()
    top = top.sort_values("abs_gex", ascending=False).head(TOP_N_LEVELS)
    top = top.sort_values("strike")  # display in price order, easier to scan

    for _, r in top.iterrows():
        nq_level = r["strike"] * scale
        tag = "+" if r["gex"] >= 0 else "-"
        print(f"Level ({tag})  {nq_level:>10,.0f}   {r['gex']/1e6:>+9.2f}M")
    print("-" * 50)


# =============================================================================
#  STEP 7 — Run one full mode: fetch -> analyze -> plot -> summary -> csv
# =============================================================================
def run_mode(mode_name, max_dte, ticker, expirations, today,
             qqq_spot, nq_spot, scale, chain_cache, candles):
    print(f"\n{'='*70}")
    print(f"  MODE: {mode_name}")
    print(f"{'='*70}")

    exp_list = select_expirations(expirations, today, max_dte)

    if not exp_list:
        print(f"⚠️  No expirations found for {mode_name} (no contracts in that DTE window today).")
        return

    df = fetch_gex_data(ticker, qqq_spot, exp_list, today, chain_cache)

    if df is None:
        print(f"⚠️  No usable option data for {mode_name}.")
        return

    levels, gamma_flip, call_wall, put_wall = analyze(df)

    used_expiries = sorted(df["expiration"].unique())
    if len(used_expiries) == 1:
        expiry_label = f"Expiry: {used_expiries[0]}"
    else:
        expiry_label = f"{len(used_expiries)} expirations ({used_expiries[0]} → {used_expiries[-1]})"

    # Chart 1: histogram dashboard (walls/flip as vertical lines)
    plot_dashboard(levels, gamma_flip, call_wall, put_wall, scale,
                    mode_name, qqq_spot, nq_spot, expiry_label)

    # Chart 2: GexView-style candlesticks with GEX levels drawn on price
    plot_levels_on_price(levels, candles, scale, mode_name, nq_spot, expiry_label)

    print_combined_summary(levels, gamma_flip, call_wall, put_wall, scale,
                            nq_spot, mode_name)

    # CSV export (NQ-converted, Colab-friendly path)
    out = levels.copy()
    out["nq_level"] = out["strike"] * scale
    fname = f"gex_{mode_name.lower()}_nq.csv"
    out.to_csv(fname, index=False)
    print(f"💾 Saved: {fname}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    print("="*70)
    print("  NQ / MNQ GEX DASHBOARD  —  LIVE")
    print(f"  Run time: {now_str}")
    print("  Data source: QQQ options (yfinance, live) scaled to NQ price levels")
    print("="*70)

    ticker = yf.Ticker("QQQ")
    qqq_spot, nq_spot, scale = get_spots()[1:]
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    print(f"\nQQQ Spot     : {qqq_spot:,.2f}")
    print(f"NQ Spot      : {nq_spot:,.2f}")
    print(f"Scale Factor : {scale:,.3f}   (multiply any QQQ strike by this)")
    print(f"MNQ note     : MNQ tracks the same index level as NQ (1/10th the $ per point), "
          f"so the price LEVELS shown here apply to both NQ and MNQ — only the dollar "
          f"value per point differs (NQ = ${NQ_MULTIPLIER}/pt, MNQ = ${MNQ_MULTIPLIER}/pt).")

    expirations = ticker.options
    print(f"\nAvailable QQQ expirations ({len(expirations)} total):")
    print(", ".join(expirations))

    print("\nFetching recent price history for chart overlay...")
    candles = get_recent_candles(scale)
    if candles is not None:
        print(f"Loaded {len(candles)} candles ({CANDLE_INTERVAL} interval, {CANDLE_PERIOD} history).")
    else:
        print("⚠️  Could not load price history — candle overlay charts will be skipped.")

    chain_cache = {}  # shared across all modes so each expiration is fetched once

    for mode_name, max_dte in DTE_MODES:
        run_mode(mode_name, max_dte, ticker, expirations, today,
                  qqq_spot, nq_spot, scale, chain_cache, candles)

    # ALL = every expiration, no DTE cutoff
    run_mode("ALL", None, ticker, expirations, today,
              qqq_spot, nq_spot, scale, chain_cache, candles)

    print("\n" + "="*70)
    print("  Done. Re-run this cell anytime for an updated, live snapshot.")
    print("="*70)


if __name__ == "__main__":
    main()
