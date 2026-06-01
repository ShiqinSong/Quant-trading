"""
Daily Cross-Sectional Catch-Up Signal Scanner
==============================================
Strategy: Use NVDA/AMD/QQQ returns to infer the latent market factor F via PCA.
          Compute residual epsilon for each target stock: epsilon = actual - predicted.
          Signal: -20% < epsilon < -1% means the stock underperformed the factor today
          and has a 60%+ probability of catching up over the next 2-3 days.

Usage:
    python3 daily_signal.py
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Parameters
# ============================================================

# Stocks with validated win rate >= 60% from backtest
SIGNAL_TICKERS = ["AVGO", "MRVL", "APP", "SMCI", "UNH", "BA"]

# Stocks used to infer the latent factor F
INPUT_TICKERS = ["NVDA", "AMD", "QQQ"]

# Full universe for PCA training
ALL_TICKERS = INPUT_TICKERS + SIGNAL_TICKERS + [
    "MSFT", "META", "GOOGL", "AMZN", "PLTR", "ORCL",
    "TSLA", "MSTR", "DELL", "MU",
]

TRAIN_DAYS = 60     # Rolling window for PCA training
N_FACTORS  = 3      # Number of principal components to use
EPS_LOW    = -20.0  # Below this: likely earnings event, exclude
EPS_HIGH   = -1.0   # Above this: signal not strong enough, exclude

# ============================================================
# Data
# ============================================================

def get_data():
    end = datetime.now()
    start = end - timedelta(days=TRAIN_DAYS * 2 + 30)
    print("Downloading data...")
    raw = yf.download(ALL_TICKERS, start=start, end=end,
                      progress=False, timeout=15)["Close"]
    raw = raw.dropna(axis=1, thresh=int(TRAIN_DAYS * 0.9))
    returns = raw.pct_change().dropna()
    tickers = list(raw.columns)
    print(f"  {len(tickers)} stocks, {len(returns)} trading days\n")
    return returns, tickers

# ============================================================
# PCA + Epsilon Calculation
# ============================================================

def calc_epsilon(train_returns, test_row, input_tickers, tickers):
    """
    Infer latent factor F from input stocks, then compute residual
    epsilon = actual - predicted for all stocks.
    """
    X = train_returns.values
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    explained = eigenvalues / eigenvalues.sum()

    # Infer factor F from input stocks via least squares
    input_idx = [tickers.index(t) for t in input_tickers if t in tickers]
    r_input = test_row.values[input_idx]
    B_input = eigenvectors[input_idx, :N_FACTORS]
    F, _, _, _ = np.linalg.lstsq(B_input, r_input, rcond=None)

    # Predict all stocks using the inferred factor
    B_all = eigenvectors[:, :N_FACTORS]
    predicted = B_all @ F
    actual = test_row.values
    epsilon = (actual - predicted) * 100

    return epsilon, predicted * 100, actual * 100, explained[0]

# ============================================================
# Main
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"  Daily Catch-Up Signal Scanner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Monitoring: {', '.join(SIGNAL_TICKERS)}")
    print(f"  Signal condition: {EPS_LOW}% < epsilon < {EPS_HIGH}%")
    print(f"  Suggested hold: 2-3 days")
    print(f"{'='*60}\n")

    returns, tickers = get_data()

    # Train on last 60 days, test on today
    train = returns.iloc[-TRAIN_DAYS-1 : -1]
    test_row = returns.iloc[-1]
    date = returns.index[-1].strftime("%Y-%m-%d")

    input_tickers = [t for t in INPUT_TICKERS if t in tickers]
    epsilon, predicted, actual, pc1_ratio = calc_epsilon(
        train, test_row, input_tickers, tickers
    )

    # Market regime
    print(f"  Market Status | Date: {date}")
    if pc1_ratio > 0.6:
        regime = "High correlation (signals weaker)"
    elif pc1_ratio < 0.4:
        regime = "Dispersed market"
    else:
        regime = "Neutral (signals most reliable)"
    print(f"  PC1 explained: {pc1_ratio*100:.1f}%  ->  {regime}\n")

    # Input stock status
    print(f"  {'Ticker':<8} {'Actual%':<10} Note")
    print(f"  {'-'*35}")
    for t in input_tickers:
        if t in tickers:
            i = tickers.index(t)
            print(f"  {t:<8} {actual[i]:+6.2f}%    (factor input)")

    # Scan for signals
    print(f"\n{'='*60}")
    print(f"  Signal Scan Results")
    print(f"{'='*60}\n")

    signals = []
    no_signals = []

    for t in SIGNAL_TICKERS:
        if t not in tickers:
            continue
        i = tickers.index(t)
        eps = epsilon[i]
        pred = predicted[i]
        act = actual[i]

        if EPS_LOW < eps < EPS_HIGH:
            signals.append((t, act, pred, eps))
        else:
            no_signals.append((t, act, pred, eps))

    if signals:
        print(f"  SIGNALS TRIGGERED (suggested hold: 2-3 days):\n")
        print(f"  {'Ticker':<8} {'Today%':<12} {'Predicted%':<12} {'Epsilon':<12} Action")
        print(f"  {'-'*65}")
        for t, act, pred, eps in sorted(signals, key=lambda x: x[3]):
            strength = "STRONG" if eps < -5 else "MODERATE"
            print(f"  {t:<8} {act:+6.2f}%      {pred:+6.2f}%      {eps:+6.2f}%      {strength} - buy, hold 2-3 days")
        print(f"\n  Options: buy slightly OTM call expiring next Friday, enter before market close")
    else:
        print(f"  No signals today. All monitored stocks tracking the factor normally.\n")

    # Full watchlist overview
    print(f"\n{'='*60}")
    print(f"  Full Watchlist")
    print(f"{'='*60}\n")
    print(f"  {'Ticker':<8} {'Today%':<12} {'Predicted%':<12} {'Epsilon%':<12} Status")
    print(f"  {'-'*60}")

    for t in SIGNAL_TICKERS:
        if t not in tickers:
            continue
        i = tickers.index(t)
        eps = epsilon[i]
        pred = predicted[i]
        act = actual[i]
        if EPS_LOW < eps < EPS_HIGH:
            status = "BUY SIGNAL"
        elif eps <= EPS_LOW:
            status = "ANOMALY (possible earnings)"
        elif eps > 2:
            status = "Outperforming factor"
        else:
            status = "Neutral"
        print(f"  {t:<8} {act:+6.2f}%      {pred:+6.2f}%      {eps:+6.2f}%      {status}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
