"""
PCA Cross-Sectional Strategy Backtest

Strategy: On any given day, some stocks outperform the latent market factor.
          Find stocks that underperformed (most negative epsilon) and check whether
          they catch up over the next 1-3 days.

Signal condition: -20% < epsilon < -1%
  - Filters out earnings-driven anomalies (epsilon < -20%)
  - Only keeps meaningful divergence from the factor

Validated stocks (>= 60% win rate): AVGO, MRVL, APP, SMCI, UNH, BA

"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


TICKERS = [
    "NVDA", "AMD", "AVGO", "MRVL", "MU", "SMCI",
    "PLTR", "ORCL", "MSFT", "META", "APP",
    "GOOGL", "AMZN", "DELL", "QQQ",
    "MSTR", "COIN", "TSLA", "BA", "UNH",
]

TRAIN_DAYS = 60 # PCA rolling training window
TEST_DAYS = 60 # Number of days to backtest
N_FACTORS = 3 # Number of principal components
INPUT_TICKERS = ["NVDA", "AMD", "QQQ"]  # Stocks used to infer latent factor
EPS_THRESHOLD = -1.0  # Epsilon must be below this to trigger a signal (%)
HOLD_DAYS = [1, 2, 3]  # Forward return horizons to evaluate

def get_data():
    end = datetime.now()
    start = end - timedelta(days=(TRAIN_DAYS + TEST_DAYS) * 2 + 30)
    print("Downloading data...")
    raw = yf.download(TICKERS, start=start, end=end, progress=False, timeout=10)["Close"]
    raw = raw.dropna(axis=1, thresh=int((TRAIN_DAYS + TEST_DAYS) * 0.9))
    valid = list(raw.columns)
    returns = raw.pct_change().dropna()
    print(f"  {len(valid)} stocks, {len(returns)} trading days\n")
    return returns, valid


# PCA + Epsilon Calculation

def calc_epsilon(train_returns, test_row, input_tickers, tickers, n_factors=3):
    """
    Infer latent factor F from input stocks via least squares,
    then compute epsilon = actual - predicted for all stocks.
    """
    X = train_returns.values
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    explained = eigenvalues / eigenvalues.sum()

    # Infer factor F from input stocks
    input_idx = [tickers.index(t) for t in input_tickers if t in tickers]
    r_input = test_row.values[input_idx]
    B_input = eigenvectors[input_idx, :n_factors]
    F, _, _, _ = np.linalg.lstsq(B_input, r_input, rcond=None)

    # Predict all stocks
    B_all = eigenvectors[:, :n_factors]
    predicted = B_all @ F
    actual = test_row.values
    epsilon = actual - predicted

    return predicted * 100, actual * 100, epsilon * 100, explained[0]


# Backtest

def run_backtest(returns, tickers):
    input_tickers = [t for t in INPUT_TICKERS if t in tickers]

    total_days = len(returns)
    signals = []

    print(f"{'='*65}")
    print(f"  PCA Cross-Sectional Strategy Backtest")
    print(f"  Train window: {TRAIN_DAYS} days  |  Test period: {TEST_DAYS} days")
    print(f"  Signal: epsilon < {EPS_THRESHOLD}%  and  epsilon > -20%")
    print(f"  Hold periods: {HOLD_DAYS} days")
    print(f"{'='*65}\n")

    for i in range(TEST_DAYS):
        test_idx = total_days - TEST_DAYS + i
        if test_idx < TRAIN_DAYS:
            continue
        if test_idx + max(HOLD_DAYS) >= total_days:
            continue

        train = returns.iloc[test_idx - TRAIN_DAYS : test_idx]
        test_row = returns.iloc[test_idx]
        date = returns.index[test_idx]

        predicted, actual, epsilon, pc1_ratio = calc_epsilon(
            train, test_row, input_tickers, tickers, N_FACTORS
        )

        for j, t in enumerate(tickers):
            if t in input_tickers:
                continue
            eps = epsilon[j]
            pred = predicted[j]
            act = actual[j]

            # Signal: normal underperformance only (exclude earnings anomalies)
            if eps < EPS_THRESHOLD and eps > -20.0:
                future_returns = {}
                for hold in HOLD_DAYS:
                    if test_idx + hold < total_days:
                        future_ret = returns.iloc[test_idx + 1 : test_idx + hold + 1][t].sum() * 100
                        future_returns[f"future_{hold}d"] = future_ret
                    else:
                        future_returns[f"future_{hold}d"] = np.nan

                signals.append({
                    "date": date,
                    "ticker": t,
                    "predicted": pred,
                    "actual_day0": act,
                    "epsilon": eps,
                    "pc1_ratio": pc1_ratio,
                    **future_returns
                })

    if not signals:
        print("No signals triggered. Try lowering EPS_THRESHOLD.")
        return

    df = pd.DataFrame(signals)
    print(f"Total signals: {len(df)}\n")

          
    # Summary 1: Overall win rate by hold period

    print(f"{'='*65}")
    print(f"  Overall Win Rate by Hold Period")
    print(f"{'='*65}\n")
    print(f"  {'Hold':<10} {'Win Rate':<10} {'Avg Return':<14} {'Median':<12} {'Max':<10} Min")
    print(f"  {'-'*60}")

    for hold in HOLD_DAYS:
        col = f"future_{hold}d"
        valid = df[col].dropna()
        win_rate = (valid > 0).mean() * 100
        avg_ret  = valid.mean()
        med_ret  = valid.median()
        max_ret  = valid.max()
        min_ret  = valid.min()
        print(f"  {hold}d{'':<8} {win_rate:.1f}%{'':<5} "
              f"{avg_ret:+.2f}%{'':<9} {med_ret:+.2f}%{'':<7} "
              f"{max_ret:+.1f}%{'':<5} {min_ret:+.1f}%")


    # Summary 2: Win rate by ticker

    print(f"\n{'='*65}")
    print(f"  Win Rate by Ticker (1-day hold)")
    print(f"{'='*65}\n")

    by_ticker = df.groupby("ticker").agg(
        count=("epsilon", "count"),
        avg_eps=("epsilon", lambda x: f"{x.mean():.1f}%"),
        win_rate=("future_1d", lambda x: f"{(x>0).mean()*100:.0f}%"),
        avg_return=("future_1d", lambda x: f"{x.mean():+.2f}%"),
    ).sort_values("win_rate", ascending=False)
    by_ticker.columns = ["Signals", "Avg Epsilon", "Win Rate", "Avg Return"]
    print(by_ticker.to_string())

    # Summary 3: Epsilon magnitude vs. catch-up return

    print(f"\n{'='*65}")
    print(f"  Epsilon Buckets vs. Next-Day Return")
    print(f"{'='*65}\n")

    df["eps_bucket"] = pd.cut(df["epsilon"],
        bins=[-999, -5, -3, -2, -1, 0],
        labels=["eps < -5%", "-5 to -3%", "-3 to -2%", "-2 to -1%", "-1 to 0%"])

    bucket_stats = df.groupby("eps_bucket", observed=True).agg(
        count=("future_1d", "count"),
        win_rate=("future_1d", lambda x: f"{(x>0).mean()*100:.0f}%"),
        avg_return=("future_1d", lambda x: f"{x.mean():+.2f}%"),
    )
    bucket_stats.columns = ["Signals", "Win Rate", "Avg Next-Day Return"]
    print(bucket_stats.to_string())
    print(f"\n  -> Does more negative epsilon lead to stronger catch-up? See above.")


    # Summary 4: Top 20 strongest signals

    print(f"\n{'='*65}")
    print(f"  Top 20 Strongest Signals (most negative epsilon)")
    print(f"{'='*65}\n")

    top_signals = df.nsmallest(20, "epsilon")[
        ["date", "ticker", "predicted", "actual_day0", "epsilon", "future_1d", "future_2d"]
    ].copy()
    top_signals["date"] = top_signals["date"].dt.strftime("%Y-%m-%d")
    top_signals.columns = ["Date", "Ticker", "Predicted%", "Actual%", "Epsilon%", "1d Fwd%", "2d Fwd%"]
    top_signals = top_signals.round(2)
    print(top_signals.to_string(index=False))


    # Summary 5: Market regime vs. signal quality

    print(f"\n{'='*65}")
    print(f"  Market Regime vs. Signal Quality")
    print(f"{'='*65}\n")

    df["regime"] = df["pc1_ratio"].apply(
        lambda x: "High corr (>60%)" if x > 0.6 else ("Dispersed (<40%)" if x < 0.4 else "Neutral")
    )
    regime_stats = df.groupby("regime").agg(
        count=("future_1d", "count"),
        win_rate=("future_1d", lambda x: f"{(x>0).mean()*100:.0f}%"),
        avg_return=("future_1d", lambda x: f"{x.mean():+.2f}%"),
    )
    regime_stats.columns = ["Signals", "Win Rate", "Avg Return"]
    print(regime_stats.to_string())
    print(f"\n  -> Signals tend to be more reliable in dispersed markets (low PC1 ratio).")

    # Final summary

    win_1d = (df["future_1d"] > 0).mean() * 100
    avg_1d = df["future_1d"].mean()

    print(f"\n{'='*65}")
    print(f"  Conclusion")
    print(f"{'='*65}\n")
    print(f"  1-day win rate:    {win_1d:.1f}%")
    print(f"  1-day avg return:  {avg_1d:+.2f}%")

    if win_1d > 60 and avg_1d > 0.5:
        print(f"\n  Cross-sectional signal is valid.")
        print(f"  Stocks with epsilon < {EPS_THRESHOLD}% tend to catch up the following day.")
    elif win_1d > 55:
        print(f"\n  Signal has moderate predictive power. Use with additional filters.")
    else:
        print(f"\n  Signal is not robust. Consider tightening filter conditions.")

    print(f"\n{'='*65}\n")
    return df

# Entry Point

if __name__ == "__main__":
    returns, tickers = get_data()
    df = run_backtest(returns, tickers)
