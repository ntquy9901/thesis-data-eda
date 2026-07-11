"""Phase 3 — Price Data EDA (per EDA Guide).

Computes returns, log returns, ATR, realized volatility, and the **leakage-safe
volatility prediction targets** rv_t+1 / rv_t+5 / rv_t+10 (future-only returns),
plus rolling diagnostics, ACF/PACF, correlation heatmap, outlier / clustering /
regime analysis.

Outputs (under ``eda_output/price/``):
- ``price_metrics_<ticker>.parquet`` — per-ticker features + targets
- ``acf_pacf_<ticker>.png``, ``rolling_vol.png``, ``corr_heatmap.png``
- ``outliers_<ticker>.csv``, ``findings.md``

Per EDA Guide rule + Arch §16.2 (ADR-006): targets use ONLY future returns;
features aligned to t use info ≤ close of day t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EDA_TICKERS, PRICE_DATA_DIR
from src.eda.common import ensure_output_dirs, phase_output_dir

TARGET_HORIZONS = (1, 5, 10)
ATR_WINDOW = 14
RV_WINDOWS = (5, 20)
ROLLING_WINDOWS = (20, 60)
OUTLIER_SIGMA = 3.0


# ===================== Story 2-1: core metrics + leakage-safe targets =====================
def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Append ``returns`` and ``log_returns`` (sorted by date).

    Non-finite values from zero/negative prices (``-inf``/``inf``/``nan``) are
    masked to NaN so they don't propagate into realized-vol / targets.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    price = out[price_col]
    out["returns"] = price.pct_change()
    out["log_returns"] = np.log(price / price.shift(1))
    # Mask any non-finite (zero/negative prices) → NaN
    out["returns"] = out["returns"].where(np.isfinite(out["returns"]))
    out["log_returns"] = out["log_returns"].where(np.isfinite(out["log_returns"]))
    return out


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range (simple rolling mean of True Range). Pure."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Realized volatility = sqrt(sum of squared log returns) over trailing window. Pure."""
    return np.sqrt((log_returns**2).rolling(window).sum())


def volatility_targets(log_returns: pd.Series, horizons: tuple[int, ...] = TARGET_HORIZONS) -> pd.DataFrame:
    """Leakage-safe forward realized-volatility targets.

    ``rv_t+h`` (aligned to date t) = sqrt( sum( lr^2 over [t+1, t+h] ) ) — uses
    ONLY returns strictly after t. Constructed as a trailing sum shifted back by h,
    so the last h rows are NaN (no future data). Verified by unit test.
    """
    sq = log_returns**2
    out = pd.DataFrame(index=log_returns.index)
    for h in horizons:
        trailing_sum = sq.rolling(h).sum()  # at j: sum(sq[j-h+1 : j+1])
        out[f"rv_t+{h}"] = np.sqrt(trailing_sum.shift(-h))  # move value from t+h → t
    return out


def price_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Full per-ticker feature frame: returns, ATR, realized vol, + targets."""
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"price_metrics requires OHLC columns; missing: {missing}")
    out = add_returns(df)
    out["atr_14"] = average_true_range(out["high"], out["low"], out["close"])
    for w in RV_WINDOWS:
        out[f"realized_vol_{w}d"] = realized_volatility(out["log_returns"], w)
    targets = volatility_targets(out["log_returns"])
    return pd.concat([out, targets], axis=1)


# ===================== Story 2-2: rolling diagnostics + viz =====================
def rolling_stats(series: pd.Series, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Rolling mean + std for each window. Pure."""
    out = {}
    for w in windows:
        out[f"mean_{w}"] = series.rolling(w).mean()
        out[f"std_{w}"] = series.rolling(w).std()
    return pd.DataFrame(out)


def plot_acf_pacf(series: pd.Series, path: Path, lags: int = 30) -> None:
    """Save ACF/PACF plot (drops NaN). Writes a placeholder if < 4 points.

    Side-effect: writes PNG.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    s = series.dropna()
    n = len(s)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    if n < 4:  # too few points for meaningful ACF/PACF — write a placeholder
        for ax in axes:
            ax.text(0.5, 0.5, f"Insufficient data ({n} pts) for ACF/PACF", ha="center", va="center")
            ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    max_lags = max(1, min(lags, n // 2 - 1))
    plot_acf(s, lags=max_lags, ax=axes[0])
    plot_pacf(s, lags=max_lags, ax=axes[1], method="ywm")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cross_ticker_corr_heatmap(returns_by_ticker: dict[str, pd.Series], path: Path) -> None:
    """Heatmap of pairwise return correlations. Side-effect: writes PNG."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(returns_by_ticker)
    corr = frame.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Cross-ticker return correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_vol(vol_by_ticker: dict[str, pd.Series], path: Path) -> None:
    """Rolling-volatility time series, one line per ticker. Side-effect: writes PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for ticker, vol in vol_by_ticker.items():
        vol.plot(ax=ax, label=ticker, alpha=0.8)
    ax.set_title("Realized volatility (20d)")
    ax.set_ylabel("rv_20d")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===================== Story 2-3: outliers / clustering / regime =====================
def detect_outliers(returns: pd.Series, threshold: float = OUTLIER_SIGMA) -> pd.Series:
    """Boolean mask of |z-score| > threshold (robust to NaN). Pure."""
    std = returns.std()
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=returns.index)
    z = (returns - returns.mean()) / std
    return z.abs() > threshold


def volatility_clustering_test(log_returns: pd.Series) -> dict:
    """Ljung-Box on squared returns (volatility clustering proxy). Returns stat + p."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    s = log_returns.dropna() ** 2
    if len(s) < 20 or s.nunique() <= 1:
        return {"lb_stat": None, "lb_pvalue": None, "clustering": "insufficient_data"}
    lb = acorr_ljungbox(s, lags=[10], return_df=True)
    stat = float(lb["lb_stat"].iloc[0])
    p = float(lb["lb_pvalue"].iloc[0])
    if not np.isfinite(stat) or not np.isfinite(p):
        return {"lb_stat": None, "lb_pvalue": None, "clustering": "inconclusive"}
    return {"lb_stat": round(stat, 3), "lb_pvalue": round(p, 4), "clustering": "yes" if p < 0.05 else "no"}


def regime_flags(realized_vol: pd.Series, window: int = 60, n_bins: int = 3) -> pd.Series:
    """Rolling quantile regime label (0=low, n-1=high vol). Pure."""
    if realized_vol.dropna().empty:
        return pd.Series(np.nan, index=realized_vol.index)
    q = realized_vol.rolling(window, min_periods=window).rank(pct=True)
    return (q * n_bins).apply(np.floor).clip(upper=n_bins - 1)


def regime_changes(realized_vol: pd.Series, window: int = 60, n_bins: int = 3) -> pd.DataFrame:
    """Detect regime *transition* points (where the regime label changes).

    Returns a DataFrame of (index, regime) rows at each transition. Pure.
    """
    flags = regime_flags(realized_vol, window, n_bins).dropna()
    if flags.empty:
        return pd.DataFrame(columns=["regime"])
    transitions = flags[flags.diff() != 0].to_frame("regime")
    return transitions


# ===================== runner =====================
def _read_ticker(ticker: str) -> pd.DataFrame:
    return pd.read_csv(PRICE_DATA_DIR / f"{ticker}_ohlcv.csv", encoding="utf-8")


def run_phase() -> list[Path]:
    """Run Phase 3 for all EDA tickers; write parquet + plots + findings."""
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()
    outdir = phase_output_dir("price")
    written: list[Path] = []
    returns_by_ticker: dict[str, pd.Series] = {}
    vol_by_ticker: dict[str, pd.Series] = {}
    findings: list[str] = ["# Phase 3 — Price EDA findings\n"]

    for ticker in EDA_TICKERS:
        path = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if not path.exists():
            continue
        try:
            df = _read_ticker(ticker)
            metrics = price_metrics(df)
        except (ValueError, KeyError) as e:  # one bad ticker must not abort the phase
            findings.append(f"## {ticker}\n- SKIPPED: {e}\n")
            continue

        pq = outdir / f"price_metrics_{ticker}.parquet"
        metrics.to_parquet(pq, index=False)
        written.append(pq)

        returns_by_ticker[ticker] = metrics.set_index("date")["log_returns"]
        vol_by_ticker[ticker] = metrics.set_index("date")["realized_vol_20d"]

        # Story 2-2: persist rolling stats (20/60d)
        rs = rolling_stats(metrics["log_returns"])
        rs.insert(0, "date", metrics["date"].values)
        rs_path = outdir / f"rolling_stats_{ticker}.csv"
        rs.to_csv(rs_path, index=False, encoding="utf-8")
        written.append(rs_path)

        plot_acf_pacf(metrics["log_returns"], outdir / f"acf_pacf_{ticker}.png")
        written.append(outdir / f"acf_pacf_{ticker}.png")

        outliers = metrics.loc[detect_outliers(metrics["log_returns"]), ["date", "log_returns"]]
        oc = outdir / f"outliers_{ticker}.csv"
        outliers.to_csv(oc, index=False, encoding="utf-8")
        written.append(oc)

        cl = volatility_clustering_test(metrics["log_returns"])
        # Story 2-3: regime change detection
        ch = regime_changes(metrics.set_index("date")["realized_vol_20d"])
        n_regime = len(ch)
        first_shift = str(ch.index[0])[:10] if n_regime else "n/a"
        n_out = len(outliers)
        findings.append(
            f"## {ticker}\n- rows: {len(metrics)} ({metrics['date'].min()} → {metrics['date'].max()})\n"
            f"- outliers (>{OUTLIER_SIGMA}σ): {n_out}\n"
            f"- vol clustering (Ljung-Box sq-ret, lag10): stat={cl['lb_stat']}, "
            f"p={cl['lb_pvalue']} → {cl['clustering']}\n"
            f"- regime shifts: {n_regime} (first: {first_shift})\n"
            f"- rv_t+10 mean: {metrics['rv_t+10'].mean():.5f} (NaN-tail: "
            f"{metrics['rv_t+10'].isna().sum()})\n"
        )

    if returns_by_ticker:
        heatmap = outdir / "corr_heatmap.png"
        cross_ticker_corr_heatmap(returns_by_ticker, heatmap)
        written.append(heatmap)
        rolling = outdir / "rolling_vol.png"
        plot_rolling_vol(vol_by_ticker, rolling)
        written.append(rolling)

    findings_path = outdir / "findings.md"
    findings_path.write_text("\n".join(findings), encoding="utf-8")
    written.append(findings_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
