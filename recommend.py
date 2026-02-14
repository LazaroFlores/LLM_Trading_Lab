from __future__ import annotations

import argparse
import json
import sys
import re
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore


DEFAULT_UNIVERSE_PATH = Path("config/universe.txt")
DEFAULT_HOLDINGS_PATH = Path("config/holdings.txt")
DEFAULT_HOLDINGS_STRATEGIES_PATH = Path("config/holdings_strategies.json")
DEFAULT_OUTPUT_DIR = Path("runs")
DEFAULT_DAILY_UPDATES_CSV = Path("Experiments/chatgpt_micro-cap/csv_files/Daily Updates.csv")


def _init_yfinance_cache(cache_dir: Path) -> None:
    """
    yfinance maintains sqlite caches under platformdirs' user cache dir.
    In sandboxed environments that path may be read-only, causing downloads to fail.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        import yfinance.cache as yf_cache  # type: ignore

        yf_cache.set_cache_location(str(cache_dir))
    except Exception:
        # Best-effort only; yfinance will still work if its default cache is writable.
        pass


def _best_effort_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _today() -> date:
    return datetime.now().date()


def _coerce_asof(asof: str | None) -> date:
    if asof is None:
        d = _today()
    else:
        d = datetime.strptime(asof, "%Y-%m-%d").date()
    # weekend -> prior Friday
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d - timedelta(days=2)
    return d


def _read_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    def _convert_prefixed_symbol(sym: str) -> str | None:
        s = sym.strip().upper()
        if not s:
            return None

        # Exchange prefixes from other projects (e.g., "NYSE:KO", "BMV:AUTLAN/B")
        if ":" in s:
            prefix, rest = s.split(":", 1)
            rest = rest.strip().upper()
            if prefix == "BMV":
                core = re.sub(r"[^A-Z0-9]", "", rest.replace("/", ""))
                return f"{core}.MX" if core else None
            # Common US prefixes -> drop prefix
            if prefix in {"NYSE", "NASDAQ", "AMEX", "BATS", "ARCA"}:
                s = rest
            else:
                s = rest

        # yfinance uses "-" for some class shares (e.g., BRK-B). Do not break suffixes like ".MX".
        if s.count(".") == 1:
            left, right = s.split(".", 1)
            if left and right and len(right) == 1:
                s = f"{left}-{right}"
        s = s.strip().upper()
        return s or None

    out: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow inline comments: "KO  # Coca-Cola"
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Allow comma-separated tickers per line
        parts = [p.strip() for p in line.split(",") if p.strip()]
        for p in parts:
            converted = _convert_prefixed_symbol(p)
            if converted:
                out.append(converted)
    # stable order, drop duplicates
    return list(dict.fromkeys(out))


def _read_holdings_strategy_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, str] = {}
    for tk, sk in raw.items():
        ticker = str(tk).strip().upper()
        strategy_key = str(sk).strip().lower()
        if not ticker or not strategy_key:
            continue
        out[ticker] = strategy_key
    return out


def _normalize_tickers(series: pd.Series) -> list[str]:
    s = series.astype(str).str.strip().str.upper()
    s = s.replace({"": np.nan, "NAN": np.nan})
    tickers = [t for t in s.dropna().tolist() if t and t != "TOTAL"]
    return list(dict.fromkeys(tickers))


def _read_daily_updates_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError(f"Daily Updates CSV invalido: falta columna 'Ticker' ({path})")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def universe_from_daily_updates(path: Path) -> list[str]:
    df = _read_daily_updates_df(path)
    return _normalize_tickers(df["Ticker"])


def holdings_from_daily_updates(path: Path) -> tuple[list[str], dict[str, float]]:
    df = _read_daily_updates_df(path)
    if "Date" not in df.columns or df["Date"].dropna().empty:
        raise ValueError(f"Daily Updates CSV invalido: no se pudo parsear 'Date' ({path})")

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()
    latest = latest[latest["Ticker"].astype(str).str.upper() != "TOTAL"].copy()

    # Excluir filas marcadas como SELL si existe la columna Action
    if "Action" in latest.columns:
        sold = latest["Action"].astype(str).str.upper().str.startswith("SELL")
        latest = latest[~sold].copy()

    # Excluir shares <= 0 si existe la columna Shares
    if "Shares" in latest.columns:
        latest["Shares"] = pd.to_numeric(latest["Shares"], errors="coerce")
        latest = latest[latest["Shares"].fillna(0) > 0].copy()

    tickers = _normalize_tickers(latest["Ticker"])

    stop_losses: dict[str, float] = {}
    if "Stop Loss" in latest.columns:
        sl = pd.to_numeric(latest["Stop Loss"], errors="coerce")
        for t, v in zip(latest["Ticker"].astype(str).str.upper().tolist(), sl.tolist(), strict=False):
            if t and t != "TOTAL" and v is not None and np.isfinite(v) and float(v) > 0:
                stop_losses[t] = float(v)

    return tickers, stop_losses


def _format_money(x: float) -> str:
    return f"${x:,.2f}"


def _safe_float(x: object) -> float | None:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _download_history(
    tickers: list[str],
    start: date,
    end_inclusive: date,
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    # yfinance end is exclusive
    end = end_inclusive + timedelta(days=1)
    try:
        # Batch download; returns MultiIndex columns when multiple tickers
        df = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {}

    out: dict[str, pd.DataFrame] = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return out

    # If single ticker, columns are not nested
    if not isinstance(df.columns, pd.MultiIndex):
        t = tickers[0]
        out[t] = df.copy()
        return out

    for t in tickers:
        if t not in df.columns.get_level_values(0):
            continue
        sub = df[t].dropna(how="all")
        if isinstance(sub, pd.DataFrame) and not sub.empty:
            out[t] = sub.copy()
    return out


def _chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@dataclass(frozen=True)
class ScoredTicker:
    ticker: str
    score: float
    close: float
    mom_126d: float
    vol_20d: float
    dollar_vol_20d: float
    trend_ok: bool
    reason: str


def _compute_features(ticker: str, hist: pd.DataFrame) -> ScoredTicker | None:
    if hist is None or hist.empty:
        return None

    hist = hist.copy()
    hist = hist.sort_index()
    for col in ("Close", "Volume"):
        if col not in hist.columns:
            return None

    close = pd.to_numeric(hist["Close"], errors="coerce")
    volume = pd.to_numeric(hist["Volume"], errors="coerce")
    if close.dropna().empty:
        return None

    last_close = float(close.dropna().iloc[-1])
    ret = close.pct_change()

    def _ret_n(n: int) -> float | None:
        if len(close) < n + 1:
            return None
        a = float(close.dropna().iloc[-(n + 1)])
        b = float(close.dropna().iloc[-1])
        if a <= 0:
            return None
        return (b / a) - 1.0

    mom_126 = _ret_n(126)
    if mom_126 is None:
        return None

    vol_20 = float(ret.tail(20).std(ddof=0)) if len(ret.dropna()) >= 20 else float("nan")
    dv_20 = float((close * volume).tail(20).mean()) if len(close.dropna()) >= 20 else float("nan")

    sma50 = float(close.tail(50).mean()) if len(close.dropna()) >= 50 else float("nan")
    sma200 = float(close.tail(200).mean()) if len(close.dropna()) >= 200 else float("nan")
    trend_ok = bool(np.isfinite(sma50) and np.isfinite(sma200) and (last_close > sma50) and (sma50 > sma200))

    # Score components (raw; ranking happens later)
    reason = f"mom_126d={mom_126:+.1%}, trend={'OK' if trend_ok else 'NO'}"
    return ScoredTicker(
        ticker=ticker,
        score=float("nan"),
        close=last_close,
        mom_126d=float(mom_126),
        vol_20d=float(vol_20),
        dollar_vol_20d=float(dv_20),
        trend_ok=trend_ok,
        reason=reason,
    )


def _rank_score(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return s.rank(ascending=ascending, method="average", na_option="bottom")


def _pct_rank_higher_is_better(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    # Higher values -> higher percentile rank (0..1)
    return s.rank(ascending=True, pct=True, method="average", na_option="bottom")


def build_top_recommendations(
    universe: list[str],
    asof: date,
    lookback_days: int = 365 * 2,
    chunk_size: int = 80,
    min_dollar_vol_20d: float = 500_000.0,
    top_n: int = 5,
) -> list[ScoredTicker]:
    if not universe:
        return []

    start = asof - timedelta(days=lookback_days)
    features: list[ScoredTicker] = []

    for chunk in _chunked(universe, chunk_size):
        data = _download_history(chunk, start=start, end_inclusive=asof)
        for t, hist in data.items():
            ft = _compute_features(t, hist)
            if ft is None:
                continue
            # Liquidity screen
            if not np.isfinite(ft.dollar_vol_20d) or ft.dollar_vol_20d < min_dollar_vol_20d:
                continue
            features.append(ft)

    if not features:
        return []

    df = pd.DataFrame([f.__dict__ for f in features])
    # risk-adjusted momentum: mom / vol
    df["mom_over_vol"] = df["mom_126d"] / df["vol_20d"].replace(0.0, np.nan)

    # ranks: higher is better
    r_mom = _pct_rank_higher_is_better(df, "mom_126d")
    r_risk = _pct_rank_higher_is_better(df, "mom_over_vol")
    r_liq = _pct_rank_higher_is_better(df, "dollar_vol_20d")
    r_trend = df["trend_ok"].astype(int).rank(ascending=True, pct=True, method="average")

    df["score"] = 0.45 * r_mom + 0.35 * r_risk + 0.15 * r_liq + 0.05 * r_trend
    df = df.sort_values("score", ascending=False).head(top_n)

    out: list[ScoredTicker] = []
    for row in df.to_dict(orient="records"):
        out.append(
            ScoredTicker(
                ticker=str(row["ticker"]),
                score=float(row["score"]),
                close=float(row["close"]),
                mom_126d=float(row["mom_126d"]),
                vol_20d=float(row["vol_20d"]) if np.isfinite(row["vol_20d"]) else float("nan"),
                dollar_vol_20d=float(row["dollar_vol_20d"]),
                trend_ok=bool(row["trend_ok"]),
                reason=str(row.get("reason", "")),
            )
        )
    return out


@dataclass(frozen=True)
class HoldingAction:
    ticker: str
    action: str  # BUY / SELL / HOLD
    close: float
    stop_loss: float | None
    reason: str
    strategy: str = "canal_hibrido"


@dataclass(frozen=True)
class HoldingBacktest:
    ticker: str
    initial_capital_1y: float
    strategy_final_capital_1y: float
    buy_hold_final_capital_1y: float
    strategy_return_1y: float
    buy_hold_return_1y: float
    max_drawdown_1y: float
    trades_1y: int
    strategy: str = "canal_hibrido"
    strategy_leverage: float = 1.0
    target_return_1y: float = 0.20
    target_met_1y: bool = False


@dataclass(frozen=True)
class HoldingStrategyProfile:
    key: str
    label: str
    style: str  # breakout | pullback | hybrid | mean_reversion | regime_momentum | ko_turbo | ko_fib618 | ko_candles_book | candles_book_pure | candles_book_context_min | channel_reversal | channel_pivot_reversal | pivot_reversal
    channel_window: int
    floor_pct: float
    ceiling_pct: float
    fast_sma: int
    slow_sma: int
    breakout_up_buffer: float
    breakout_down_buffer: float
    leverage: float = 1.0
    pivot_left: int = 7
    pivot_right: int = 3


@dataclass(frozen=True)
class HoldingStrategyPlan:
    ticker: str
    strategy_key: str
    strategy_label: str
    strategy_style: str
    annualized_volatility: float | None
    train_return: float | None
    train_max_drawdown: float | None
    train_trades: int
    target_return: float
    target_met: bool
    forced: bool = False


STRATEGY_PROFILES: tuple[HoldingStrategyProfile, ...] = (
    HoldingStrategyProfile(
        key="breakout_fast",
        label="Breakout Rapido",
        style="breakout",
        channel_window=20,
        floor_pct=0.20,
        ceiling_pct=0.80,
        fast_sma=20,
        slow_sma=100,
        breakout_up_buffer=0.006,
        breakout_down_buffer=0.006,
    ),
    HoldingStrategyProfile(
        key="breakout_swing",
        label="Breakout Swing",
        style="breakout",
        channel_window=34,
        floor_pct=0.22,
        ceiling_pct=0.78,
        fast_sma=30,
        slow_sma=150,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
    ),
    HoldingStrategyProfile(
        key="pullback_trend",
        label="Pullback Tendencial",
        style="pullback",
        channel_window=55,
        floor_pct=0.35,
        ceiling_pct=0.72,
        fast_sma=50,
        slow_sma=200,
        breakout_up_buffer=0.005,
        breakout_down_buffer=0.005,
    ),
    HoldingStrategyProfile(
        key="hybrid_channel",
        label="Canal Hibrido",
        style="hybrid",
        channel_window=55,
        floor_pct=0.27,
        ceiling_pct=0.75,
        fast_sma=50,
        slow_sma=200,
        breakout_up_buffer=0.005,
        breakout_down_buffer=0.005,
    ),
    HoldingStrategyProfile(
        key="mean_reversion",
        label="Mean Reversion",
        style="mean_reversion",
        channel_window=40,
        floor_pct=0.18,
        ceiling_pct=0.82,
        fast_sma=30,
        slow_sma=120,
        breakout_up_buffer=0.008,
        breakout_down_buffer=0.008,
    ),
    HoldingStrategyProfile(
        key="mx_defensive",
        label="MX Defensiva",
        style="pullback",
        channel_window=70,
        floor_pct=0.30,
        ceiling_pct=0.78,
        fast_sma=40,
        slow_sma=180,
        breakout_up_buffer=0.006,
        breakout_down_buffer=0.006,
    ),
    HoldingStrategyProfile(
        key="regime_momentum",
        label="Regime Momentum",
        style="regime_momentum",
        channel_window=45,
        floor_pct=0.30,
        ceiling_pct=0.78,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
    ),
    HoldingStrategyProfile(
        key="ko_turbo",
        label="KO Turbo",
        style="ko_turbo",
        channel_window=30,
        floor_pct=0.35,
        ceiling_pct=0.90,
        fast_sma=13,
        slow_sma=55,
        breakout_up_buffer=0.003,
        breakout_down_buffer=0.009,
        leverage=2.5,
    ),
    HoldingStrategyProfile(
        key="ko_fib618",
        label="KO Fib 61.8",
        style="ko_fib618",
        channel_window=55,
        floor_pct=0.28,
        ceiling_pct=0.86,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=2.5,
    ),
    HoldingStrategyProfile(
        key="ko_channel_reversal",
        label="KO Canal Reversal",
        style="channel_reversal",
        channel_window=55,
        floor_pct=0.22,
        ceiling_pct=0.78,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=2.5,
    ),
    HoldingStrategyProfile(
        key="ko_channel_pivots",
        label="KO Canal + Pivots",
        style="channel_pivot_reversal",
        channel_window=55,
        floor_pct=0.22,
        ceiling_pct=0.78,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=2.5,
        pivot_left=7,
        pivot_right=3,
    ),
    HoldingStrategyProfile(
        key="ko_candles_book",
        label="KO Velas (eBook)",
        style="ko_candles_book",
        channel_window=55,
        floor_pct=0.26,
        ceiling_pct=0.74,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=2.5,
    ),
    HoldingStrategyProfile(
        key="candles_book_pdf",
        label="Velas PDF (Puro)",
        style="candles_book_pure",
        channel_window=55,
        floor_pct=0.25,
        ceiling_pct=0.75,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=1.5,
    ),
    HoldingStrategyProfile(
        key="candles_book_pdf_ctx",
        label="Velas PDF + Contexto",
        style="candles_book_context_min",
        channel_window=55,
        floor_pct=0.30,
        ceiling_pct=0.70,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=1.5,
    ),
    HoldingStrategyProfile(
        key="ko_pivot_reversal",
        label="KO Pivots Locales",
        style="pivot_reversal",
        channel_window=55,
        floor_pct=0.45,
        ceiling_pct=0.55,
        fast_sma=21,
        slow_sma=89,
        breakout_up_buffer=0.004,
        breakout_down_buffer=0.004,
        leverage=2.5,
        pivot_left=7,
        pivot_right=3,
    ),
    HoldingStrategyProfile(
        key="meli_turbo",
        label="MELI Turbo",
        style="mean_reversion",
        channel_window=40,
        floor_pct=0.18,
        ceiling_pct=0.82,
        fast_sma=30,
        slow_sma=120,
        breakout_up_buffer=0.008,
        breakout_down_buffer=0.008,
        leverage=3.0,
    ),
)
STRATEGY_PROFILE_BY_KEY: dict[str, HoldingStrategyProfile] = {p.key: p for p in STRATEGY_PROFILES}
DEFAULT_PROFILE_KEY = "hybrid_channel"


def _prepare_ohlc(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    base_cols = ["Close", "High", "Low"]
    if not all(c in hist.columns for c in base_cols):
        return pd.DataFrame()
    cols = ["Open", "Close", "High", "Low"]
    keep = [c for c in cols if c in hist.columns]
    out = hist[keep].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=base_cols).sort_index()
    return out


def _channel_indicators(
    ohlc: pd.DataFrame,
    profile: HoldingStrategyProfile | None = None,
) -> pd.DataFrame:
    p = profile or STRATEGY_PROFILE_BY_KEY[DEFAULT_PROFILE_KEY]
    d = ohlc.copy()
    d["sma_fast"] = d["Close"].rolling(p.fast_sma, min_periods=p.fast_sma).mean()
    d["sma_slow"] = d["Close"].rolling(p.slow_sma, min_periods=p.slow_sma).mean()
    d["sma50"] = d["sma_fast"]
    d["sma200"] = d["sma_slow"]
    d["channel_high"] = d["High"].rolling(p.channel_window, min_periods=p.channel_window).max()
    d["channel_low"] = d["Low"].rolling(p.channel_window, min_periods=p.channel_window).min()
    width = (d["channel_high"] - d["channel_low"]).clip(lower=np.nan)
    d["channel_width"] = width
    d["zone"] = (d["Close"] - d["channel_low"]) / d["channel_width"]

    d["trend_up"] = (d["sma_fast"] > d["sma_slow"]) & (d["Close"] > d["sma_fast"])
    d["trend_down"] = (d["sma_fast"] < d["sma_slow"]) & (d["Close"] < d["sma_fast"])

    # Breakouts use previous channel to avoid same-bar lookahead.
    d["breakout_up"] = d["Close"] > (d["channel_high"].shift(1) * (1.0 + p.breakout_up_buffer))
    d["breakout_down"] = d["Close"] < (d["channel_low"].shift(1) * (1.0 - p.breakout_down_buffer))

    # Fibonacci retracements over the rolling channel range.
    d["fib_236"] = d["channel_high"] - (d["channel_width"] * 0.236)
    d["fib_382"] = d["channel_high"] - (d["channel_width"] * 0.382)
    d["fib_500"] = d["channel_high"] - (d["channel_width"] * 0.500)
    d["fib_618"] = d["channel_high"] - (d["channel_width"] * 0.618)
    d["fib_786"] = d["channel_high"] - (d["channel_width"] * 0.786)
    d["fib_support"] = (d["Close"] <= d["fib_618"]) & (d["Close"] >= d["fib_786"])
    d["fib_resistance"] = (d["Close"] <= d["fib_236"]) & (d["Close"] >= d["fib_382"])
    d["fib_momo_up"] = d["Close"] >= d["fib_500"]
    d["fib_momo_down"] = d["Close"] <= d["fib_500"]

    d["near_floor"] = d["zone"] <= p.floor_pct
    d["near_ceiling"] = d["zone"] >= p.ceiling_pct
    d["exit_long"] = (
        (d["Close"] < d["channel_low"] * (1.0 - p.breakout_down_buffer))
        | (d["Close"] < d["fib_786"] * (1.0 - p.breakout_down_buffer))
        | (d["breakout_down"] & d["trend_down"])
    )
    d["exit_short"] = (
        (d["Close"] > d["channel_high"] * (1.0 + p.breakout_up_buffer))
        | (d["Close"] > d["fib_236"] * (1.0 + p.breakout_up_buffer))
        | (d["breakout_up"] & d["trend_up"])
    )

    delta = d["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    d["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    d["strategy_profile_key"] = p.key
    d["strategy_profile_label"] = p.label

    # Local pivots (floors/ceilings) confirmed after pivot_right bars (no lookahead at signal time).
    # A pivot low/high is defined at the center of a (L+R+1) window; we act when it's confirmed.
    w = int(max(3, (int(p.pivot_left) + int(p.pivot_right) + 1)))
    if w >= 3 and int(p.pivot_left) >= 1 and int(p.pivot_right) >= 1:
        low_roll = d["Low"].rolling(w, center=True, min_periods=w).min()
        high_roll = d["High"].rolling(w, center=True, min_periods=w).max()
        pivot_low_center = (d["Low"] == low_roll)
        pivot_high_center = (d["High"] == high_roll)
        # Use pandas nullable boolean to avoid dtype downcast warnings on fillna.
        d["pivot_low"] = pivot_low_center.shift(int(p.pivot_right)).astype("boolean").fillna(False).astype(bool)
        d["pivot_high"] = pivot_high_center.shift(int(p.pivot_right)).astype("boolean").fillna(False).astype(bool)
    else:
        d["pivot_low"] = False
        d["pivot_high"] = False
    return d


def _signals_from_profile(ind: pd.DataFrame, profile: HoldingStrategyProfile) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = ind.index
    empty_false = pd.Series(False, index=idx)
    if ind.empty:
        return empty_false, empty_false, empty_false, pd.Series(index=idx, dtype="object")

    zone = pd.to_numeric(ind["zone"], errors="coerce").fillna(0.5)
    trend_up = ind["trend_up"].fillna(False)
    trend_down = ind["trend_down"].fillna(False)
    near_floor = ind["near_floor"].fillna(False)
    near_ceiling = ind["near_ceiling"].fillna(False)
    fib_support = ind["fib_support"].fillna(False)
    fib_resistance = ind["fib_resistance"].fillna(False)
    fib_momo_up = ind["fib_momo_up"].fillna(False)
    fib_momo_down = ind["fib_momo_down"].fillna(False)
    breakout_up = ind["breakout_up"].fillna(False)
    breakout_down = ind["breakout_down"].fillna(False)
    exit_long = ind["exit_long"].fillna(False)
    channel_high = pd.to_numeric(ind.get("channel_high"), errors="coerce")
    channel_low = pd.to_numeric(ind.get("channel_low"), errors="coerce")
    pivot_low = ind.get("pivot_low", pd.Series(False, index=idx)).fillna(False)
    pivot_high = ind.get("pivot_high", pd.Series(False, index=idx)).fillna(False)
    rsi14 = (
        pd.to_numeric(ind["rsi14"], errors="coerce")
        if "rsi14" in ind.columns
        else pd.Series(np.nan, index=idx, dtype=float)
    )
    close = pd.to_numeric(ind["Close"], errors="coerce")
    open_ = pd.to_numeric(ind.get("Open"), errors="coerce") if "Open" in ind.columns else pd.Series(np.nan, index=idx)
    high_ = pd.to_numeric(ind.get("High"), errors="coerce") if "High" in ind.columns else pd.Series(np.nan, index=idx)
    low_ = pd.to_numeric(ind.get("Low"), errors="coerce") if "Low" in ind.columns else pd.Series(np.nan, index=idx)
    fib_618 = (
        pd.to_numeric(ind["fib_618"], errors="coerce")
        if "fib_618" in ind.columns
        else pd.Series(np.nan, index=idx, dtype=float)
    )
    mom_20 = close.pct_change(20)
    mom_63 = close.pct_change(63)

    breakout_buy = breakout_up & trend_up
    breakout_sell = breakout_down & trend_down
    pullback_buy = (near_floor | fib_support) & trend_up
    pullback_sell = (near_ceiling | fib_resistance) & trend_down
    momentum_buy = fib_momo_up & trend_up & (zone >= 0.55)
    momentum_sell = fib_momo_down & trend_down & (zone <= 0.45)
    mean_buy = (near_floor | fib_support) & (rsi14 <= 38.0)
    mean_sell = (near_ceiling | fib_resistance) & (rsi14 >= 62.0)
    regime_bull = trend_up & (mom_63.fillna(0.0) > 0.0) & (rsi14.fillna(50.0) >= 48.0)
    regime_entry = regime_bull & (
        breakout_up
        | (fib_momo_up & (zone >= 0.56))
        | ((near_floor | fib_support) & (rsi14 >= 44.0) & (rsi14 <= 60.0))
    )
    regime_exit = (
        breakout_down
        | (trend_down & fib_momo_down)
        | (rsi14.fillna(50.0) < 43.0)
        | ((mom_20.fillna(0.0) < -0.05) & (zone <= 0.42))
    )
    turbo_entry = (
        trend_up
        | fib_momo_up
        | (zone >= 0.50)
        | ((rsi14.fillna(50.0) >= 45.0) & (mom_20.fillna(0.0) >= -0.02))
    )
    turbo_exit = (
        breakout_down
        & (
            trend_down
            | (rsi14.fillna(50.0) < 40.0)
            | (mom_63.fillna(0.0) < -0.06)
        )
    )
    # KO Fib 61.8: crossover regime around the rolling Fib 61.8% retracement.
    # Entry on cross above Fib 61.8 with a light momentum filter; exit on cross below.
    above_618 = close > fib_618
    below_618 = close < fib_618
    cross_up_618 = above_618 & (close.shift(1) <= fib_618.shift(1))
    cross_down_618 = below_618 & (close.shift(1) >= fib_618.shift(1))
    ko_fib_entry = cross_up_618 & (trend_up | (mom_63.fillna(0.0) > 0.0)) & (mom_20.fillna(0.0) >= -0.03)
    ko_fib_exit = cross_down_618 & (trend_down | (mom_20.fillna(0.0) < -0.03))
    exit_long_eff = exit_long

    if profile.style == "breakout":
        buy_sig = breakout_buy | momentum_buy
        sell_sig = breakout_sell | momentum_sell
    elif profile.style == "pullback":
        buy_sig = pullback_buy | (fib_support & trend_up & (zone <= 0.45))
        sell_sig = pullback_sell | breakout_sell
    elif profile.style == "mean_reversion":
        buy_sig = mean_buy
        sell_sig = mean_sell | breakout_sell
    elif profile.style == "regime_momentum":
        buy_sig = regime_entry
        sell_sig = regime_exit
    elif profile.style == "ko_turbo":
        buy_sig = turbo_entry
        sell_sig = turbo_exit
        exit_long_eff = turbo_exit
    elif profile.style == "ko_fib618":
        buy_sig = ko_fib_entry
        sell_sig = ko_fib_exit
        exit_long_eff = exit_long | ko_fib_exit
    elif profile.style in {"ko_candles_book", "candles_book_pure", "candles_book_context_min"}:
        # Candlestick patterns based on the eBook content (Spanish names):
        # hammer / inverted hammer, hanging man / shooting star, engulfing, harami,
        # morning/evening star, 3 white soldiers / 3 black crows, marubozu, piercing/dark cloud cover.
        if open_.isna().all() or high_.isna().all() or low_.isna().all():
            buy_sig = empty_false
            sell_sig = empty_false
            if profile.style in {"candles_book_pure", "candles_book_context_min"}:
                exit_long_eff = empty_false
        else:
            # Candlestick context from recent price direction (independent from SMA channel trend).
            trend_up_cdl = (close > close.shift(3)).fillna(False)
            trend_down_cdl = (close < close.shift(3)).fillna(False)
            body = (close - open_).abs()
            rng = (high_ - low_).replace(0.0, np.nan)
            upper = high_ - np.maximum(open_, close)
            lower = np.minimum(open_, close) - low_
            body_pct = body / rng
            upper_pct = upper / rng
            lower_pct = lower / rng
            is_bull = close > open_
            is_bear = close < open_

            # Single-candle reversal patterns
            doji = body_pct <= 0.10
            dragonfly_doji = doji & (lower_pct >= 0.60) & (upper_pct <= 0.10)
            gravestone_doji = doji & (upper_pct >= 0.60) & (lower_pct <= 0.10)

            small_body = body_pct <= 0.25
            hammer_shape = small_body & (lower >= (2.0 * body)) & (upper <= (0.25 * body))
            inv_hammer_shape = small_body & (upper >= (2.0 * body)) & (lower <= (0.25 * body))
            # Context variants
            hammer = hammer_shape & trend_down_cdl
            hanging_man = hammer_shape & trend_up_cdl
            inverted_hammer = inv_hammer_shape & trend_down_cdl
            shooting_star = inv_hammer_shape & trend_up_cdl

            # Marubozu (continuation/strength)
            maru = (body_pct >= 0.90) & (upper_pct <= 0.06) & (lower_pct <= 0.06)
            marubozu_white = maru & is_bull
            marubozu_black = maru & is_bear

            # Two-candle patterns
            prev_open = open_.shift(1)
            prev_close = close.shift(1)
            prev_high = high_.shift(1)
            prev_low = low_.shift(1)
            prev_body = (prev_close - prev_open).abs()
            prev_rng = (prev_high - prev_low).replace(0.0, np.nan)
            prev_body_pct = prev_body / prev_rng
            prev_bull = prev_close > prev_open
            prev_bear = prev_close < prev_open

            # Pauta envolvente / Toro 180 (bullish engulfing with optional large bodies)
            bullish_engulf = prev_bear & is_bull & (open_ <= prev_close) & (close >= prev_open)
            bearish_engulf = prev_bull & is_bear & (open_ >= prev_close) & (close <= prev_open)
            toro_180 = bullish_engulf & (prev_body_pct >= 0.55) & (body_pct >= 0.55)
            oso_180 = bearish_engulf & (prev_body_pct >= 0.55) & (body_pct >= 0.55)

            # Harami
            curr_max = np.maximum(open_, close)
            curr_min = np.minimum(open_, close)
            prev_max = np.maximum(prev_open, prev_close)
            prev_min = np.minimum(prev_open, prev_close)
            inside_prev_body = (curr_max <= prev_max) & (curr_min >= prev_min)
            bullish_harami = prev_bear & is_bull & inside_prev_body & (prev_body_pct >= 0.40) & (body_pct <= 0.35)
            bearish_harami = prev_bull & is_bear & inside_prev_body & (prev_body_pct >= 0.40) & (body_pct <= 0.35)

            # Pauta penetrante (Piercing) / Nube oscura (Dark cloud cover)
            prev_mid = (prev_open + prev_close) / 2.0
            piercing = prev_bear & is_bull & (open_ <= prev_low) & (close >= prev_mid) & (close <= prev_open)
            dark_cloud = prev_bull & is_bear & (open_ >= prev_high) & (close <= prev_mid) & (close >= prev_open)

            # 3-candle patterns: Morning/Evening Star (gaps are ignored; uses body sizes + close location)
            o2 = open_.shift(2)
            c2 = close.shift(2)
            h2 = high_.shift(2)
            l2 = low_.shift(2)
            body2 = (c2 - o2).abs()
            rng2 = (h2 - l2).replace(0.0, np.nan)
            body2_pct = body2 / rng2
            is_bear2 = c2 < o2
            is_bull2 = c2 > o2
            mid2 = (o2 + c2) / 2.0
            body1 = (close.shift(1) - open_.shift(1)).abs()
            rng1 = (high_.shift(1) - low_.shift(1)).replace(0.0, np.nan)
            body1_pct = body1 / rng1
            small1 = body1_pct <= 0.25
            morning_star = is_bear2 & (body2_pct >= 0.55) & small1 & is_bull & (close >= mid2)
            evening_star = is_bull2 & (body2_pct >= 0.55) & small1 & is_bear & (close <= mid2)

            # 3 soldiers / 3 crows
            b0 = is_bull
            b1 = close.shift(1) > open_.shift(1)
            b2 = close.shift(2) > open_.shift(2)
            r0 = is_bear
            r1 = close.shift(1) < open_.shift(1)
            r2 = close.shift(2) < open_.shift(2)
            long0 = body_pct >= 0.45
            long1 = (body1_pct >= 0.45)
            long2 = (body2_pct >= 0.45)
            higher_closes = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
            lower_closes = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
            open_in_prev_body = (open_ >= np.minimum(open_.shift(1), close.shift(1))) & (open_ <= np.maximum(open_.shift(1), close.shift(1)))
            open1_in_body2 = (open_.shift(1) >= np.minimum(o2, c2)) & (open_.shift(1) <= np.maximum(o2, c2))
            three_soldiers = b0 & b1 & b2 & long0 & long1 & long2 & higher_closes & open_in_prev_body & open1_in_body2
            three_crows = r0 & r1 & r2 & long0 & long1 & long2 & lower_closes & open_in_prev_body & open1_in_body2

            bullish_core = (
                hammer
                | inverted_hammer
                | bullish_engulf
                | toro_180
                | bullish_harami
                | piercing
                | morning_star
                | three_soldiers
                | marubozu_white
            )
            bearish_core = (
                hanging_man
                | shooting_star
                | bearish_engulf
                | oso_180
                | bearish_harami
                | dark_cloud
                | evening_star
                | three_crows
                | marubozu_black
            )

            # Doji variants are treated as "alert" patterns; require confirmation candle the next session.
            dragonfly_confirm = dragonfly_doji.shift(1).astype("boolean").fillna(False).astype(bool) & is_bull
            gravestone_confirm = gravestone_doji.shift(1).astype("boolean").fillna(False).astype(bool) & is_bear

            bullish = bullish_core | dragonfly_confirm
            bearish = bearish_core | gravestone_confirm

            if profile.style == "ko_candles_book":
                # Hybrid mode: candlesticks + channel/trend context.
                buy_sig = bullish & (trend_down | ind["near_floor"].fillna(False) | (zone <= 0.35))
                sell_sig = bearish & (trend_up | ind["near_ceiling"].fillna(False) | (zone >= 0.65))
            elif profile.style == "candles_book_context_min":
                # Minimal context: lightweight regime+location filter and a simple protective stop.
                # Keeps the strategy mostly candle-driven while reducing high-frequency noise.
                buy_sig = bullish & (trend_down_cdl | (zone <= 0.50))
                sell_sig = bearish & (trend_up_cdl | (zone >= 0.50))
                exit_long_eff = (
                    (close < (channel_low * (1.0 - float(profile.breakout_down_buffer))))
                    | (mom_20.fillna(0.0) < -0.08)
                ).fillna(False)
            else:
                # Pure mode: use only candlestick patterns from the PDF.
                buy_sig = bullish
                sell_sig = bearish
                exit_long_eff = empty_false
    elif profile.style == "channel_reversal":
        # Pure channel logic (no Fibonacci):
        # - Buy near channel low
        # - Sell near channel high
        # - Extra protection: exit if price breaks below channel low (buffered)
        exit_long_channel = (
            (close < (channel_low * (1.0 - float(profile.breakout_down_buffer))))
            | (breakout_down & trend_down)
        ).fillna(False)
        buy_sig = near_floor
        sell_sig = near_ceiling | exit_long_channel
        exit_long_eff = exit_long_channel
    elif profile.style == "channel_pivot_reversal":
        # Strict: require pivots at the channel boundaries.
        exit_long_channel = (
            (close < (channel_low * (1.0 - float(profile.breakout_down_buffer))))
            | (breakout_down & trend_down)
        ).fillna(False)
        buy_sig = pivot_low & near_floor
        sell_sig = (pivot_high & near_ceiling) | exit_long_channel
        exit_long_eff = exit_long_channel
    elif profile.style == "pivot_reversal":
        # Buy on confirmed local pivot low, sell on confirmed local pivot high.
        # Optional gating using zone to avoid buying pivots that happen too high in the channel.
        exit_long_channel = (
            (close < (channel_low * (1.0 - float(profile.breakout_down_buffer))))
            | (breakout_down & trend_down)
        ).fillna(False)
        buy_sig = pivot_low & near_floor
        sell_sig = (pivot_high & near_ceiling) | exit_long_channel
        exit_long_eff = exit_long_channel
    else:
        buy_sig = pullback_buy | breakout_buy | momentum_buy
        sell_sig = pullback_sell | breakout_sell | momentum_sell

    buy_sig = buy_sig.fillna(False)
    sell_sig = (sell_sig | exit_long_eff).fillna(False)
    signal = pd.Series(
        np.where(
            buy_sig & ~sell_sig,
            "BUY",
            np.where(sell_sig & ~buy_sig, "SELL", "HOLD"),
        ),
        index=idx,
    )
    return buy_sig, sell_sig, exit_long_eff.fillna(False), signal


def _long_only_state_from_signals(
    buy_sig: pd.Series,
    sell_sig: pd.Series,
    exit_long: pd.Series,
) -> pd.Series:
    n = len(buy_sig)
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        prev = state[i - 1]
        b = bool(buy_sig.iloc[i])
        s = bool(sell_sig.iloc[i])
        xl = bool(exit_long.iloc[i])
        new_state = prev
        if prev == 0 and b:
            new_state = 1
        elif prev == 1 and (s or xl):
            new_state = 0
        state[i] = new_state
    return pd.Series(state, index=buy_sig.index, dtype=float)


def _candidate_profile_keys_for_ticker(ticker: str, ohlc: pd.DataFrame) -> list[str]:
    t = ticker.strip().upper()
    if t == "KO":
        return ["ko_channel_pivots", "ko_candles_book", "ko_channel_reversal", "ko_turbo", "regime_momentum", "hybrid_channel", "pullback_trend"]
    if t == "MELI":
        return ["meli_turbo", "mean_reversion", "regime_momentum", "hybrid_channel"]
    close = pd.to_numeric(ohlc.get("Close"), errors="coerce")
    ret = close.pct_change().dropna()
    if ret.empty:
        return [DEFAULT_PROFILE_KEY]
    ann_vol = float(ret.tail(252).std(ddof=0) * np.sqrt(252.0))
    if ticker.endswith(".MX"):
        return ["mx_defensive", "pullback_trend", "hybrid_channel", "regime_momentum"]
    if ann_vol < 0.22:
        return ["pullback_trend", "hybrid_channel", "regime_momentum", "mean_reversion"]
    if ann_vol < 0.42:
        return ["hybrid_channel", "regime_momentum", "pullback_trend", "breakout_swing"]
    return ["regime_momentum", "breakout_fast", "breakout_swing", "hybrid_channel"]


def _score_training_timeseries(ts: pd.DataFrame) -> tuple[float, float, float, int]:
    if ts.empty or len(ts) < 40:
        return float("-inf"), float("nan"), float("nan"), 0
    eq = pd.to_numeric(ts["strategy_eq"], errors="coerce").dropna()
    if eq.empty or float(eq.iloc[0]) <= 0:
        return float("-inf"), float("nan"), float("nan"), 0
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    max_dd = float((eq / eq.cummax() - 1.0).min())
    trades = int(pd.to_numeric(ts["state"], errors="coerce").fillna(0.0).diff().abs().fillna(0.0).sum())
    dd_penalty = abs(min(0.0, max_dd)) * 0.35
    low_trade_penalty = 0.05 if trades < 2 else 0.0
    high_trade_penalty = max(0, trades - 30) * 0.004
    score = ret - dd_penalty - low_trade_penalty - high_trade_penalty
    return score, ret, max_dd, trades


def _select_profile_for_ticker(
    ticker: str,
    ohlc: pd.DataFrame,
    backtest_days: int,
    target_return: float,
    forced_profile_key: str | None = None,
) -> tuple[HoldingStrategyProfile, HoldingStrategyPlan]:
    default_profile = STRATEGY_PROFILE_BY_KEY[DEFAULT_PROFILE_KEY]
    close = pd.to_numeric(ohlc.get("Close"), errors="coerce")
    ret = close.pct_change().dropna()
    ann_vol = float(ret.tail(252).std(ddof=0) * np.sqrt(252.0)) if not ret.empty else None
    forced_profile = STRATEGY_PROFILE_BY_KEY.get(str(forced_profile_key).strip().lower()) if forced_profile_key else None
    if ohlc.empty or len(ohlc) < 260:
        p = forced_profile or default_profile
        plan = HoldingStrategyPlan(
            ticker=ticker,
            strategy_key=p.key,
            strategy_label=p.label,
            strategy_style=p.style,
            annualized_volatility=ann_vol,
            train_return=None,
            train_max_drawdown=None,
            train_trades=0,
            target_return=target_return,
            target_met=False,
            forced=bool(forced_profile is not None),
        )
        return p, plan

    split_n = len(ohlc) - max(120, backtest_days)
    if split_n < 220:
        split_n = len(ohlc) - 120
    train_ohlc = ohlc.iloc[:split_n] if split_n > 220 else ohlc.copy()
    train_window = min(504, max(220, len(train_ohlc) - 5))

    if forced_profile is not None:
        train_ind = _channel_indicators(train_ohlc, profile=forced_profile)
        ts_train = _build_strategy_timeseries(
            ind=train_ind,
            backtest_days=train_window,
            initial_capital=100.0,
            profile=forced_profile,
        )
        _, train_ret, train_dd, train_trades = _score_training_timeseries(ts_train)
        train_ret_v = train_ret if np.isfinite(train_ret) else None
        train_dd_v = train_dd if np.isfinite(train_dd) else None
        plan = HoldingStrategyPlan(
            ticker=ticker,
            strategy_key=forced_profile.key,
            strategy_label=forced_profile.label,
            strategy_style=forced_profile.style,
            annualized_volatility=ann_vol if ann_vol is None or np.isfinite(ann_vol) else None,
            train_return=train_ret_v,
            train_max_drawdown=train_dd_v,
            train_trades=int(train_trades),
            target_return=target_return,
            target_met=bool(train_ret_v is not None and train_ret_v >= target_return),
            forced=True,
        )
        return forced_profile, plan

    candidate_keys = _candidate_profile_keys_for_ticker(ticker=ticker, ohlc=ohlc)

    best_profile = default_profile
    best_score = float("-inf")
    best_train_return = None
    best_train_dd = None
    best_train_trades = 0

    for key in candidate_keys:
        p = STRATEGY_PROFILE_BY_KEY.get(key)
        if p is None:
            continue
        train_ind = _channel_indicators(train_ohlc, profile=p)
        ts_train = _build_strategy_timeseries(
            ind=train_ind,
            backtest_days=train_window,
            initial_capital=100.0,
            profile=p,
        )
        score, train_ret, train_dd, train_trades = _score_training_timeseries(ts_train)
        if score > best_score:
            best_score = score
            best_profile = p
            best_train_return = train_ret if np.isfinite(train_ret) else None
            best_train_dd = train_dd if np.isfinite(train_dd) else None
            best_train_trades = int(train_trades)

    target_met = bool(best_train_return is not None and best_train_return >= target_return)
    plan = HoldingStrategyPlan(
        ticker=ticker,
        strategy_key=best_profile.key,
        strategy_label=best_profile.label,
        strategy_style=best_profile.style,
        annualized_volatility=ann_vol if ann_vol is None or np.isfinite(ann_vol) else None,
        train_return=best_train_return,
        train_max_drawdown=best_train_dd,
        train_trades=best_train_trades,
        target_return=target_return,
        target_met=target_met,
        forced=False,
    )
    return best_profile, plan


def _holding_action_from_channels(
    ticker: str,
    ind: pd.DataFrame,
    stop_loss: float | None,
    profile: HoldingStrategyProfile | None = None,
) -> HoldingAction:
    p = profile or STRATEGY_PROFILE_BY_KEY[DEFAULT_PROFILE_KEY]
    if ind.empty:
        return HoldingAction(
            ticker=ticker,
            action="HOLD",
            close=float("nan"),
            stop_loss=stop_loss,
            reason="sin datos OHLC",
            strategy=p.label,
        )

    last = ind.iloc[-1]
    close = float(last["Close"])
    stop_v = float(stop_loss) if stop_loss is not None else None

    if not np.isfinite(close):
        return HoldingAction(
            ticker=ticker,
            action="HOLD",
            close=float("nan"),
            stop_loss=stop_v,
            reason="close invalido",
            strategy=p.label,
        )

    if stop_v is not None and stop_v > 0 and close < stop_v:
        return HoldingAction(
            ticker=ticker,
            action="SELL",
            close=close,
            stop_loss=stop_v,
            reason=f"precio < stop_loss ({close:.2f} < {stop_v:.2f})",
            strategy=p.label,
        )

    needed = ["channel_high", "channel_low", "zone", "sma50", "sma200"]
    if any(not np.isfinite(float(last[c])) for c in needed):
        return HoldingAction(
            ticker=ticker,
            action="HOLD",
            close=close,
            stop_loss=stop_v,
            reason="historial insuficiente para canales (>=200d)",
            strategy=p.label,
        )

    zone = float(last["zone"])
    buy_sig, sell_sig, _, signal = _signals_from_profile(ind=ind, profile=p)
    last_signal = str(signal.iloc[-1]) if len(signal) else "HOLD"
    if last_signal == "BUY" and bool(buy_sig.iloc[-1]):
        return HoldingAction(
            ticker=ticker,
            action="BUY",
            close=close,
            stop_loss=stop_v,
            reason=f"senal {p.label} ({p.style}) activa (zone={zone:.2f})",
            strategy=p.label,
        )
    if last_signal == "SELL" and bool(sell_sig.iloc[-1]):
        return HoldingAction(
            ticker=ticker,
            action="SELL",
            close=close,
            stop_loss=stop_v,
            reason=f"salida defensiva {p.label} ({p.style}) (zone={zone:.2f})",
            strategy=p.label,
        )
    return HoldingAction(
        ticker=ticker,
        action="HOLD",
        close=close,
        stop_loss=stop_v,
        reason=f"sin gatillo {p.label} ({p.style}) (zone={zone:.2f})",
        strategy=p.label,
    )


def _backtest_channel_1y(
    ticker: str,
    ind: pd.DataFrame,
    backtest_days: int = 252,
    fee_bps: float = 0.0,
    initial_capital: float = 100.0,
    profile: HoldingStrategyProfile | None = None,
    target_return: float = 0.20,
) -> HoldingBacktest | None:
    p = profile or STRATEGY_PROFILE_BY_KEY[DEFAULT_PROFILE_KEY]
    lev = float(p.leverage) if np.isfinite(p.leverage) and p.leverage > 0 else 1.0
    ts = _build_strategy_timeseries(
        ind=ind,
        backtest_days=backtest_days,
        fee_bps=fee_bps,
        initial_capital=initial_capital,
        profile=p,
    )
    if ts.empty:
        return None

    eq = pd.to_numeric(ts["strategy_eq"], errors="coerce").dropna()
    bh = pd.to_numeric(ts["buyhold_eq"], errors="coerce").dropna()
    if eq.empty or bh.empty:
        return None

    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    max_dd = float((eq / eq.cummax() - 1.0).min())
    trades_1y = int(pd.to_numeric(ts["state"], errors="coerce").fillna(0.0).diff().abs().fillna(0.0).sum())
    strat_final = float(eq.iloc[-1])
    bh_final = float(bh.iloc[-1])
    strat_ret = float((strat_final / cap0) - 1.0)
    return HoldingBacktest(
        ticker=ticker,
        initial_capital_1y=cap0,
        strategy_final_capital_1y=strat_final,
        buy_hold_final_capital_1y=bh_final,
        strategy_return_1y=strat_ret,
        buy_hold_return_1y=float((bh_final / cap0) - 1.0),
        max_drawdown_1y=max_dd,
        trades_1y=trades_1y,
        strategy=p.label,
        strategy_leverage=lev,
        target_return_1y=float(target_return),
        target_met_1y=bool(strat_ret >= float(target_return)),
    )


def _build_strategy_timeseries(
    ind: pd.DataFrame,
    backtest_days: int = 252,
    fee_bps: float = 0.0,
    initial_capital: float = 100.0,
    profile: HoldingStrategyProfile | None = None,
) -> pd.DataFrame:
    p = profile or STRATEGY_PROFILE_BY_KEY[DEFAULT_PROFILE_KEY]
    if ind.empty or len(ind) < 220:
        return pd.DataFrame()

    d = ind.copy().dropna(subset=["Close"])
    if len(d) < 220:
        return pd.DataFrame()

    buy_sig, sell_sig, exit_long, signal = _signals_from_profile(ind=d, profile=p)
    state_s = _long_only_state_from_signals(
        buy_sig=buy_sig,
        sell_sig=sell_sig,
        exit_long=exit_long,
    )
    lev = float(p.leverage) if np.isfinite(p.leverage) and p.leverage > 0 else 1.0
    ret = d["Close"].pct_change().fillna(0.0)
    strat_ret = state_s.shift(1).fillna(0.0) * ret * lev
    if fee_bps > 0:
        turn = state_s.diff().abs().fillna(0.0)
        strat_ret = strat_ret - (turn * (fee_bps / 10_000.0))

    n = min(backtest_days, len(d))
    if n <= 2:
        return pd.DataFrame()

    signal_code = signal.map({"SELL": -1, "HOLD": 0, "BUY": 1}).astype(float)

    out = pd.DataFrame(
        {
            "Close": d["Close"].iloc[-n:],
            "strategy_ret": strat_ret.iloc[-n:],
            "buyhold_ret": ret.iloc[-n:],
            "state": state_s.iloc[-n:],
            "channel_high": d["channel_high"].iloc[-n:],
            "channel_low": d["channel_low"].iloc[-n:],
            "pivot_low": d["pivot_low"].iloc[-n:] if "pivot_low" in d.columns else False,
            "pivot_high": d["pivot_high"].iloc[-n:] if "pivot_high" in d.columns else False,
            "fib_236": d["fib_236"].iloc[-n:] if "fib_236" in d.columns else np.nan,
            "fib_382": d["fib_382"].iloc[-n:] if "fib_382" in d.columns else np.nan,
            "fib_618": d["fib_618"].iloc[-n:] if "fib_618" in d.columns else np.nan,
            "fib_786": d["fib_786"].iloc[-n:] if "fib_786" in d.columns else np.nan,
            "signal": signal.iloc[-n:],
            "signal_code": signal_code.iloc[-n:],
            "strategy": p.label,
            "strategy_style": p.style,
            "strategy_leverage": lev,
        }
    )
    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    out["strategy_eq"] = cap0 * (1.0 + out["strategy_ret"]).cumprod()
    out["buyhold_eq"] = cap0 * (1.0 + out["buyhold_ret"]).cumprod()
    return out


def _save_holding_plot(
    ticker: str,
    ts: pd.DataFrame,
    out_path: Path,
    strategy_label: str | None = None,
) -> None:
    if ts.empty:
        return

    import matplotlib

    if "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    ax1 = axes[0]
    ax1.plot(ts.index, ts["Close"], label="Close", linewidth=1.6, color="#1f77b4")
    if "channel_high" in ts and "channel_low" in ts:
        ax1.plot(ts.index, ts["channel_high"], label="Canal Alto", linewidth=1.0, linestyle="--", color="#d62728", alpha=0.8)
        ax1.plot(ts.index, ts["channel_low"], label="Canal Bajo", linewidth=1.0, linestyle="--", color="#2ca02c", alpha=0.8)
    # Some profiles explicitly avoid Fibonacci in both logic and visualization.
    style = str(ts.get("strategy_style", pd.Series([""])).iloc[-1]) if "strategy_style" in ts else ""
    show_fib = style not in {
        "channel_reversal",
        "channel_pivot_reversal",
        "pivot_reversal",
        "candles_book_pure",
        "candles_book_context_min",
    }
    if show_fib:
        if "fib_382" in ts and pd.to_numeric(ts["fib_382"], errors="coerce").notna().any():
            ax1.plot(ts.index, ts["fib_382"], label="Fib 38.2%", linewidth=0.9, linestyle="-.", color="#8c564b", alpha=0.75)
        if "fib_618" in ts and pd.to_numeric(ts["fib_618"], errors="coerce").notna().any():
            ax1.plot(ts.index, ts["fib_618"], label="Fib 61.8%", linewidth=0.9, linestyle="-.", color="#17becf", alpha=0.75)
    if "state" in ts:
        # Plot actual entries/exits (derived from state transitions) instead of raw BUY/SELL signals.
        state_i = pd.to_numeric(ts["state"], errors="coerce").fillna(0.0).astype(int)
        prev_state_i = state_i.shift(1).fillna(0).astype(int)
        buy_mask = (prev_state_i == 0) & (state_i == 1)
        sell_mask = (prev_state_i == 1) & (state_i == 0)
        hold_mask = ~(buy_mask | sell_mask)

        ax1.scatter(ts.index[buy_mask], ts.loc[buy_mask, "Close"], marker="^", s=46, color="#2ca02c", label="BUY", zorder=4)
        ax1.scatter(ts.index[sell_mask], ts.loc[sell_mask, "Close"], marker="v", s=46, color="#d62728", label="SELL", zorder=4)
        ax1.scatter(ts.index[hold_mask], ts.loc[hold_mask, "Close"], marker="o", s=9, color="#7f7f7f", alpha=0.18, label="HOLD", zorder=2)

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        hold_count = int(hold_mask.sum())
        counts_label = f"BUY={buy_count} | SELL={sell_count} | HOLD={hold_count}"
        strategy_text = f" | {strategy_label}" if strategy_label else ""
        ax1.set_title(f"{ticker}{strategy_text} - Precio, Canal y Entradas/Salidas ({len(ts)} sesiones) [{counts_label}]")
        ax1.text(
            0.015,
            0.98,
            f"Entradas/Salidas: {counts_label}",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#aaaaaa"},
        )
    else:
        ax1.set_title(f"{ticker} - Precio, Canal y Entradas/Salidas ({len(ts)} sesiones)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(ts.index, ts["strategy_eq"], label="Strategy", linewidth=1.8, color="#ff7f0e")
    ax2.plot(ts.index, ts["buyhold_eq"], label="Buy&Hold", linewidth=1.6, color="#9467bd")
    initial_capital = float(ts["strategy_eq"].iloc[0])
    final_capital = float(ts["strategy_eq"].iloc[-1])
    strategy_return = (final_capital / initial_capital - 1.0) if initial_capital > 0 else float("nan")
    ax2.set_title(
        f"Equity Curve (capital inicial={_format_money(initial_capital)} | "
        f"capital final={_format_money(final_capital)} | rendimiento={strategy_return:+.1%})"
    )
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_holdings_actions(
    holdings: list[str],
    asof: date,
    stop_losses: dict[str, float] | None = None,
    lookback_days: int = 365 * 5,
    backtest_days: int = 252,
    plot_dir: Path | None = None,
    initial_capital: float = 100.0,
    target_return: float = 0.20,
    strategy_overrides: dict[str, str] | None = None,
) -> tuple[list[HoldingAction], list[HoldingBacktest], pd.DataFrame, list[HoldingStrategyPlan]]:
    if not holdings:
        return [], [], pd.DataFrame(), []

    stop_losses = stop_losses or {}
    strategy_overrides = strategy_overrides or {}
    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    start = asof - timedelta(days=lookback_days)

    data = _download_history(holdings, start=start, end_inclusive=asof)
    out: list[HoldingAction] = []
    bt: list[HoldingBacktest] = []
    trade_logs: list[pd.DataFrame] = []
    plans: list[HoldingStrategyPlan] = []

    for t in holdings:
        hist = data.get(t)
        sl = stop_losses.get(t)
        ohlc = _prepare_ohlc(hist if isinstance(hist, pd.DataFrame) else pd.DataFrame())
        profile, plan = _select_profile_for_ticker(
            ticker=t,
            ohlc=ohlc,
            backtest_days=backtest_days,
            target_return=target_return,
            forced_profile_key=strategy_overrides.get(t.upper()),
        )
        plans.append(plan)
        ind = _channel_indicators(ohlc, profile=profile) if not ohlc.empty else pd.DataFrame()
        out.append(_holding_action_from_channels(ticker=t, ind=ind, stop_loss=sl, profile=profile))
        bt_row = _backtest_channel_1y(
            ticker=t,
            ind=ind,
            backtest_days=backtest_days,
            initial_capital=cap0,
            profile=profile,
            target_return=target_return,
        )
        if bt_row is not None:
            bt.append(bt_row)

        ts = _build_strategy_timeseries(
            ind=ind,
            backtest_days=backtest_days,
            initial_capital=cap0,
            profile=profile,
        )
        if not ts.empty:
            trade_logs.append(_build_daily_trade_log(ticker=t, asof=asof, ts=ts))
            if plot_dir is not None:
                safe_name = re.sub(r"[^A-Z0-9._-]", "_", t.upper())
                _save_holding_plot(
                    ticker=t,
                    ts=ts,
                    out_path=plot_dir / f"{safe_name}_strategy.png",
                    strategy_label=profile.label,
                )

    if trade_logs:
        trade_log_df = pd.concat(trade_logs, axis=0, ignore_index=True)
        trade_log_df = trade_log_df.sort_values(["ticker", "date"], ascending=[True, True], kind="stable")
    else:
        trade_log_df = pd.DataFrame(
            columns=[
                "asof",
                "date",
                "ticker",
                "strategy",
                "close",
                "signal",
                "position",
                "trade_action",
                "strategy_ret",
                "strategy_equity_usd",
                "buyhold_equity_usd",
            ]
        )

    return out, bt, trade_log_df, plans


def _action_from_transition(prev_state: int, curr_state: int) -> str:
    if prev_state == curr_state:
        return "HOLD"
    if prev_state == 0 and curr_state == 1:
        return "BUY"
    if prev_state == 1 and curr_state == 0:
        return "SELL_TO_CLOSE"
    if prev_state == 0 and curr_state == -1:
        return "SHORT"
    if prev_state == -1 and curr_state == 0:
        return "COVER"
    if prev_state == 1 and curr_state == -1:
        return "FLIP_TO_SHORT"
    if prev_state == -1 and curr_state == 1:
        return "FLIP_TO_LONG"
    return "HOLD"


def _build_daily_trade_log(ticker: str, asof: date, ts: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame()

    out = ts.copy()
    state_i = out["state"].fillna(0.0).astype(int)
    prev_state_i = state_i.shift(1).fillna(0).astype(int)

    trade_action = [
        _action_from_transition(int(prev_state_i.iloc[i]), int(state_i.iloc[i]))
        for i in range(len(out))
    ]
    position = state_i.map({1: "LONG", 0: "FLAT", -1: "SHORT"}).fillna("FLAT")

    log = pd.DataFrame(
        {
            "asof": asof.isoformat(),
            "date": pd.to_datetime(out.index).date,
            "ticker": ticker,
            "strategy": out["strategy"].astype(str) if "strategy" in out.columns else "",
            "close": pd.to_numeric(out["Close"], errors="coerce"),
            "signal": out["signal"].astype(str),
            "position": position.astype(str),
            "trade_action": trade_action,
            "strategy_ret": pd.to_numeric(out["strategy_ret"], errors="coerce"),
            "strategy_equity_usd": pd.to_numeric(out["strategy_eq"], errors="coerce"),
            "buyhold_equity_usd": pd.to_numeric(out["buyhold_eq"], errors="coerce"),
        }
    )
    return log


def _write_orders(path: Path, orders: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not orders:
        return
    df = pd.DataFrame(orders)
    df.to_csv(path, index=False)


def _write_trade_log(path: Path, trade_log: pd.DataFrame) -> None:
    if trade_log is None or trade_log.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(path, index=False)


def _prompt_yes_no(msg: str) -> bool:
    while True:
        ans = input(msg).strip().lower()
        if ans in ("s", "si", "sí", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Respuesta inválida. Usa 's' o 'n'.")


def _prompt_float(msg: str) -> float:
    while True:
        raw = input(msg).strip()
        v = _safe_float(raw)
        if v is not None and v > 0:
            return float(v)
        print("Número inválido. Intenta otra vez.")

def _sanitize_for_json(x: object) -> object:
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, dict):
        return {k: _sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]
    return x


def _safe_run_tag(tag: str | None) -> str:
    if tag is None:
        return ""
    t = str(tag).strip()
    if not t:
        return ""
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", t)
    t = t.strip("._-")
    return t[:64]


def main() -> int:
    _best_effort_utf8_stdout()
    parser = argparse.ArgumentParser(description="Recomendador simple (Top N + seguimiento holdings).")
    parser.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE_PATH), help="Archivo con tickers (uno por línea).")
    parser.add_argument("--holdings", type=str, default=str(DEFAULT_HOLDINGS_PATH), help="Archivo con tickers en cartera (uno por línea).")
    parser.add_argument("--holdings-strategies", type=str, default=str(DEFAULT_HOLDINGS_STRATEGIES_PATH), help="JSON de estrategia fija por ticker (ej. AVAV->breakout_fast).")
    parser.add_argument("--daily-updates", type=str, default=str(DEFAULT_DAILY_UPDATES_CSV), help="Ruta al Daily Updates.csv (para construir universo/holdings).")
    parser.add_argument("--universe-from-daily", action="store_true", help="Usar Daily Updates.csv como universo (en vez de config/universe.txt).")
    parser.add_argument("--holdings-from-daily", action="store_true", help="Derivar holdings desde el último día en Daily Updates.csv (incluye stop-loss si existe).")
    parser.add_argument("--asof", type=str, default=None, help="Fecha YYYY-MM-DD (default: hoy; si es fin de semana usa viernes).")
    parser.add_argument("--top", type=int, default=5, help="Cantidad de recomendaciones Top N (default: 5).")
    parser.add_argument("--min-dollar-vol", type=float, default=500_000.0, help="Filtro de liquidez: promedio 20d Close*Volume.")
    parser.add_argument("--holdings-lookback-years", type=int, default=5, help="Lookback para holdings/channels (default: 5 anios).")
    parser.add_argument("--backtest-days", type=int, default=252, help="Ventana de backtest para holdings (default: 252).")
    parser.add_argument("--initial-capital", type=float, default=100.0, help="Capital inicial para backtest/equity de holdings (default: 100 USD).")
    parser.add_argument("--target-return", type=float, default=0.20, help="Objetivo de rendimiento 1Y para holdings (default: 0.20 = +20%%).")
    parser.add_argument("--plot-holdings", dest="plot_holdings", action="store_true", help="Genera graficos por holding (default: activado).")
    parser.add_argument("--no-plot-holdings", dest="plot_holdings", action="store_false", help="Desactiva generacion de graficos por holding.")
    parser.set_defaults(plot_holdings=True)
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directorio de salida para graficos (default: <out-dir>/YYYY-MM-DD[/<run-tag>]/plots).",
    )
    parser.add_argument("--non-interactive", action="store_true", help="No preguntar confirmaciones; solo imprime y guarda outputs.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directorio base para outputs (default: runs/).")
    parser.add_argument("--run-tag", type=str, default="", help="Subcarpeta opcional bajo runs/YYYY-MM-DD/ (evita crear runs_*).")
    args = parser.parse_args()

    asof = _coerce_asof(args.asof)
    universe_path = Path(args.universe)
    holdings_path = Path(args.holdings)
    holdings_strategies_path = Path(args.holdings_strategies)
    daily_updates_path = Path(args.daily_updates)
    out_root = Path(args.out_dir)
    run_tag = _safe_run_tag(args.run_tag)
    out_dir = out_root / asof.isoformat()
    if run_tag:
        out_dir = out_dir / run_tag
    _init_yfinance_cache(out_root / "_yfinance_cache")

    if args.universe_from_daily:
        if not daily_updates_path.exists():
            raise FileNotFoundError(f"No existe Daily Updates.csv en: {daily_updates_path}")
        universe = universe_from_daily_updates(daily_updates_path)
    else:
        universe = _read_ticker_file(universe_path)

    stop_losses: dict[str, float] = {}
    if args.holdings_from_daily:
        if not daily_updates_path.exists():
            raise FileNotFoundError(f"No existe Daily Updates.csv en: {daily_updates_path}")
        holdings, stop_losses = holdings_from_daily_updates(daily_updates_path)
    else:
        holdings = _read_ticker_file(holdings_path)
    strategy_overrides = _read_holdings_strategy_overrides(holdings_strategies_path)

    print("\n" + "=" * 72)
    print(f"Recomendaciones (asof={asof.isoformat()})")
    print("=" * 72)

    top: list[ScoredTicker] = []
    if int(args.top) > 0:
        top = build_top_recommendations(
            universe=universe,
            asof=asof,
            min_dollar_vol_20d=float(args.min_dollar_vol),
            top_n=int(args.top),
        )

    if not top:
        print("\n[Top]")
        print("Sin resultados (revisa universo, conectividad o filtros de liquidez).")
    else:
        print("\n[Top]")
        for i, r in enumerate(top, 1):
            print(f"{i:>2}. {r.ticker:>8}  close={r.close:>8.2f}  mom_126d={r.mom_126d:+6.1%}  dv20={_format_money(r.dollar_vol_20d):>12}  {r.reason}")

    plot_dir: Path | None = None
    if args.plot_holdings:
        plot_dir = Path(args.plots_dir) if args.plots_dir else (out_dir / "plots")

    initial_capital = float(args.initial_capital) if np.isfinite(args.initial_capital) and args.initial_capital > 0 else 100.0
    target_return = float(args.target_return) if np.isfinite(args.target_return) else 0.20
    actions, backtests, holdings_trade_log, strategy_plans = build_holdings_actions(
        holdings=holdings,
        asof=asof,
        stop_losses=stop_losses,
        lookback_days=max(365, int(args.holdings_lookback_years) * 365),
        backtest_days=max(60, int(args.backtest_days)),
        plot_dir=plot_dir,
        initial_capital=initial_capital,
        target_return=target_return,
        strategy_overrides=strategy_overrides,
    )
    buy_count = int(sum(1 for a in actions if a.action == "BUY"))
    sell_count = int(sum(1 for a in actions if a.action == "SELL"))
    hold_count = int(sum(1 for a in actions if a.action == "HOLD"))
    if not actions:
        print("\n[Holdings]")
        print("Sin holdings configurados.")
    else:
        print("\n[Holdings]")
        for a in actions:
            sl = f"{a.stop_loss:.2f}" if a.stop_loss is not None else "-"
            print(f"{a.ticker:>10}  {a.action:<4}  strat={a.strategy:<18}  close={a.close:>9.2f}  stop={sl:>8}  {a.reason}")
        print(f"Resumen señales -> BUY={buy_count}  SELL={sell_count}  HOLD={hold_count}")

    if strategy_plans:
        print(f"\n[Perfiles Holdings | objetivo={target_return:+.1%}]")
        for plan in strategy_plans:
            vol_txt = f"{plan.annualized_volatility:.1%}" if plan.annualized_volatility is not None else "-"
            train_txt = f"{plan.train_return:+.1%}" if plan.train_return is not None else "-"
            dd_txt = f"{plan.train_max_drawdown:7.1%}" if plan.train_max_drawdown is not None else "-"
            status_txt = "OK" if plan.target_met else "NO"
            mode_txt = "PIN" if plan.forced else "AUTO"
            print(
                f"{plan.ticker:>10}  {plan.strategy_label:<18}  mode={mode_txt:<4}  style={plan.strategy_style:<14}  "
                f"vol={vol_txt:>7}  train={train_txt:>7}  maxDD={dd_txt:>7}  trades={plan.train_trades:>3}  target={status_txt}"
            )

    if backtests:
        print(f"\n[Holdings Backtest 1Y | capital inicial={_format_money(initial_capital)} | objetivo={target_return:+.1%}]")
        for b in backtests:
            hit = "OK" if b.target_met_1y else "NO"
            print(
                f"{b.ticker:>10}  strat={b.strategy_return_1y:+7.1%} ({_format_money(b.strategy_final_capital_1y)})  "
                f"buy&hold={b.buy_hold_return_1y:+7.1%} ({_format_money(b.buy_hold_final_capital_1y)})  "
                f"maxDD={b.max_drawdown_1y:7.1%}  trades={b.trades_1y:>3}  lev={b.strategy_leverage:.1f}x  target={hit}"
            )
        agg_initial = float(sum(b.initial_capital_1y for b in backtests))
        agg_strat = float(sum(b.strategy_final_capital_1y for b in backtests))
        agg_bh = float(sum(b.buy_hold_final_capital_1y for b in backtests))
        agg_strat_ret = (agg_strat / agg_initial - 1.0) if agg_initial > 0 else float("nan")
        agg_bh_ret = (agg_bh / agg_initial - 1.0) if agg_initial > 0 else float("nan")
        agg_target_hit = "OK" if np.isfinite(agg_strat_ret) and agg_strat_ret >= target_return else "NO"
        print(
            f"\n[Holdings Portfolio 1Y]  strat={agg_strat_ret:+7.1%} ({_format_money(agg_strat)})  "
            f"buy&hold={agg_bh_ret:+7.1%} ({_format_money(agg_bh)})  target={agg_target_hit}"
        )

    # ---- Interactive order generation (no execution) ----
    interactive_ok = _is_tty() and (not args.non_interactive)
    orders: list[dict] = []
    if interactive_ok and top:
        print("\n" + "-" * 72)
        print("Ordenes (simuladas): el sistema pregunta si quieres comprar.")
        print("No ejecuta operaciones; solo genera un CSV de ordenes sugeridas.")
        print("-" * 72)

        for r in top:
            if not _prompt_yes_no(f"Quieres generar una orden de COMPRA para {r.ticker}? (s/n): "):
                continue
            notional = _prompt_float("Monto USD a invertir (ej: 250): ")
            est_shares = notional / r.close if r.close > 0 else float("nan")
            orders.append(
                {
                    "asof": asof.isoformat(),
                    "ticker": r.ticker,
                    "side": "BUY",
                    "notional_usd": round(notional, 2),
                    "est_shares": round(est_shares, 6) if np.isfinite(est_shares) else "",
                    "price_ref": round(r.close, 4),
                    "source": "top",
                    "reason": r.reason,
                }
            )

    # Also allow BUY actions for holdings (add-to-winners), avoiding duplicates already in Top
    if interactive_ok and actions:
        top_set = {r.ticker for r in top}
        buy_holdings = [a for a in actions if a.action == "BUY" and a.ticker not in top_set]
        if buy_holdings:
            print("\n" + "-" * 72)
            print("Holdings con senal BUY (add): confirmacion de ordenes sugeridas.")
            print("-" * 72)
        for a in buy_holdings:
            if not _prompt_yes_no(f"Quieres generar una orden de COMPRA (add) para {a.ticker}? (s/n): "):
                continue
            notional = _prompt_float("Monto USD a invertir (ej: 250): ")
            est_shares = notional / a.close if a.close > 0 else float("nan")
            orders.append(
                {
                    "asof": asof.isoformat(),
                    "ticker": a.ticker,
                    "side": "BUY",
                    "notional_usd": round(notional, 2),
                    "est_shares": round(est_shares, 6) if np.isfinite(est_shares) else "",
                    "price_ref": round(a.close, 4),
                    "source": "holding",
                    "reason": a.reason,
                }
            )

        sell_holdings = [a for a in actions if a.action == "SELL"]
        if sell_holdings:
            print("\n" + "-" * 72)
            print("Holdings con senal SELL: confirmacion de ordenes (venta total).")
            print("-" * 72)
        for a in sell_holdings:
            if not _prompt_yes_no(f"Quieres generar una orden de VENTA TOTAL para {a.ticker}? (s/n): "):
                continue
            orders.append(
                {
                    "asof": asof.isoformat(),
                    "ticker": a.ticker,
                    "side": "SELL",
                    "qty": "ALL",
                    "price_ref": round(a.close, 4),
                    "source": "holding",
                    "reason": a.reason,
                }
            )
            if _prompt_yes_no(f"Quieres abrir CORTO en {a.ticker} despues de vender? (s/n): "):
                short_notional = _prompt_float("Monto USD para corto (ej: 250): ")
                orders.append(
                    {
                        "asof": asof.isoformat(),
                        "ticker": a.ticker,
                        "side": "SHORT",
                        "notional_usd": round(short_notional, 2),
                        "price_ref": round(a.close, 4),
                        "source": "holding",
                        "reason": f"{a.reason} | entrada short",
                    }
                )

    if orders:
        orders_path = out_dir / "orders.csv"
        _write_orders(orders_path, orders)
        print(f"\nOrdenes guardadas en: {orders_path}")

    trade_log_path: Path | None = None
    if holdings_trade_log is not None and not holdings_trade_log.empty:
        trade_log_path = out_dir / "holdings_trade_log.csv"
        _write_trade_log(trade_log_path, holdings_trade_log)
        print(f"\nTrade log diario holdings guardado en: {trade_log_path}")

    if plot_dir is not None:
        print(f"\nGraficos holdings guardados en: {plot_dir}")

    # Save a machine-readable snapshot
    snapshot = {
        "asof": asof.isoformat(),
        "top": [r.__dict__ for r in top],
        "holdings": [a.__dict__ for a in actions],
        "holdings_strategy_profiles": [p.__dict__ for p in strategy_plans],
        "holdings_signal_counts": {"BUY": buy_count, "SELL": sell_count, "HOLD": hold_count},
        "holdings_backtest_1y": [b.__dict__ for b in backtests],
        "holdings_portfolio_backtest_1y": {
            "initial_capital_total": float(sum(b.initial_capital_1y for b in backtests)) if backtests else 0.0,
            "strategy_final_capital_total": float(sum(b.strategy_final_capital_1y for b in backtests)) if backtests else 0.0,
            "buy_hold_final_capital_total": float(sum(b.buy_hold_final_capital_1y for b in backtests)) if backtests else 0.0,
            "target_return": target_return,
            "strategy_return": (
                float(sum(b.strategy_final_capital_1y for b in backtests) / sum(b.initial_capital_1y for b in backtests) - 1.0)
                if backtests and sum(b.initial_capital_1y for b in backtests) > 0
                else 0.0
            ),
            "buy_hold_return": (
                float(sum(b.buy_hold_final_capital_1y for b in backtests) / sum(b.initial_capital_1y for b in backtests) - 1.0)
                if backtests and sum(b.initial_capital_1y for b in backtests) > 0
                else 0.0
            ),
            "target_met": (
                bool(
                    backtests
                    and sum(b.initial_capital_1y for b in backtests) > 0
                    and (
                        float(sum(b.strategy_final_capital_1y for b in backtests) / sum(b.initial_capital_1y for b in backtests) - 1.0)
                        >= target_return
                    )
                )
            ),
        },
        "holdings_trade_log_path": str(trade_log_path) if trade_log_path is not None else None,
        "holdings_trade_log_rows": int(len(holdings_trade_log)) if holdings_trade_log is not None else 0,
        "initial_capital": initial_capital,
        "target_return": target_return,
        "plots_dir": str(plot_dir) if plot_dir is not None else None,
        "generated_orders": orders,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _sanitize_for_json(snapshot)
    (out_dir / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nOutputs guardados en: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
