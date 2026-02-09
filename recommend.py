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
DEFAULT_OUTPUT_DIR = Path("runs")
DEFAULT_DAILY_UPDATES_CSV = Path("Experiments/chatgpt_micro-cap/csv_files/Daily Updates.csv")


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


def _prepare_ohlc(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    cols = ["Close", "High", "Low"]
    if not all(c in hist.columns for c in cols):
        return pd.DataFrame()
    out = hist[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols).sort_index()
    return out


def _channel_indicators(
    ohlc: pd.DataFrame,
    channel_window: int = 55,
    floor_pct: float = 0.25,
    ceiling_pct: float = 0.75,
) -> pd.DataFrame:
    d = ohlc.copy()
    d["sma50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["sma200"] = d["Close"].rolling(200, min_periods=200).mean()
    d["channel_high"] = d["High"].rolling(channel_window, min_periods=channel_window).max()
    d["channel_low"] = d["Low"].rolling(channel_window, min_periods=channel_window).min()
    width = (d["channel_high"] - d["channel_low"]).clip(lower=np.nan)
    d["channel_width"] = width
    d["zone"] = (d["Close"] - d["channel_low"]) / d["channel_width"]

    d["trend_up"] = (d["sma50"] > d["sma200"]) & (d["Close"] > d["sma50"])
    d["trend_down"] = (d["sma50"] < d["sma200"]) & (d["Close"] < d["sma50"])

    # Breakouts use previous channel to avoid same-bar lookahead.
    d["breakout_up"] = d["Close"] > (d["channel_high"].shift(1) * 1.005)
    d["breakout_down"] = d["Close"] < (d["channel_low"].shift(1) * 0.995)

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

    d["near_floor"] = d["zone"] <= floor_pct
    d["near_ceiling"] = d["zone"] >= ceiling_pct
    d["exit_long"] = (d["Close"] < d["channel_low"] * 0.995) | (d["Close"] < d["fib_786"] * 0.995) | (d["breakout_down"] & d["trend_down"])
    d["exit_short"] = (d["Close"] > d["channel_high"] * 1.005) | (d["Close"] > d["fib_236"] * 1.005) | (d["breakout_up"] & d["trend_up"])
    return d


def _holding_action_from_channels(
    ticker: str,
    ind: pd.DataFrame,
    stop_loss: float | None,
) -> HoldingAction:
    if ind.empty:
        return HoldingAction(ticker=ticker, action="HOLD", close=float("nan"), stop_loss=stop_loss, reason="sin datos OHLC")

    last = ind.iloc[-1]
    close = float(last["Close"])
    stop_v = float(stop_loss) if stop_loss is not None else None

    if not np.isfinite(close):
        return HoldingAction(ticker=ticker, action="HOLD", close=float("nan"), stop_loss=stop_v, reason="close invalido")

    if stop_v is not None and stop_v > 0 and close < stop_v:
        return HoldingAction(
            ticker=ticker,
            action="SELL",
            close=close,
            stop_loss=stop_v,
            reason=f"precio < stop_loss ({close:.2f} < {stop_v:.2f})",
        )

    needed = ["channel_high", "channel_low", "zone", "sma50", "sma200"]
    if any(not np.isfinite(float(last[c])) for c in needed):
        return HoldingAction(ticker=ticker, action="HOLD", close=close, stop_loss=stop_v, reason="historial insuficiente para canales (>=200d)")

    zone = float(last["zone"])
    up = bool(last["trend_up"])
    down = bool(last["trend_down"])
    breakout_up = bool(last["breakout_up"])
    breakout_down = bool(last["breakout_down"])
    near_floor = bool(last["near_floor"])
    near_ceiling = bool(last["near_ceiling"])
    fib_support = bool(last["fib_support"])
    fib_resistance = bool(last["fib_resistance"])
    fib_momo_up = bool(last["fib_momo_up"])
    fib_momo_down = bool(last["fib_momo_down"])

    if breakout_down and down:
        return HoldingAction(
            ticker=ticker,
            action="SELL",
            close=close,
            stop_loss=stop_v,
            reason=f"ruptura bajista de piso/canal (zone={zone:.2f}); salida defensiva",
        )
    if (near_ceiling or fib_resistance) and down:
        reason = f"techo/canal con sesgo bajista (zone={zone:.2f}); salida defensiva"
        if fib_resistance:
            reason = f"resistencia Fibonacci + sesgo bajista (zone={zone:.2f}); salida defensiva"
        return HoldingAction(
            ticker=ticker,
            action="SELL",
            close=close,
            stop_loss=stop_v,
            reason=reason,
        )
    if breakout_up and up:
        return HoldingAction(
            ticker=ticker,
            action="BUY",
            close=close,
            stop_loss=stop_v,
            reason=f"ruptura alcista de canal (zone={zone:.2f})",
        )
    if (near_floor or fib_support) and up:
        reason = f"piso/canal con sesgo alcista (zone={zone:.2f})"
        if fib_support:
            reason = f"soporte Fibonacci + sesgo alcista (zone={zone:.2f})"
        return HoldingAction(
            ticker=ticker,
            action="BUY",
            close=close,
            stop_loss=stop_v,
            reason=reason,
        )
    if fib_momo_up and up and (not fib_momo_down):
        return HoldingAction(
            ticker=ticker,
            action="BUY",
            close=close,
            stop_loss=stop_v,
            reason=f"momentum sobre Fib 50% con sesgo alcista (zone={zone:.2f})",
        )
    return HoldingAction(
        ticker=ticker,
        action="HOLD",
        close=close,
        stop_loss=stop_v,
        reason=f"en rango de canal (zone={zone:.2f})",
    )


def _backtest_channel_1y(
    ticker: str,
    ind: pd.DataFrame,
    backtest_days: int = 252,
    fee_bps: float = 0.0,
    initial_capital: float = 100.0,
) -> HoldingBacktest | None:
    if ind.empty or len(ind) < 220:
        return None

    d = ind.copy()
    d = d.dropna(subset=["Close"])
    if len(d) < 220:
        return None

    fib_support = d["fib_support"].fillna(False) if "fib_support" in d.columns else pd.Series(False, index=d.index)
    fib_resistance = d["fib_resistance"].fillna(False) if "fib_resistance" in d.columns else pd.Series(False, index=d.index)
    fib_momo_up = d["fib_momo_up"].fillna(False) if "fib_momo_up" in d.columns else pd.Series(False, index=d.index)
    fib_momo_down = d["fib_momo_down"].fillna(False) if "fib_momo_down" in d.columns else pd.Series(False, index=d.index)
    buy_sig = ((((d["near_floor"] | fib_support | fib_momo_up) & d["trend_up"]) | d["breakout_up"])).fillna(False)
    sell_sig = ((((d["near_ceiling"] | fib_resistance | fib_momo_down) & d["trend_down"]) | d["breakout_down"])).fillna(False)
    exit_long = d["exit_long"].fillna(False)
    exit_short = d["exit_short"].fillna(False)

    state = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        prev = state[i - 1]
        b = bool(buy_sig.iloc[i])
        s = bool(sell_sig.iloc[i])
        xl = bool(exit_long.iloc[i])
        new_state = prev
        if prev == 0:
            if b:
                new_state = 1
        elif prev == 1:
            if s or xl:
                new_state = 0
        state[i] = new_state

    state_s = pd.Series(state, index=d.index, dtype=float)
    ret = d["Close"].pct_change().fillna(0.0)
    strat_ret = state_s.shift(1).fillna(0.0) * ret
    if fee_bps > 0:
        turn = state_s.diff().abs().fillna(0.0)
        strat_ret = strat_ret - (turn * (fee_bps / 10_000.0))

    n = min(backtest_days, len(d))
    if n <= 2:
        return None

    strat_slice = strat_ret.iloc[-n:]
    bh_slice = ret.iloc[-n:]
    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    eq = cap0 * (1.0 + strat_slice).cumprod()
    bh = cap0 * (1.0 + bh_slice).cumprod()

    max_dd = float((eq / eq.cummax() - 1.0).min())
    trades_1y = int(state_s.iloc[-n:].diff().abs().fillna(0.0).sum())
    strat_final = float(eq.iloc[-1])
    bh_final = float(bh.iloc[-1])
    return HoldingBacktest(
        ticker=ticker,
        initial_capital_1y=cap0,
        strategy_final_capital_1y=strat_final,
        buy_hold_final_capital_1y=bh_final,
        strategy_return_1y=float((strat_final / cap0) - 1.0),
        buy_hold_return_1y=float((bh_final / cap0) - 1.0),
        max_drawdown_1y=max_dd,
        trades_1y=trades_1y,
    )


def _build_strategy_timeseries(
    ind: pd.DataFrame,
    backtest_days: int = 252,
    fee_bps: float = 0.0,
    initial_capital: float = 100.0,
) -> pd.DataFrame:
    if ind.empty or len(ind) < 220:
        return pd.DataFrame()

    d = ind.copy().dropna(subset=["Close"])
    if len(d) < 220:
        return pd.DataFrame()

    fib_support = d["fib_support"].fillna(False) if "fib_support" in d.columns else pd.Series(False, index=d.index)
    fib_resistance = d["fib_resistance"].fillna(False) if "fib_resistance" in d.columns else pd.Series(False, index=d.index)
    fib_momo_up = d["fib_momo_up"].fillna(False) if "fib_momo_up" in d.columns else pd.Series(False, index=d.index)
    fib_momo_down = d["fib_momo_down"].fillna(False) if "fib_momo_down" in d.columns else pd.Series(False, index=d.index)
    buy_sig = ((((d["near_floor"] | fib_support | fib_momo_up) & d["trend_up"]) | d["breakout_up"])).fillna(False)
    sell_sig = ((((d["near_ceiling"] | fib_resistance | fib_momo_down) & d["trend_down"]) | d["breakout_down"])).fillna(False)
    exit_long = d["exit_long"].fillna(False)
    exit_short = d["exit_short"].fillna(False)

    state = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        prev = state[i - 1]
        b = bool(buy_sig.iloc[i])
        s = bool(sell_sig.iloc[i])
        xl = bool(exit_long.iloc[i])
        new_state = prev
        if prev == 0:
            if b:
                new_state = 1
        elif prev == 1:
            if s or xl:
                new_state = 0
        state[i] = new_state

    state_s = pd.Series(state, index=d.index, dtype=float)
    ret = d["Close"].pct_change().fillna(0.0)
    strat_ret = state_s.shift(1).fillna(0.0) * ret
    if fee_bps > 0:
        turn = state_s.diff().abs().fillna(0.0)
        strat_ret = strat_ret - (turn * (fee_bps / 10_000.0))

    n = min(backtest_days, len(d))
    if n <= 2:
        return pd.DataFrame()

    signal = pd.Series(
        np.where(
            buy_sig & ~sell_sig,
            "BUY",
            np.where(sell_sig & ~buy_sig, "SELL", "HOLD"),
        ),
        index=d.index,
    )
    signal_code = signal.map({"SELL": -1, "HOLD": 0, "BUY": 1}).astype(float)

    out = pd.DataFrame(
        {
            "Close": d["Close"].iloc[-n:],
            "strategy_ret": strat_ret.iloc[-n:],
            "buyhold_ret": ret.iloc[-n:],
            "state": state_s.iloc[-n:],
            "channel_high": d["channel_high"].iloc[-n:],
            "channel_low": d["channel_low"].iloc[-n:],
            "fib_236": d["fib_236"].iloc[-n:] if "fib_236" in d.columns else np.nan,
            "fib_382": d["fib_382"].iloc[-n:] if "fib_382" in d.columns else np.nan,
            "fib_618": d["fib_618"].iloc[-n:] if "fib_618" in d.columns else np.nan,
            "fib_786": d["fib_786"].iloc[-n:] if "fib_786" in d.columns else np.nan,
            "signal": signal.iloc[-n:],
            "signal_code": signal_code.iloc[-n:],
        }
    )
    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    out["strategy_eq"] = cap0 * (1.0 + out["strategy_ret"]).cumprod()
    out["buyhold_eq"] = cap0 * (1.0 + out["buyhold_ret"]).cumprod()
    return out


def _save_holding_plot(ticker: str, ts: pd.DataFrame, out_path: Path) -> None:
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
    if "fib_382" in ts and pd.to_numeric(ts["fib_382"], errors="coerce").notna().any():
        ax1.plot(ts.index, ts["fib_382"], label="Fib 38.2%", linewidth=0.9, linestyle="-.", color="#8c564b", alpha=0.75)
    if "fib_618" in ts and pd.to_numeric(ts["fib_618"], errors="coerce").notna().any():
        ax1.plot(ts.index, ts["fib_618"], label="Fib 61.8%", linewidth=0.9, linestyle="-.", color="#17becf", alpha=0.75)
    if "signal" in ts:
        buy_mask = ts["signal"] == "BUY"
        sell_mask = ts["signal"] == "SELL"
        hold_mask = ts["signal"] == "HOLD"
        ax1.scatter(ts.index[buy_mask], ts.loc[buy_mask, "Close"], marker="^", s=40, color="#2ca02c", label="BUY", zorder=4)
        ax1.scatter(ts.index[sell_mask], ts.loc[sell_mask, "Close"], marker="v", s=40, color="#d62728", label="SELL", zorder=4)
        ax1.scatter(ts.index[hold_mask], ts.loc[hold_mask, "Close"], marker="o", s=9, color="#7f7f7f", alpha=0.25, label="HOLD", zorder=2)
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        hold_count = int(hold_mask.sum())
        counts_label = f"BUY={buy_count} | SELL={sell_count} | HOLD={hold_count}"
        ax1.set_title(f"{ticker} - Precio, Canal y Senales ({len(ts)} sesiones) [{counts_label}]")
        ax1.text(
            0.015,
            0.98,
            f"Senales: {counts_label}",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#aaaaaa"},
        )
    else:
        ax1.set_title(f"{ticker} - Precio, Canal y Senales ({len(ts)} sesiones)")
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
    lookback_days: int = 365 * 3,
    backtest_days: int = 252,
    plot_dir: Path | None = None,
    initial_capital: float = 100.0,
) -> tuple[list[HoldingAction], list[HoldingBacktest], pd.DataFrame]:
    if not holdings:
        return [], [], pd.DataFrame()

    stop_losses = stop_losses or {}
    cap0 = float(initial_capital) if np.isfinite(initial_capital) and initial_capital > 0 else 100.0
    start = asof - timedelta(days=lookback_days)

    data = _download_history(holdings, start=start, end_inclusive=asof)
    out: list[HoldingAction] = []
    bt: list[HoldingBacktest] = []
    trade_logs: list[pd.DataFrame] = []

    for t in holdings:
        hist = data.get(t)
        sl = stop_losses.get(t)
        ohlc = _prepare_ohlc(hist if isinstance(hist, pd.DataFrame) else pd.DataFrame())
        ind = _channel_indicators(ohlc)
        out.append(_holding_action_from_channels(ticker=t, ind=ind, stop_loss=sl))
        bt_row = _backtest_channel_1y(ticker=t, ind=ind, backtest_days=backtest_days, initial_capital=cap0)
        if bt_row is not None:
            bt.append(bt_row)

        ts = _build_strategy_timeseries(ind=ind, backtest_days=backtest_days, initial_capital=cap0)
        if not ts.empty:
            trade_logs.append(_build_daily_trade_log(ticker=t, asof=asof, ts=ts))
            if plot_dir is not None:
                safe_name = re.sub(r"[^A-Z0-9._-]", "_", t.upper())
                _save_holding_plot(ticker=t, ts=ts, out_path=plot_dir / f"{safe_name}_strategy.png")

    if trade_logs:
        trade_log_df = pd.concat(trade_logs, axis=0, ignore_index=True)
        trade_log_df = trade_log_df.sort_values(["ticker", "date"], ascending=[True, True], kind="stable")
    else:
        trade_log_df = pd.DataFrame(
            columns=[
                "asof",
                "date",
                "ticker",
                "close",
                "signal",
                "position",
                "trade_action",
                "strategy_ret",
                "strategy_equity_usd",
                "buyhold_equity_usd",
            ]
        )

    return out, bt, trade_log_df


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


def main() -> int:
    _best_effort_utf8_stdout()
    parser = argparse.ArgumentParser(description="Recomendador simple (Top N + seguimiento holdings).")
    parser.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE_PATH), help="Archivo con tickers (uno por línea).")
    parser.add_argument("--holdings", type=str, default=str(DEFAULT_HOLDINGS_PATH), help="Archivo con tickers en cartera (uno por línea).")
    parser.add_argument("--daily-updates", type=str, default=str(DEFAULT_DAILY_UPDATES_CSV), help="Ruta al Daily Updates.csv (para construir universo/holdings).")
    parser.add_argument("--universe-from-daily", action="store_true", help="Usar Daily Updates.csv como universo (en vez de config/universe.txt).")
    parser.add_argument("--holdings-from-daily", action="store_true", help="Derivar holdings desde el último día en Daily Updates.csv (incluye stop-loss si existe).")
    parser.add_argument("--asof", type=str, default=None, help="Fecha YYYY-MM-DD (default: hoy; si es fin de semana usa viernes).")
    parser.add_argument("--top", type=int, default=5, help="Cantidad de recomendaciones Top N (default: 5).")
    parser.add_argument("--min-dollar-vol", type=float, default=500_000.0, help="Filtro de liquidez: promedio 20d Close*Volume.")
    parser.add_argument("--holdings-lookback-years", type=int, default=3, help="Lookback para holdings/channels (default: 3 anios).")
    parser.add_argument("--backtest-days", type=int, default=252, help="Ventana de backtest para holdings (default: 252).")
    parser.add_argument("--initial-capital", type=float, default=100.0, help="Capital inicial para backtest/equity de holdings (default: 100 USD).")
    parser.add_argument("--plot-holdings", action="store_true", help="Genera un grafico por holding (precio+canal con marcas BUY/SELL/HOLD y equity curve).")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directorio de salida para graficos (default: runs/YYYY-MM-DD/plots).")
    parser.add_argument("--non-interactive", action="store_true", help="No preguntar confirmaciones; solo imprime y guarda outputs.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directorio base para outputs (default: runs/).")
    args = parser.parse_args()

    asof = _coerce_asof(args.asof)
    universe_path = Path(args.universe)
    holdings_path = Path(args.holdings)
    daily_updates_path = Path(args.daily_updates)
    out_dir = Path(args.out_dir) / asof.isoformat()

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
    actions, backtests, holdings_trade_log = build_holdings_actions(
        holdings=holdings,
        asof=asof,
        stop_losses=stop_losses,
        lookback_days=max(365, int(args.holdings_lookback_years) * 365),
        backtest_days=max(60, int(args.backtest_days)),
        plot_dir=plot_dir,
        initial_capital=initial_capital,
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
            print(f"{a.ticker:>10}  {a.action:<4}  close={a.close:>9.2f}  stop={sl:>8}  {a.reason}")
        print(f"Resumen señales -> BUY={buy_count}  SELL={sell_count}  HOLD={hold_count}")

    if backtests:
        print(f"\n[Holdings Backtest 1Y | capital inicial={_format_money(initial_capital)}]")
        for b in backtests:
            print(
                f"{b.ticker:>10}  strat={b.strategy_return_1y:+7.1%} ({_format_money(b.strategy_final_capital_1y)})  "
                f"buy&hold={b.buy_hold_return_1y:+7.1%} ({_format_money(b.buy_hold_final_capital_1y)})  "
                f"maxDD={b.max_drawdown_1y:7.1%}  trades={b.trades_1y:>3}"
            )
        agg_initial = float(sum(b.initial_capital_1y for b in backtests))
        agg_strat = float(sum(b.strategy_final_capital_1y for b in backtests))
        agg_bh = float(sum(b.buy_hold_final_capital_1y for b in backtests))
        agg_strat_ret = (agg_strat / agg_initial - 1.0) if agg_initial > 0 else float("nan")
        agg_bh_ret = (agg_bh / agg_initial - 1.0) if agg_initial > 0 else float("nan")
        print(
            f"\n[Holdings Portfolio 1Y]  strat={agg_strat_ret:+7.1%} ({_format_money(agg_strat)})  "
            f"buy&hold={agg_bh_ret:+7.1%} ({_format_money(agg_bh)})"
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
        "holdings_signal_counts": {"BUY": buy_count, "SELL": sell_count, "HOLD": hold_count},
        "holdings_backtest_1y": [b.__dict__ for b in backtests],
        "holdings_portfolio_backtest_1y": {
            "initial_capital_total": float(sum(b.initial_capital_1y for b in backtests)) if backtests else 0.0,
            "strategy_final_capital_total": float(sum(b.strategy_final_capital_1y for b in backtests)) if backtests else 0.0,
            "buy_hold_final_capital_total": float(sum(b.buy_hold_final_capital_1y for b in backtests)) if backtests else 0.0,
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
        },
        "holdings_trade_log_path": str(trade_log_path) if trade_log_path is not None else None,
        "holdings_trade_log_rows": int(len(holdings_trade_log)) if holdings_trade_log is not None else 0,
        "initial_capital": initial_capital,
        "plots_dir": str(plot_dir) if plot_dir is not None else None,
        "generated_orders": orders,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
