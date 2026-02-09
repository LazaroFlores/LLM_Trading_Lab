from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore


DAILY_UPDATES_FILE = "Daily Updates.csv"
TRADE_LOG_FILE = "Trade Log.csv"


def _best_effort_utf8_stdout() -> None:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_ymd(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _orders_from_holdings_trade_log(
    path: Path,
    mode: str,
    log_date: date | None,
    open_notional: float,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe holdings_trade_log.csv: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["asof", "ticker", "side", "notional_usd", "qty", "reason"])

    required = {"date", "ticker", "trade_action"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"holdings_trade_log.csv requiere columnas: {sorted(required)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["trade_action"] = df["trade_action"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date"])
    df = df[df["ticker"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["asof", "ticker", "side", "notional_usd", "qty", "reason"])

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"latest", "date", "all"}:
        raise ValueError("holdings_log_mode invalido. Usa: latest | date | all")

    if mode_norm == "latest":
        target = df["date"].max().date()
        df = df[df["date"].dt.date == target].copy()
    elif mode_norm == "date":
        if log_date is None:
            raise ValueError("Con holdings_log_mode=date debes enviar --holdings-log-date YYYY-MM-DD")
        df = df[df["date"].dt.date == log_date].copy()
    # mode all -> no filter

    df = df.sort_values(["date", "ticker"], ascending=[True, True], kind="stable")

    n_open = float(open_notional)
    if not np.isfinite(n_open) or n_open <= 0:
        raise ValueError("--open-notional debe ser > 0")

    rows: list[dict] = []
    for _, row in df.iterrows():
        asof_iso = pd.Timestamp(row["date"]).date().isoformat()
        ticker = str(row["ticker"]).strip().upper()
        action = str(row["trade_action"]).strip().upper()
        reason = f"holdings_trade_log:{action}"

        if action in {"", "HOLD", "NONE", "NAN"}:
            continue
        if action == "BUY":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "BUY", "notional_usd": round(n_open, 6), "reason": reason})
            continue
        if action == "SHORT":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "SHORT", "notional_usd": round(n_open, 6), "reason": reason})
            continue
        if action == "SELL_TO_CLOSE":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "SELL", "qty": "ALL", "reason": reason})
            continue
        if action == "COVER":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "COVER", "qty": "ALL", "reason": reason})
            continue
        if action == "FLIP_TO_SHORT":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "SELL", "qty": "ALL", "reason": reason})
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "SHORT", "notional_usd": round(n_open, 6), "reason": reason})
            continue
        if action == "FLIP_TO_LONG":
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "COVER", "qty": "ALL", "reason": reason})
            rows.append({"asof": asof_iso, "ticker": ticker, "side": "BUY", "notional_usd": round(n_open, 6), "reason": reason})
            continue

    if not rows:
        return pd.DataFrame(columns=["asof", "ticker", "side", "notional_usd", "qty", "reason"])
    return pd.DataFrame(rows)


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float
    stop_loss: float | None = None


def _latest_snapshot(daily: pd.DataFrame) -> tuple[pd.Timestamp, dict[str, Position], float]:
    if "Date" not in daily.columns or "Ticker" not in daily.columns:
        raise ValueError("Daily Updates no tiene columnas requeridas: Date, Ticker")

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    if daily["Date"].dropna().empty:
        raise ValueError("Daily Updates: no se pudo parsear Date")

    last_dt = daily["Date"].max()
    day = daily[daily["Date"] == last_dt].copy()

    total = day[day["Ticker"].astype(str).str.upper() == "TOTAL"]
    if total.empty:
        raise ValueError("Daily Updates: falta fila TOTAL en el último día")

    cash = float(pd.to_numeric(total.iloc[-1].get("Cash Balance"), errors="coerce"))
    if not np.isfinite(cash):
        raise ValueError("Daily Updates: Cash Balance inválido en TOTAL")

    positions: dict[str, Position] = {}
    non_total = day[day["Ticker"].astype(str).str.upper() != "TOTAL"].copy()
    if not non_total.empty:
        non_total["Shares"] = pd.to_numeric(non_total.get("Shares"), errors="coerce")
        non_total["Buy Price"] = pd.to_numeric(non_total.get("Buy Price"), errors="coerce")
        non_total["Stop Loss"] = pd.to_numeric(non_total.get("Stop Loss"), errors="coerce")
        for _, row in non_total.iterrows():
            t = str(row["Ticker"]).strip().upper()
            sh = float(row["Shares"]) if np.isfinite(row["Shares"]) else 0.0
            if sh == 0.0:
                continue
            ap = float(row["Buy Price"]) if np.isfinite(row["Buy Price"]) else np.nan
            if not np.isfinite(ap) or ap <= 0:
                # fallback: if no buy price, approximate using Cost Basis / Shares if possible
                cb = pd.to_numeric(row.get("Cost Basis"), errors="coerce")
                if np.isfinite(cb) and cb and sh:
                    ap = float(cb) / float(sh)
            sl = float(row["Stop Loss"]) if np.isfinite(row["Stop Loss"]) and float(row["Stop Loss"]) > 0 else None
            positions[t] = Position(ticker=t, shares=float(sh), avg_price=float(ap), stop_loss=sl)

    return last_dt.normalize(), positions, cash


def _download_day(ticker: str, start: date, end_inclusive: date) -> pd.DataFrame:
    end_excl = end_inclusive + timedelta(days=1)
    df = yf.download(
        tickers=[ticker],
        start=start.isoformat(),
        end=end_excl.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    # Single ticker download typically returns flat columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def _next_session_price(
    ticker: str,
    asof: date,
    price_field: str,
    max_lookahead_days: int = 14,
) -> tuple[date, float]:
    """
    Return (session_date, price) for the first available trading session after `asof`.
    Uses yfinance data; skips weekends/holidays automatically by scanning forward.
    """
    if price_field not in {"Open", "Close"}:
        raise ValueError("price_field debe ser Open o Close")

    start = asof + timedelta(days=1)
    end = asof + timedelta(days=max_lookahead_days)
    df = _download_day(ticker, start=start, end_inclusive=end)
    if df.empty or price_field not in df.columns:
        raise RuntimeError(f"No hay datos para {ticker} entre {start} y {end}")

    df = df.sort_index()
    # take first row
    idx = df.index[0]
    session = pd.Timestamp(idx).date()
    px = float(df[price_field].iloc[0])
    if not np.isfinite(px) or px <= 0:
        raise RuntimeError(f"Precio inválido {ticker} {session} {price_field}={px}")
    return session, px


def _session_close(ticker: str, session: date) -> float:
    df = _download_day(ticker, start=session, end_inclusive=session)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No se pudo obtener Close para {ticker} en {session}")
    px = float(df.sort_index()["Close"].iloc[-1])
    if not np.isfinite(px) or px <= 0:
        raise RuntimeError(f"Close inválido {ticker} {session}={px}")
    return px


def _apply_slippage(side: str, price: float, slippage_bps: float) -> float:
    bps = float(slippage_bps)
    if bps <= 0:
        return price
    mult = 1.0 + (bps / 10_000.0)
    if side in {"BUY", "COVER"}:
        return price * mult
    if side in {"SELL", "SHORT"}:
        return price / mult
    return price


def _realized_pnl_for_close(
    side: str,
    shares: float,
    fill_price: float,
    pos: Position,
) -> float:
    """
    Realized pnl for SELL/COVER when reducing an existing position at avg_price.
    Assumes `shares` is positive quantity being sold/bought-to-cover.
    """
    if side == "SELL" and pos.shares > 0:
        return shares * (fill_price - pos.avg_price)
    if side == "COVER" and pos.shares < 0:
        return shares * (pos.avg_price - fill_price)
    return 0.0


def _update_position_after_trade(
    positions: dict[str, Position],
    ticker: str,
    side: str,
    shares: float,
    fill_price: float,
) -> tuple[Position | None, float]:
    """
    Returns (new_position_or_none, realized_pnl).
    """
    t = ticker.upper()
    pos = positions.get(t)

    if side == "BUY":
        if pos is None:
            return Position(ticker=t, shares=shares, avg_price=fill_price), 0.0
        # If covering part of a short, treat as COVER
        if pos.shares < 0:
            cover_qty = min(shares, abs(pos.shares))
            realized = _realized_pnl_for_close("COVER", cover_qty, fill_price, pos)
            remaining_buy = shares - cover_qty
            new_shares = pos.shares + cover_qty  # moves toward 0
            if new_shares == 0 and remaining_buy == 0:
                return None, realized
            if new_shares == 0 and remaining_buy > 0:
                return Position(ticker=t, shares=remaining_buy, avg_price=fill_price), realized
            # still short after cover
            if remaining_buy == 0:
                return Position(ticker=t, shares=new_shares, avg_price=pos.avg_price, stop_loss=pos.stop_loss), realized
            # cover fully and buy long
            return Position(ticker=t, shares=remaining_buy, avg_price=fill_price), realized

        # add to long
        new_shares = pos.shares + shares
        new_avg = (pos.avg_price * pos.shares + fill_price * shares) / new_shares if new_shares else fill_price
        return Position(ticker=t, shares=new_shares, avg_price=new_avg, stop_loss=pos.stop_loss), 0.0

    if side == "SELL":
        if pos is None or pos.shares == 0:
            # Selling without a position -> treat as SHORT entry
            return Position(ticker=t, shares=-shares, avg_price=fill_price), 0.0
        if pos.shares > 0:
            sell_qty = min(shares, pos.shares)
            realized = _realized_pnl_for_close("SELL", sell_qty, fill_price, pos)
            new_shares = pos.shares - sell_qty
            if new_shares == 0:
                # If user sells more than long shares, remainder becomes short
                extra = shares - sell_qty
                if extra > 0:
                    return Position(ticker=t, shares=-extra, avg_price=fill_price), realized
                return None, realized
            return Position(ticker=t, shares=new_shares, avg_price=pos.avg_price, stop_loss=pos.stop_loss), realized
        # increasing short (SELL more)
        new_shares = pos.shares - shares
        new_avg = (pos.avg_price * abs(pos.shares) + fill_price * shares) / abs(new_shares) if new_shares else fill_price
        return Position(ticker=t, shares=new_shares, avg_price=new_avg, stop_loss=pos.stop_loss), 0.0

    if side == "SHORT":
        if pos is None:
            return Position(ticker=t, shares=-shares, avg_price=fill_price), 0.0
        if pos.shares > 0:
            # First close long, then open short with remainder
            sell_qty = min(shares, pos.shares)
            realized = _realized_pnl_for_close("SELL", sell_qty, fill_price, pos)
            remaining = shares - sell_qty
            if remaining <= 0:
                new_shares = pos.shares - sell_qty
                if new_shares == 0:
                    return None, realized
                return Position(ticker=t, shares=new_shares, avg_price=pos.avg_price, stop_loss=pos.stop_loss), realized
            # open short
            return Position(ticker=t, shares=-remaining, avg_price=fill_price), realized
        # already short -> add
        new_shares = pos.shares - shares
        new_avg = (pos.avg_price * abs(pos.shares) + fill_price * shares) / abs(new_shares)
        return Position(ticker=t, shares=new_shares, avg_price=new_avg, stop_loss=pos.stop_loss), 0.0

    if side == "COVER":
        if pos is None or pos.shares >= 0:
            return pos, 0.0
        cover_qty = min(shares, abs(pos.shares))
        realized = _realized_pnl_for_close("COVER", cover_qty, fill_price, pos)
        new_shares = pos.shares + cover_qty
        if new_shares == 0:
            return None, realized
        return Position(ticker=t, shares=new_shares, avg_price=pos.avg_price, stop_loss=pos.stop_loss), realized

    raise ValueError(f"Side no soportado: {side}")


def _append_trade_log(trade_log_path: Path, rows: list[dict]) -> None:
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    if trade_log_path.exists():
        df = pd.read_csv(trade_log_path)
        out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if not df.empty else pd.DataFrame(rows)
    else:
        out = pd.DataFrame(rows)
    out.to_csv(trade_log_path, index=False)


def _append_daily_updates(daily_path: Path, rows: list[dict], replace_date: date | None = None) -> None:
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    if daily_path.exists():
        df = pd.read_csv(daily_path)
        if replace_date is not None and not df.empty and "Date" in df.columns:
            iso = replace_date.isoformat()
            df = df[df["Date"].astype(str) != iso].copy()
        out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if not df.empty else pd.DataFrame(rows)
    else:
        out = pd.DataFrame(rows)
    out.to_csv(daily_path, index=False)


def _mtm_rows(
    session: date,
    positions: dict[str, Position],
    cash: float,
) -> tuple[list[dict], float]:
    rows: list[dict] = []
    equity_positions = 0.0
    pnl_sum = 0.0
    for t, pos in sorted(positions.items(), key=lambda x: x[0]):
        close = _session_close(t, session)
        total_value = pos.shares * close
        pnl = pos.shares * (close - pos.avg_price)
        equity_positions += total_value
        pnl_sum += pnl
        rows.append(
            {
                "Date": session.isoformat(),
                "Ticker": t,
                "Shares": pos.shares,
                "Buy Price": round(pos.avg_price, 6),
                "Cost Basis": round(abs(pos.shares) * pos.avg_price, 6),
                "Stop Loss": pos.stop_loss if pos.stop_loss is not None else "",
                "Current Price": round(close, 6),
                "Total Value": round(total_value, 6),
                "PnL": round(pnl, 6),
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            }
        )

    total_equity = cash + equity_positions
    rows.append(
        {
            "Date": session.isoformat(),
            "Ticker": "TOTAL",
            "Shares": "",
            "Buy Price": "",
            "Cost Basis": "",
            "Stop Loss": "",
            "Current Price": "",
            "Total Value": round(equity_positions, 6),
            "PnL": round(pnl_sum, 6),
            "Action": "",
            "Cash Balance": round(cash, 6),
            "Total Equity": round(total_equity, 6),
        }
    )
    return rows, total_equity


def main() -> int:
    _best_effort_utf8_stdout()
    parser = argparse.ArgumentParser(description="Paper trading: aplica orders.csv y actualiza CSVs del experimento (long/short).")
    parser.add_argument("--data-dir", required=True, help="Directorio que contiene Daily Updates.csv y Trade Log.csv.")
    parser.add_argument("--orders", default=None, help="Ruta a orders.csv (generado por recommend.py).")
    parser.add_argument("--holdings-trade-log", default=None, help="Ruta a holdings_trade_log.csv generado por recommend.py.")
    parser.add_argument("--holdings-log-mode", choices=["latest", "date", "all"], default="latest", help="Como convertir holdings_trade_log a ordenes: latest/date/all.")
    parser.add_argument("--holdings-log-date", default=None, help="Fecha YYYY-MM-DD para holdings_log_mode=date.")
    parser.add_argument("--open-notional", type=float, default=100.0, help="Notional por entrada BUY/SHORT al usar holdings_trade_log (default: 100).")
    parser.add_argument("--asof", default=None, help="Fecha YYYY-MM-DD para MTM si no hay ordenes (default: hoy).")
    parser.add_argument("--fill", choices=["next_open", "next_close"], default="next_open", help="Regla de ejecucion.")
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="Slippage en bps (default: 10 = 0.10%%).")
    parser.add_argument("--fee", type=float, default=0.0, help="Fee fijo por orden (en moneda del experimento).")
    parser.add_argument("--force", action="store_true", help="Permite re-aplicar ordenes en una fecha (puede duplicar el Trade Log).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    daily_path = data_dir / DAILY_UPDATES_FILE
    trade_path = data_dir / TRADE_LOG_FILE

    if not daily_path.exists() or not trade_path.exists():
        raise FileNotFoundError(f"No encuentro CSVs en {data_dir}. Inicializa con init_experiment.py o crea los archivos.")

    daily = _read_csv(daily_path)
    last_snapshot_dt, positions, cash = _latest_snapshot(daily)

    existing_trades = pd.DataFrame()
    if trade_path.exists():
        existing_trades = pd.read_csv(trade_path)
        if not existing_trades.empty and "Date" in existing_trades.columns:
            existing_trades["Date"] = pd.to_datetime(existing_trades["Date"], errors="coerce")
        if not existing_trades.empty and "Ticker" in existing_trades.columns:
            existing_trades["Ticker"] = existing_trades["Ticker"].astype(str).str.upper()

    if args.orders and args.holdings_trade_log:
        raise ValueError("Usa solo uno: --orders o --holdings-trade-log")

    orders_df: pd.DataFrame | None = None
    if args.orders:
        orders_path = Path(args.orders)
        if not orders_path.exists():
            raise FileNotFoundError(f"No existe orders.csv: {orders_path}")
        orders_df = pd.read_csv(orders_path)
    elif args.holdings_trade_log:
        log_path = Path(args.holdings_trade_log)
        log_date = _parse_ymd(args.holdings_log_date)
        orders_df = _orders_from_holdings_trade_log(
            path=log_path,
            mode=str(args.holdings_log_mode),
            log_date=log_date,
            open_notional=float(args.open_notional),
        )
        print(
            "Ordenes derivadas desde holdings_trade_log: "
            f"{len(orders_df)} (mode={args.holdings_log_mode}, date={args.holdings_log_date or 'auto'})"
        )

    fill_field = "Open" if args.fill == "next_open" else "Close"
    slippage_bps = float(args.slippage_bps)
    fee = float(args.fee)

    trade_rows: list[dict] = []
    mtm_session: date | None = None

    if orders_df is not None and not orders_df.empty:
        required = {"asof", "ticker", "side"}
        if not required.issubset(set(orders_df.columns)):
            raise ValueError(f"orders.csv requiere columnas: {sorted(required)}")

        # Apply orders in file order (stable)
        for _, o in orders_df.iterrows():
            asof = _coerce_asof(str(o["asof"]) if pd.notna(o["asof"]) else None)
            t = str(o["ticker"]).strip().upper()
            side = str(o["side"]).strip().upper()
            notional = float(o["notional_usd"]) if "notional_usd" in o and pd.notna(o["notional_usd"]) else np.nan
            shares = float(o["shares"]) if "shares" in o and pd.notna(o["shares"]) else np.nan
            qty = str(o["qty"]).strip().upper() if "qty" in o and pd.notna(o["qty"]) else ""
            reason = str(o["reason"]) if "reason" in o and pd.notna(o["reason"]) else ""

            session, ref_px = _next_session_price(t, asof=asof, price_field=fill_field)
            mtm_session = session  # last applied session wins (typical daily run)
            fill_px = _apply_slippage(side, ref_px, slippage_bps=slippage_bps)

            pos = positions.get(t)
            # Determine shares
            if qty == "ALL":
                if pos is None:
                    continue
                if side in {"SELL", "COVER"}:
                    shares_qty = abs(pos.shares)
                else:
                    shares_qty = abs(pos.shares)
            elif np.isfinite(shares) and shares > 0:
                shares_qty = shares
            elif np.isfinite(notional) and notional > 0:
                shares_qty = notional / fill_px
            else:
                raise ValueError(f"Orden invalida para {t}: requiere shares, notional_usd o qty=ALL")

            # Idempotence guard: avoid duplicate trade rows for the same session/ticker
            if (not args.force) and (not existing_trades.empty) and ("Date" in existing_trades.columns) and ("Ticker" in existing_trades.columns):
                mask = (existing_trades["Date"].dt.date == session) & (existing_trades["Ticker"] == t)
                if mask.any():
                    prior = existing_trades.loc[mask].copy()
                    sb = prior.get("Shares Bought")
                    ss = prior.get("Shares Sold")
                    has_buy = False
                    has_sell = False
                    if sb is not None:
                        sbn = pd.to_numeric(sb, errors="coerce")
                        has_buy = bool((sbn.fillna(0) > 0).any())
                    if ss is not None:
                        ssn = pd.to_numeric(ss, errors="coerce")
                        has_sell = bool((ssn.fillna(0) > 0).any())
                    if side in {"BUY", "COVER"} and has_buy:
                        raise RuntimeError(f"Ya existen compras en Trade Log para {t} el {session}. Usa --force si quieres duplicar.")
                    if side in {"SELL", "SHORT"} and has_sell:
                        raise RuntimeError(f"Ya existen ventas en Trade Log para {t} el {session}. Usa --force si quieres duplicar.")

            # Cash impact
            gross = shares_qty * fill_px
            if side in {"BUY", "COVER"}:
                cash -= gross
            elif side in {"SELL", "SHORT"}:
                cash += gross
            else:
                raise ValueError(f"Side no soportado: {side}")
            cash -= fee

            new_pos, realized = _update_position_after_trade(positions, t, side, shares_qty, fill_px)
            if new_pos is None:
                positions.pop(t, None)
            else:
                positions[t] = new_pos

            # Trade log row (compatible schema)
            row = {
                "Date": session.isoformat(),
                "Ticker": t,
                "Shares Bought": "",
                "Buy Price": "",
                "Cost Basis": round(abs(shares_qty) * fill_px, 6),
                "PnL": round(float(realized), 6),
                "Reason": reason or side,
                "Shares Sold": "",
                "Sell Price": "",
            }
            if side in {"BUY", "COVER"}:
                row["Shares Bought"] = round(float(shares_qty), 6)
                row["Buy Price"] = round(float(fill_px), 6)
            else:
                row["Shares Sold"] = round(float(shares_qty), 6)
                row["Sell Price"] = round(float(fill_px), 6)
            trade_rows.append(row)

    # Mark-to-market snapshot
    if mtm_session is None:
        # MTM only mode
        mtm_session = _coerce_asof(args.asof)

    daily_rows, total_equity = _mtm_rows(mtm_session, positions, cash)

    if trade_rows:
        _append_trade_log(trade_path, trade_rows)
    _append_daily_updates(daily_path, daily_rows, replace_date=mtm_session)

    print(f"MTM date: {mtm_session}  total_equity={total_equity:,.2f}  cash={cash:,.2f}  positions={len(positions)}")
    if trade_rows:
        print(f"Applied orders: {len(trade_rows)} (fill={args.fill}, slippage_bps={slippage_bps}, fee={fee})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
