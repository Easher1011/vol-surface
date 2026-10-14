"""Daily snapshot of index option chains. Meant to run once per trading day during US hours."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf
from vol_surface.data import fetch_chain

TICKERS = {"spx": "^SPX", "ndx": "^NDX", "rut": "^RUT"}
ROOT = Path(__file__).resolve().parents[2]          # the vol-surface folder, wherever this runs from
OUT = ROOT / "data" / "raw"
LOG = ROOT / "data" / "collect_log.txt"


def log(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a") as f:
        f.write(f"{stamp}  {message}\n")
    print(message)


def market_open():
    """True only on weekdays between 09:35 and 16:00 New York time."""
    ny = datetime.now(ZoneInfo("America/New_York"))
    minutes = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and 9 * 60 + 35 <= minutes <= 16 * 60


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not market_open():
        log("US market closed: skipped")
    else:
        today = datetime.now(ZoneInfo("America/New_York")).date()
        for name, ticker in TICKERS.items():
            try:
                df = fetch_chain(ticker, n_expiries=40)
                df["fetched_at"] = datetime.now().isoformat(timespec="seconds")
                df.to_csv(OUT / f"{name}_{today}.csv", index=False)
                log(f"{name}: {len(df)} quotes saved")
            except Exception as e:  # one index failing shouldn't stop the others
                log(f"{name}: FAILED ({e})")

        # record the official VIX at the same moment, for a fair comparison later
        try:
            vix_now = yf.Ticker("^VIX").fast_info.last_price
            with open(OUT / "vix_official.csv", "a") as f:
                f.write(f"{today},{datetime.now().isoformat(timespec='seconds')},{vix_now:.2f}\n")
            log(f"vix: {vix_now:.2f} recorded")
        except Exception as e:
            log(f"vix: FAILED ({e})")
            