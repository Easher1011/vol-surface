from datetime import date
from pathlib import Path
import pandas as pd
import yfinance as yf

def fetch_chain(ticker: str = "^SPX", n_expiries: int = 40) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    frames = []
    for expiry in tk.options[:n_expiries]:
        chain = tk.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.assign(type=kind, expiry=expiry, spot=spot)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_chain()
    df.to_csv(out / f"spx_{date.today()}.csv", index=False)
    print(df[["expiry", "type", "strike", "bid", "ask", "impliedVolatility"]].head(20))