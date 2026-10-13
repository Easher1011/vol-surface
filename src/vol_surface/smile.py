from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vol_surface.black_scholes import implied_vol

MIN_DAYS = 7        # very short expiries give unstable implied vols
MAX_SPREAD = 0.10   # drop quotes where bid-ask spread is over 10% of mid
PARITY_BAND = 0.05  # use strikes within 5% of spot to estimate the forward


def load_latest():
    path = sorted(Path("data/raw").glob("spx_*.csv"))[-1]
    as_of = pd.Timestamp(path.stem.split("_")[1])
    return pd.read_csv(path), as_of


def prepare(df, as_of):
    """Add mid price and time to expiry; keep liquid quotes only."""
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["T"] = (pd.to_datetime(df["expiry"]) - as_of).dt.days / 365
    liquid = (
        (df["T"] >= MIN_DAYS / 365)
        & (df["bid"] > 0)
        & ((df["ask"] - df["bid"]) / df["mid"] < MAX_SPREAD)
    )
    return df[liquid]


def implied_forwards(df):
    """Put-call parity: C - P = DF * (F - K). Regress C - P on K for each expiry."""
    rows = []
    for expiry, g in df.groupby("expiry"):
        calls = g[g["type"] == "call"].set_index("strike")["mid"]
        puts = g[g["type"] == "put"].set_index("strike")["mid"]
        spot = g["spot"].iloc[0]
        strikes = [k for k in calls.index.intersection(puts.index)
                   if abs(k / spot - 1) < PARITY_BAND]
        if len(strikes) < 5:  # not enough matched strikes to trust the fit
            continue
        slope, intercept = np.polyfit(strikes, calls[strikes] - puts[strikes], 1)
        DF = -slope
        rows.append({"expiry": expiry, "F": intercept / DF, "DF": DF,
                     "T": g["T"].iloc[0], "spot": spot})
    fwd = pd.DataFrame(rows)
    fwd["implied_rate"] = -np.log(fwd["DF"]) / fwd["T"]
    fwd["F_over_spot"] = fwd["F"] / fwd["spot"]
    return fwd


def keep_otm(df):
    """Out-of-the-money relative to the forward, not spot."""
    otm = ((df["type"] == "call") & (df["strike"] > df["F"])) | \
          ((df["type"] == "put") & (df["strike"] < df["F"]))
    return df[otm]


def add_implied_vols(df):
    vols = []
    for row in df.itertuples():
        r = -np.log(row.DF) / row.T
        S = row.F * row.DF  # equivalent spot that reproduces the market forward
        try:
            vols.append(implied_vol(row.mid, S, row.strike, row.T, r, row.type))
        except ValueError:
            vols.append(np.nan)
    return df.assign(my_iv=vols).dropna(subset=["my_iv"])


def plot_smile(df, expiry):
    sub = df[df["expiry"] == expiry].sort_values("strike")
    plt.figure(figsize=(10, 5))
    plt.plot(sub["strike"], sub["my_iv"], "o-", label="My implied vol")
    plt.plot(sub["strike"], sub["impliedVolatility"], "x", alpha=0.6, label="yfinance")
    plt.axvline(sub["F"].iloc[0], ls="--", c="grey", label="Forward")
    plt.title(f"SPX volatility smile, expiry {expiry}")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    raw, as_of = load_latest()
    liquid = prepare(raw, as_of)

    fwd = implied_forwards(liquid)
    print(fwd[["expiry", "T", "F", "F_over_spot", "implied_rate"]].round(4).head(10))

    df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
    df = add_implied_vols(keep_otm(df))
    print(f"{len(raw)} raw quotes -> {len(df)} after cleaning")

    expiry = df.loc[(df["T"] - 30 / 365).abs().idxmin(), "expiry"]
    plot_smile(df, expiry)