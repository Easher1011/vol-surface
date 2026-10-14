import numpy as np
import pandas as pd
import yfinance as yf
from vol_surface.smile import load_latest

R = 0.04                # flat rate; changes VIX by only hundredths of a point
MIN_PER_YEAR = 525_600
MIN_30D = 43_200        # 30 days in minutes


def snapshot_time(raw, as_of):
    """When the data was taken, in New York time (4pm close if the file has no timestamp)."""
    if "fetched_at" in raw:
        return (pd.Timestamp(raw["fetched_at"].iloc[0])
                .tz_localize("Europe/London").tz_convert("America/New_York"))
    return pd.Timestamp(as_of.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16)


def minutes_to_expiry(expiry, now):
    """SPXW options settle at 4pm New York time on expiry day."""
    settle = pd.Timestamp(expiry).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return (settle - now).total_seconds() / 60


def expiry_variance(chain, T):
    """Cboe single-expiry variance, following the VIX methodology."""
    chain = chain.assign(bid=chain["bid"].fillna(0))    # missing bid counts as a zero bid
    chain = chain[chain["ask"].notna()]                  # can't price a quote with no ask
    chain = chain.assign(mid=(chain["bid"] + chain["ask"]) / 2)
    calls = chain[chain["type"] == "call"].set_index("strike").sort_index()
    puts = chain[chain["type"] == "put"].set_index("strike").sort_index()
    both = calls.index.intersection(puts.index)

    # forward: from the strike where call and put prices are closest
    gap = (calls.loc[both, "mid"] - puts.loc[both, "mid"]).abs()
    K_star = gap.idxmin()
    growth = np.exp(R * T)
    F = K_star + growth * (calls.at[K_star, "mid"] - puts.at[K_star, "mid"])
    K0 = both[both <= F].max()  # first strike at or below the forward

    def walk(side, strikes):
        """Move away from K0, keeping options until two zero bids in a row."""
        picked, zeros = [], 0
        for K in strikes:
            if side.at[K, "bid"] <= 0:
                zeros += 1
                if zeros == 2:
                    break
                continue
            zeros = 0
            picked.append((K, side.at[K, "mid"]))
        return picked

    put_side = walk(puts, sorted(puts.index[puts.index < K0], reverse=True))
    call_side = walk(calls, sorted(calls.index[calls.index > K0]))
    k0_price = (calls.at[K0, "mid"] + puts.at[K0, "mid"]) / 2   # average of call and put at K0

    pts = sorted(put_side + [(K0, k0_price)] + call_side)
    K = np.array([p[0] for p in pts])
    Q = np.array([p[1] for p in pts])
    dK = np.gradient(K)  # half the gap between neighbouring strikes (one-sided at the ends)

    var = (2 / T) * np.sum(dK / K ** 2 * growth * Q) - (1 / T) * (F / K0 - 1) ** 2
    return var, F, K0, len(K)


if __name__ == "__main__":
    raw, as_of = load_latest()
    now = snapshot_time(raw, as_of)
    raw = raw[raw["contractSymbol"].str.startswith("SPXW")]

    mins = {e: minutes_to_expiry(e, now) for e in raw["expiry"].unique()}
    near = max((e for e, m in mins.items() if m <= MIN_30D), key=lambda e: mins[e])
    nxt = min((e for e, m in mins.items() if m > MIN_30D), key=lambda e: mins[e])

    print(f"Snapshot: {now:%Y-%m-%d %H:%M} New York time\n")
    parts = {}
    for e in (near, nxt):
        T = mins[e] / MIN_PER_YEAR
        var, F, K0, n = expiry_variance(raw[raw["expiry"] == e], T)
        parts[e] = (T, var)
        print(f"{e}: {mins[e] / 1440:.2f} days, forward {F:.1f}, K0 {K0:.0f}, "
              f"{n} options, vol {np.sqrt(var):.2%}")

    (T1, v1), (T2, v2) = parts[near], parts[nxt]
    N1, N2 = mins[near], mins[nxt]
    w1 = (N2 - MIN_30D) / (N2 - N1)   # weight on the near expiry
    my_vix = 100 * np.sqrt((T1 * v1 * w1 + T2 * v2 * (1 - w1)) * MIN_PER_YEAR / MIN_30D)

    hist = yf.Ticker("^VIX").history(period="5d", interval="1m")
    i = hist.index.get_indexer([now], method="nearest")[0]
    official = hist["Close"].iloc[i]

    print(f"\nMy VIX:       {my_vix:.2f}")
    print(f"Official VIX: {official:.2f}  (at {hist.index[i]:%H:%M} New York time)")
    print(f"Difference:   {my_vix - official:+.2f}")