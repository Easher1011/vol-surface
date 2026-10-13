import numpy as np
import pandas as pd
from vol_surface.local_vol import build_local_vol
from vol_surface.svi import svi_total_variance
from vol_surface.black_scholes import bs_price

N_PATHS = 50_000
TARGET_DAYS = 90      # price options expiring around 3 months out
BARRIER_PCT = 0.90    # barrier 10% below today's spot
SEED = 42             # same random numbers for both models, so the comparison is fair

# --- local vol map, with gaps filled flat so paths that wander far still get a vol ---
lv = build_local_vol(k_range=0.25)
fits, fwd, k_grid, T_edges = lv["fits"], lv["fwd"], lv["k_grid"], lv["T_edges"]
LV = pd.DataFrame(lv["LV"]).ffill(axis=1).bfill(axis=1).values
LV = np.clip(LV, 0.05, 1.0)  # safety cap on extreme edge values

# --- the option: expiry closest to 3 months, strike at the forward ---
i = (fits["T"] - TARGET_DAYS / 365).abs().idxmin()
expiry, T, params = fits.loc[i, ["expiry", "T", "params"]]
F_T, DF = fwd.loc[fwd["expiry"] == expiry, ["F", "DF"]].iloc[0]
K = F_T
B = BARRIER_PCT * lv["spot"]
iv_K = np.sqrt(svi_total_variance(np.log(K / F_T), *params) / T)  # market implied vol at K

print(f"Expiry {expiry} ({round(T * 365)} days), forward {F_T:.1f}")
print(f"Strike {K:.1f}, barrier {B:.1f} ({BARRIER_PCT:.0%} of spot {lv['spot']:.1f})")
print(f"Market implied vol at the strike: {iv_K:.4f}\n")


def local_vol_fn(x, t):
    """Vol for each path, given where it is (x = log of price / forward) and when (t)."""
    row = np.clip(np.searchsorted(T_edges, t, side="right") - 1, 0, len(LV) - 1)
    return np.interp(x, k_grid, LV[row])


def flat_vol_fn(x, t):
    """Black-Scholes: the same vol everywhere, always."""
    return iv_K


def simulate(vol_fn):
    """Simulate daily SPX paths; record final price and whether the barrier was touched."""
    rng = np.random.default_rng(SEED)
    n_steps = int(round(T * 365))
    dt = T / n_steps
    x = np.zeros(N_PATHS)                     # log(price / forward), starts at 0
    hit = np.zeros(N_PATHS, dtype=bool)
    for step in range(n_steps):
        sigma = vol_fn(x, step * dt)
        x += -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(N_PATHS)
        F_t = np.interp((step + 1) * dt, fwd["T"], fwd["F"])   # forward at this date
        hit |= F_t * np.exp(x) <= B                            # daily barrier check
    return F_T * np.exp(x), hit


def price(S_T, hit):
    """Discounted average payoff, plus its standard error."""
    put = np.maximum(K - S_T, 0)
    payoffs = {"Vanilla put": put,
               "Down-and-in put": put * hit,
               "Down-and-out put": put * ~hit}
    out = {name: DF * p.mean() for name, p in payoffs.items()}
    out["Std error (vanilla)"] = DF * put.std() / np.sqrt(N_PATHS)
    out["Barrier hit probability"] = hit.mean()
    return out


results = pd.DataFrame({
    "Local vol": price(*simulate(local_vol_fn)),
    "Flat vol": price(*simulate(flat_vol_fn)),
})
print(results.round(3).to_string())

bs = bs_price(F_T * DF, K, T, -np.log(DF) / T, iv_K, "put")
print(f"\nBlack-Scholes vanilla put (exact formula): {bs:.3f}")