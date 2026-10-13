import numpy as np
import pandas as pd
import QuantLib as ql
from vol_surface.local_vol import build_local_vol
from vol_surface.heston import calibrate_heston
from vol_surface.svi import svi_total_variance
from vol_surface.black_scholes import bs_price

N_PATHS = 50_000
TARGET_DAYS = 90      # price options expiring around 3 months out
BARRIER_PCT = 0.90    # barrier 10% below today's spot
SEED = 42
HESTON_SUBSTEPS = 4   # smaller steps for Heston: its vol moves fast, so daily steps are too coarse

# --- local vol map, with gaps filled flat so paths that wander far still get a vol ---
lv = build_local_vol(k_range=0.25)
fits, fwd, k_grid, T_edges = lv["fits"], lv["fwd"], lv["k_grid"], lv["T_edges"]
LV = pd.DataFrame(lv["LV"]).ffill(axis=1).bfill(axis=1).values
LV = np.clip(LV, 0.05, 1.0)

# --- Heston, calibrated to the same surface ---
hp, _, rates = calibrate_heston(lv)

# --- the option: expiry closest to 3 months, strike at the forward ---
i = (fits["T"] - TARGET_DAYS / 365).abs().idxmin()
expiry, T, params = fits.loc[i, ["expiry", "T", "params"]]
F_T, DF = fwd.loc[fwd["expiry"] == expiry, ["F", "DF"]].iloc[0]
K = F_T
B = BARRIER_PCT * lv["spot"]
iv_K = np.sqrt(svi_total_variance(np.log(K / F_T), *params) / T)
n_days = int(round(T * 365))

print(f"Expiry {expiry} ({n_days} days), forward {F_T:.1f}")
print(f"Strike {K:.1f}, barrier {B:.1f} ({BARRIER_PCT:.0%} of spot {lv['spot']:.1f})")
print(f"Market implied vol at the strike: {iv_K:.4f}\n")


def local_vol_fn(x, t):
    row = np.clip(np.searchsorted(T_edges, t, side="right") - 1, 0, len(LV) - 1)
    return np.interp(x, k_grid, LV[row])


def flat_vol_fn(x, t):
    return iv_K


def simulate(vol_fn):
    """Local vol or flat vol: daily steps, daily barrier check."""
    rng = np.random.default_rng(SEED)
    dt = T / n_days
    x = np.zeros(N_PATHS)                     # log(price / forward)
    hit = np.zeros(N_PATHS, dtype=bool)
    for day in range(n_days):
        sigma = vol_fn(x, day * dt)
        x += -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(N_PATHS)
        F_t = np.interp((day + 1) * dt, fwd["T"], fwd["F"])
        hit |= F_t * np.exp(x) <= B
    return F_T * np.exp(x), hit


def simulate_heston(p):
    """Heston: price and variance both random, correlated by rho. Barrier checked daily."""
    rng = np.random.default_rng(SEED)
    dt = T / (n_days * HESTON_SUBSTEPS)
    x = np.zeros(N_PATHS)
    v = np.full(N_PATHS, p["v0"])
    hit = np.zeros(N_PATHS, dtype=bool)
    for day in range(n_days):
        for _ in range(HESTON_SUBSTEPS):
            z1 = rng.standard_normal(N_PATHS)
            z2 = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * rng.standard_normal(N_PATHS)
            vp = np.maximum(v, 0)             # full truncation: negative variance treated as zero
            x += -0.5 * vp * dt + np.sqrt(vp * dt) * z1
            v += p["kappa"] * (p["theta"] - vp) * dt + p["sigma"] * np.sqrt(vp * dt) * z2
        F_t = np.interp((day + 1) / 365, fwd["T"], fwd["F"])
        hit |= F_t * np.exp(x) <= B
    return F_T * np.exp(x), hit


def price(S_T, hit):
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
    "Heston": price(*simulate_heston(hp)),
    "Flat vol": price(*simulate(flat_vol_fn)),
})
print(results.round(3).to_string())

# --- exact benchmarks for the vanilla put ---
bs = bs_price(F_T * DF, K, T, -np.log(DF) / T, iv_K, "put")

today = ql.Settings.instance().evaluationDate
dc = ql.Actual365Fixed()
r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rates["r"], dc))
q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rates["q"], dc))
proc = ql.HestonProcess(r_ts, q_ts, ql.QuoteHandle(ql.SimpleQuote(rates["spot"])),
                        hp["v0"], hp["kappa"], hp["theta"], hp["sigma"], hp["rho"])
option = ql.EuropeanOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                           ql.EuropeanExercise(today + n_days))
option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))

print(f"\nMarket vanilla put (Black-Scholes at market vol): {bs:.3f}")
print(f"Heston vanilla put (exact formula):              {option.NPV():.3f}")