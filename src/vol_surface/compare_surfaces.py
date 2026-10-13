from pathlib import Path
import numpy as np
import QuantLib as ql
import plotly.graph_objects as go
from vol_surface.local_vol import build_local_vol
from vol_surface.heston import calibrate_heston, MIN_DAYS
from vol_surface.svi import svi_total_variance

K_GRID = np.linspace(-0.15, 0.05, 21)   # log-moneyness range, same as the Heston calibration
TARGET_DAYS = 90                        # flat vol = market vol of the 3-month at-the-money option

# --- build everything ---
lv = build_local_vol(k_range=0.25)
hp, _, rates = calibrate_heston(lv)
fits = lv["fits"][lv["fits"]["T"] >= MIN_DAYS / 365].reset_index(drop=True)
T_vals = fits["T"].values

# --- 1. market surface (SVI). Local vol reproduces this exactly, by construction ---
market = np.array([np.sqrt(svi_total_variance(K_GRID, *p) / T)
                   for p, T in zip(fits["params"], T_vals)])

# --- 2. Heston: price every option on the grid with Heston, convert back to implied vol ---
today = ql.Settings.instance().evaluationDate
dc = ql.Actual365Fixed()
r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rates["r"], dc))
q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rates["q"], dc))
spot = rates["spot"]
proc = ql.HestonProcess(r_ts, q_ts, ql.QuoteHandle(ql.SimpleQuote(spot)),
                        hp["v0"], hp["kappa"], hp["theta"], hp["sigma"], hp["rho"])
engine = ql.AnalyticHestonEngine(ql.HestonModel(proc))

heston = np.zeros_like(market)
for i, T in enumerate(T_vals):
    days = int(round(T * 365))
    F_flat = spot * np.exp((rates["r"] - rates["q"]) * days / 365)
    for j, k in enumerate(K_GRID):
        h = ql.HestonModelHelper(ql.Period(days, ql.Days), ql.NullCalendar(), spot,
                                 F_flat * np.exp(k), ql.QuoteHandle(ql.SimpleQuote(0.2)),
                                 r_ts, q_ts)
        h.setPricingEngine(engine)
        heston[i, j] = h.impliedVolatility(h.modelValue(), 1e-6, 1000, 0.001, 4.0)

# --- 3. flat vol: one number everywhere ---
i3m = (fits["T"] - TARGET_DAYS / 365).abs().idxmin()
flat_vol = np.sqrt(svi_total_variance(0.0, *fits.loc[i3m, "params"]) / fits.loc[i3m, "T"])
flat = np.full_like(market, flat_vol)

# --- hide anything outside the market's quoted range ---
for i, e in enumerate(fits["expiry"]):
    lo, hi = lv["data_range"].loc[e]
    outside = (K_GRID < lo) | (K_GRID > hi)
    market[i, outside] = np.nan
    heston[i, outside] = np.nan

# --- one 3D chart, three surfaces; click legend entries to show/hide each ---
KK, TT = np.meshgrid(K_GRID, T_vals * 365)


def surface(z, name, colour, opacity):
    return go.Surface(x=KK, y=TT, z=z, name=name, showlegend=True, showscale=False,
                      colorscale=[[0, colour], [1, colour]], opacity=opacity)


fig = go.Figure([
    surface(market, "Market (SVI) = Local vol", "#2ca02c", 0.9),
    surface(heston, "Heston", "#ff7f0e", 0.7),
    surface(flat, f"Flat vol ({flat_vol:.1%})", "#1f77b4", 0.5),
])
fig.update_layout(
    title=f"SPX implied vol surface by model ({lv['as_of'].date()})",
    template="plotly_dark",
    scene=dict(xaxis_title="ln(K/F)", yaxis_title="Days", zaxis_title="Implied vol",
               camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
    legend=dict(x=0.02, y=0.95),
)

Path("output").mkdir(exist_ok=True)
fig.write_html("output/model_surfaces.html")
fig.show()

# save results for the website
import json
Path("results").mkdir(exist_ok=True)
fig.write_json("results/model_surfaces.json")
with open("results/heston_params.json", "w") as f:
    json.dump(hp, f, indent=2)