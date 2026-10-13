from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols
from vol_surface.svi import fit_svi, svi_total_variance

K_RANGE = 0.25   # data used for fitting: log-moneyness within ±0.25
PLOT_RANGE = 0.15  # plotted range; kept narrower so short expiries aren't extrapolated far
MIN_POINTS = 8   # need enough strikes to pin down five parameters

# --- build the cleaned dataset ---
raw, as_of = load_latest()
liquid = prepare(raw, as_of)
fwd = implied_forwards(liquid)
df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
df = add_implied_vols(keep_otm(df))
df["k"] = np.log(df["strike"] / df["F"])
df = df[df["k"].abs() <= K_RANGE]
df["w"] = df["my_iv"] ** 2 * df["T"]

# --- fit SVI to each expiry ---
rows = []
for expiry, g in df.groupby("expiry"):
    if len(g) < MIN_POINTS:
        continue
    T = g["T"].iloc[0]
    a, b, rho, m, sigma = fit_svi(g["k"].values, g["w"].values)
    fitted_vol = np.sqrt(svi_total_variance(g["k"].values, a, b, rho, m, sigma) / T)
    error = np.sqrt(np.mean((fitted_vol - g["my_iv"]) ** 2))  # average miss, in vol
    rows.append(dict(expiry=expiry, days=round(T * 365), a=a, b=b, rho=rho, m=m,
                     sigma=sigma, points=len(g), fit_error=error, T=T))
params = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
print(params.drop(columns="T").round(4).to_string(index=False))

# --- fitted surface: evaluate each expiry's SVI curve on a common grid ---
k_grid = np.linspace(-PLOT_RANGE, PLOT_RANGE, 60)
P = params[["a", "b", "rho", "m", "sigma"]].values
T_vals = params["T"].values
W_fit = np.array([svi_total_variance(k_grid, *p) for p in P])  # total variance
IV_fit = np.sqrt(W_fit / T_vals[:, None])
# blank out the parts of each curve that are beyond that expiry's market data
data_range = df.groupby("expiry")["k"].agg(["min", "max"])
for i, expiry in enumerate(params["expiry"]):
    lo, hi = data_range.loc[expiry]
    IV_fit[i, (k_grid < lo) | (k_grid > hi)] = np.nan

# --- calendar arbitrage check, only where both neighbouring expiries have market data ---
data_range = df.groupby("expiry")["k"].agg(["min", "max"])
print("\nCalendar arbitrage check (total variance must rise with expiry):")
n_bad = 0
for i in range(len(params) - 1):
    e1, e2 = params["expiry"][i], params["expiry"][i + 1]
    lo = max(data_range.loc[e1, "min"], data_range.loc[e2, "min"])
    hi = min(data_range.loc[e1, "max"], data_range.loc[e2, "max"])
    inside = (k_grid >= lo) & (k_grid <= hi)       # strikes both expiries have data for
    drop = W_fit[i] - W_fit[i + 1]                 # positive = variance falls = arbitrage
    if inside.any() and drop[inside].max() > 0:
        n_bad += 1
        pct = 100 * drop[inside].max() / W_fit[i][inside].mean()
        print(f"  {e1} -> {e2}: variance falls by up to {pct:.1f}%")
print(f"{n_bad} of {len(params) - 1} expiry pairs violate where there's real data")

# --- raw surface on the same grid, for comparison ---
plot_df = df[df["k"].abs() <= PLOT_RANGE]
KK, TT = np.meshgrid(k_grid, T_vals)
IV_raw = griddata((plot_df["k"], plot_df["T"]), plot_df["my_iv"], (KK, TT), method="linear")

# --- side-by-side chart with a shared colour scale ---
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                    subplot_titles=("Raw market vols", "SVI fitted"))
fig.add_trace(go.Surface(x=KK, y=TT * 365, z=IV_raw, coloraxis="coloraxis"), row=1, col=1)
fig.add_trace(go.Surface(x=KK, y=TT * 365, z=IV_fit, coloraxis="coloraxis"), row=1, col=2)
axes = dict(xaxis_title="ln(K/F)", yaxis_title="Days", zaxis_title="Implied vol")
fig.update_layout(
    title=f"SPX implied volatility surface ({as_of.date()}): raw vs SVI",
    template="plotly_dark",
    coloraxis=dict(colorscale="Viridis", cmin=np.nanmin(IV_fit), cmax=np.nanmax(IV_fit)),
        scene=dict(**axes, camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
    scene2=dict(**axes, camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
)

Path("output").mkdir(exist_ok=True)
params.to_csv("output/svi_params.csv", index=False)
fig.write_html("output/svi_surface.html")
fig.show()