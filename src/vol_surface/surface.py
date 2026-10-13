from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols

K_RANGE = 0.25  # show log-moneyness between -0.25 and +0.25 (roughly ±25% strikes)

# build the dataset, reusing the functions from smile.py
raw, as_of = load_latest()
liquid = prepare(raw, as_of)
fwd = implied_forwards(liquid)
df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
df = add_implied_vols(keep_otm(df))

df["k"] = np.log(df["strike"] / df["F"])
df = df[df["k"].abs() <= K_RANGE]

# fill in a smooth grid between the market points so it draws as a surface
k_grid = np.linspace(df["k"].min(), df["k"].max(), 60)
T_grid = np.linspace(df["T"].min(), df["T"].max(), 40)
KK, TT = np.meshgrid(k_grid, T_grid)
IV = griddata((df["k"], df["T"]), df["my_iv"], (KK, TT), method="linear")

fig = go.Figure()
fig.add_surface(x=KK, y=TT * 365, z=IV, colorscale="Viridis", opacity=0.9,
                colorbar=dict(title="Implied vol"))
fig.add_scatter3d(x=df["k"], y=df["T"] * 365, z=df["my_iv"], mode="markers",
                  marker=dict(size=2, color="white"), name="Market points")
fig.update_layout(
    title=f"SPX implied volatility surface ({as_of.date()})",
    template="plotly_dark",
    scene=dict(xaxis_title="Log-moneyness ln(K/F)",
               yaxis_title="Days to expiry",
               zaxis_title="Implied vol"),
)

Path("output").mkdir(exist_ok=True)
fig.write_html("output/surface.html")
fig.show()