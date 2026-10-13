from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols
from vol_surface.svi import fit_svi, svi_total_variance

MIN_GAP_DAYS = 10   # expiries closer than this make the time-derivative too noisy
MIN_POINTS = 8


def svi_dk(k, a, b, rho, m, sigma):
    """Slope of total variance with respect to log-moneyness."""
    return b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_dkk(k, a, b, rho, m, sigma):
    """Curvature of total variance with respect to log-moneyness."""
    return b * sigma ** 2 / ((k - m) ** 2 + sigma ** 2) ** 1.5


def butterfly_g(k, w, w1, w2):
    """Gatheral's g(k): must be positive (no butterfly arbitrage). Also Dupire's denominator."""
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def build_local_vol(k_range=0.10):
    """Fit SVI per expiry and compute Dupire local vol. Returns everything later steps need."""
    raw, as_of = load_latest()
    liquid = prepare(raw, as_of)
    fwd = implied_forwards(liquid)
    df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
    df = add_implied_vols(keep_otm(df))
    df["k"] = np.log(df["strike"] / df["F"])
    df = df[df["k"].abs() <= 0.25]
    df["w"] = df["my_iv"] ** 2 * df["T"]

    rows = []
    for expiry, g in df.groupby("expiry"):
        if len(g) >= MIN_POINTS:
            rows.append(dict(expiry=expiry, T=g["T"].iloc[0],
                             params=fit_svi(g["k"].values, g["w"].values)))
    fits = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    keep, last_T = [], -1.0
    for i, T in enumerate(fits["T"]):
        if T - last_T >= MIN_GAP_DAYS / 365:
            keep.append(i)
            last_T = T
    fits = fits.loc[keep].reset_index(drop=True)

    data_range = df.groupby("expiry")["k"].agg(["min", "max"])
    k_grid = np.linspace(-k_range, k_range, 101)

    LV, IV = [], []
    for i in range(len(fits) - 1):
        (e1, T1, p1), (e2, T2, p2) = fits.loc[i].values, fits.loc[i + 1].values
        w_a, w_b = svi_total_variance(k_grid, *p1), svi_total_variance(k_grid, *p2)
        w = (w_a + w_b) / 2
        w1 = (svi_dk(k_grid, *p1) + svi_dk(k_grid, *p2)) / 2
        w2 = (svi_dkk(k_grid, *p1) + svi_dkk(k_grid, *p2)) / 2
        dw_dT = (w_b - w_a) / (T2 - T1)

        g = butterfly_g(k_grid, w, w1, w2)
        local_var = dw_dT / g                                   # Dupire's formula
        lv = np.where((local_var > 0) & (g > 0), np.sqrt(np.clip(local_var, 0, None)), np.nan)
        iv = np.sqrt(w / ((T1 + T2) / 2))

        lo = max(data_range.loc[e1, "min"], data_range.loc[e2, "min"])
        hi = min(data_range.loc[e1, "max"], data_range.loc[e2, "max"])
        outside = (k_grid < lo) | (k_grid > hi)
        lv[outside], iv[outside] = np.nan, np.nan
        LV.append(lv)
        IV.append(iv)

    return dict(as_of=as_of, spot=raw["spot"].iloc[0], fwd=fwd, fits=fits,
                data_range=data_range, k_grid=k_grid, T_edges=fits["T"].values,
                LV=np.array(LV), IV=np.array(IV))


if __name__ == "__main__":
    lv = build_local_vol(k_range=0.10)
    fits, k_grid, data_range = lv["fits"], lv["k_grid"], lv["data_range"]
    print("Expiries used:", ", ".join(fits["expiry"]))

    print("\nButterfly check (g must be positive where there's data):")
    any_bad = False
    for _, row in fits.iterrows():
        p = row["params"]
        w = svi_total_variance(k_grid, *p)
        g = butterfly_g(k_grid, w, svi_dk(k_grid, *p), svi_dkk(k_grid, *p))
        lo, hi = data_range.loc[row["expiry"]]
        bad = ((g <= 0) & (k_grid >= lo) & (k_grid <= hi)).sum()
        if bad:
            any_bad = True
            print(f"  {row['expiry']}: {bad} violating points")
    if not any_bad:
        print("  no violations")

    T_mid = (lv["T_edges"][:-1] + lv["T_edges"][1:]) / 2
    KK, TT = np.meshgrid(k_grid, T_mid)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                        subplot_titles=("Implied vol (SVI)", "Local vol (Dupire)"))
    fig.add_trace(go.Surface(x=KK, y=TT * 365, z=lv["IV"], coloraxis="coloraxis"), row=1, col=1)
    fig.add_trace(go.Surface(x=KK, y=TT * 365, z=lv["LV"], coloraxis="coloraxis"), row=1, col=2)
    camera = dict(eye=dict(x=1.6, y=-1.6, z=0.8))
    axes = dict(xaxis_title="ln(K/F)", yaxis_title="Days", zaxis_title="Vol", camera=camera)
    fig.update_layout(
        title=f"SPX implied vs local volatility ({lv['as_of'].date()})",
        template="plotly_dark",
        coloraxis=dict(colorscale="Viridis",
                       cmin=np.nanmin([lv["IV"], lv["LV"]]), cmax=np.nanmax([lv["IV"], lv["LV"]])),
        scene=axes, scene2=axes,
    )
    Path("output").mkdir(exist_ok=True)
    fig.write_html("output/local_vol.html")
    fig.show()