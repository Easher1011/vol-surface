from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols
from vol_surface.svi import fit_svi, svi_total_variance

INDICES = {"spx": "S&P 500", "ndx": "Nasdaq-100", "rut": "Russell 2000"}
COLOURS = {"spx": "#636efa", "ndx": "#00cc96", "rut": "#ffa15a"}
MIN_POINTS = 20        # less liquid indices need a stricter minimum
DOWN = -0.10           # log-moneyness of roughly a 90% strike
SHOW_FORWARDS = False  # set to True to print each index's forward table


def one_root_per_expiry(raw):
    """Some dates have both AM monthly and PM weekly options; keep whichever has more quotes."""
    raw = raw.assign(root=raw["contractSymbol"].str.extract(r"^([A-Z]+)", expand=False))
    counts = raw.groupby(["expiry", "root"]).size().reset_index(name="n")
    keep = counts.sort_values("n").drop_duplicates("expiry", keep="last")
    return raw.merge(keep[["expiry", "root"]], on=["expiry", "root"])


def index_fits(prefix):
    """Same pipeline as the SPX surface, with a fixed discount rate and forward sanity checks."""
    raw, as_of = load_latest(prefix)
    liquid = prepare(one_root_per_expiry(raw), as_of)

    # forwards: one market discount rate for all indices, forward from each index's own quotes
    fwd = implied_forwards(liquid, rate=RATE)
    if SHOW_FORWARDS:
        print(f"\n{prefix} forwards:\n",
              fwd[["expiry", "T", "F_over_spot"]].round(4).to_string(index=False))

    # drop expiries whose forward is out of line with the index's typical carry (stale quotes)
    long = fwd["T"] > 0.1
    carry = np.median(np.log(fwd.loc[long, "F_over_spot"]) / fwd.loc[long, "T"])
    sane = (fwd["F_over_spot"] - np.exp(carry * fwd["T"])).abs() < 0.01
    if (~sane).any():
        print(f"{prefix}: dropped expiries with implausible forwards: "
              f"{list(fwd.loc[~sane, 'expiry'])}")
    fwd = fwd[sane]

    df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
    df = add_implied_vols(keep_otm(df))
    df["k"] = np.log(df["strike"] / df["F"])
    df = df[df["k"].abs() <= 0.25]
    df["w"] = df["my_iv"] ** 2 * df["T"]

    rows = []
    for expiry, g in df.groupby("expiry"):
        # need enough quotes, and quotes on both sides of the forward
        if len(g) < MIN_POINTS or g["k"].min() > -0.05 or g["k"].max() < 0.02:
            continue
        p = fit_svi(g["k"].values, g["w"].values)
        a, b, rho, m, sigma = p
        if a + b * sigma * np.sqrt(1 - rho ** 2) < 0:  # variance would go negative: unusable
            continue
        rows.append(dict(expiry=expiry, T=g["T"].iloc[0], k_min=g["k"].min(),
                         k_max=g["k"].max(), params=p))
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True), as_of


def vol_at(row, k):
    """SVI implied vol at log-moneyness k; blank if outside the quoted range."""
    if not row["k_min"] <= k <= row["k_max"]:
        return np.nan
    return float(np.sqrt(svi_total_variance(k, *row["params"]) / row["T"]))


def nearest(fits, target_days):
    """Expiry closest to the target, preferring ones quoted down to ~90% strike."""
    reach = fits[fits["k_min"] <= DOWN]
    pool = reach if len(reach) else fits
    return pool.loc[(pool["T"] - target_days / 365).abs().idxmin()]


# one discount rate for all indices, from the most liquid market (SPX, expiries over ~5 weeks)
spx_raw, spx_as_of = load_latest("spx")
spx_fwd = implied_forwards(prepare(one_root_per_expiry(spx_raw), spx_as_of))
RATE = float(spx_fwd.loc[spx_fwd["T"] > 0.1, "implied_rate"].median())
print(f"Discount rate from SPX put-call parity: {RATE:.2%}\n")

fig = make_subplots(rows=1, cols=2, subplot_titles=("~30-day smiles", "At-the-money vol by expiry"))
metrics = []
for prefix, name in INDICES.items():
    fits, as_of = index_fits(prefix)
    colour = COLOURS[prefix]

    # summary numbers at ~30 and ~90 days
    for target in (30, 90):
        row = nearest(fits, target)
        atm, down = vol_at(row, 0.0), vol_at(row, DOWN)
        metrics.append(dict(index=name, tenor=f"{target}d", expiry=row["expiry"],
                            days=round(row["T"] * 365), atm_vol=atm, vol_90=down,
                            skew=down - atm, skew_ratio=down / atm))

    # chart 1: the ~30-day smile, only where the market has quotes
    row = nearest(fits, 30)
    k = np.linspace(row["k_min"], row["k_max"], 200)
    fig.add_scatter(x=k, y=np.sqrt(svi_total_variance(k, *row["params"]) / row["T"]),
                    name=f"{name} ({row['expiry']})", legendgroup=name,
                    line=dict(color=colour), row=1, col=1)

    # chart 2: at-the-money vol across expiries
    atm_curve = [vol_at(r, 0.0) for _, r in fits.iterrows()]
    fig.add_scatter(x=fits["T"] * 365, y=atm_curve, mode="lines+markers", name=name,
                    legendgroup=name, showlegend=False, line=dict(color=colour), row=1, col=2)

table = pd.DataFrame(metrics)
print(f"\nData as of {as_of.date()}\n")
print(table.drop(columns="expiry").round(3).to_string(index=False))

fig.update_xaxes(title_text="Log-moneyness ln(K/F)", row=1, col=1)
fig.update_xaxes(title_text="Days to expiry", row=1, col=2)
fig.update_yaxes(title_text="Implied vol", row=1, col=1)
fig.update_layout(template="plotly_dark", height=550,
                  title=f"S&P 500 vs Nasdaq-100 vs Russell 2000 ({as_of.date()})")

Path("output").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
fig.write_html("output/index_compare.html")
fig.write_json("results/index_compare.json")
table.to_csv("results/index_metrics.csv", index=False)
fig.show()