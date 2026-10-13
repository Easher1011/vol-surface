import numpy as np
import matplotlib.pyplot as plt
from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols
from vol_surface.svi import fit_svi, svi_total_variance

EXPIRIES = ["2026-11-30", "2026-12-31", "2027-03-31"]  # one normal, two problem expiries

raw, as_of = load_latest()
liquid = prepare(raw, as_of)
fwd = implied_forwards(liquid)
df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
df = add_implied_vols(keep_otm(df))
df["k"] = np.log(df["strike"] / df["F"])
df = df[df["k"].abs() <= 0.25]
df["root"] = df["contractSymbol"].str.extract(r"^([A-Z]+)", expand=False)  # SPX or SPXW

fig, axes = plt.subplots(1, len(EXPIRIES), figsize=(16, 5))
for ax, expiry in zip(axes, EXPIRIES):
    sub = df[df["expiry"] == expiry]
    print(expiry, "option types:", sub["root"].value_counts().to_dict())

    T, F = sub["T"].iloc[0], sub["F"].iloc[0]
    params = fit_svi(sub["k"].values, (sub["my_iv"] ** 2 * T).values)

    for root, g in sub.groupby("root"):
        ax.plot(g["strike"], g["my_iv"], "o", alpha=0.6, label=f"{root} ({len(g)} quotes)")
    k_line = np.linspace(sub["k"].min(), sub["k"].max(), 200)
    ax.plot(F * np.exp(k_line), np.sqrt(svi_total_variance(k_line, *params) / T),
            "-", lw=2, c="black", label="SVI fit")
    ax.axvline(F, ls="--", c="grey")
    ax.set_title(f"Expiry {expiry}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
