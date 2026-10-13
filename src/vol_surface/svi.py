import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI: total implied variance (vol^2 * T) as a function of log-moneyness k."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(k, w):
    """Find the five SVI parameters that best fit market total variance w at log-moneyness k."""

    def residuals(params):
        a, b, rho, m, sigma = params
        return svi_total_variance(k, a, b, rho, m, sigma) - w

    x0 = [max(0.9 * w.min(), 1e-6), 0.1, -0.5, 0.0, 0.1]  # starting guess
    lower = [-w.max(), 0.0, -0.999, -1.0, 1e-4]           # a may be negative...
    upper = [w.max(), 5.0, 0.999, 1.0, 2.0]
    result = least_squares(residuals, x0, bounds=(lower, upper))
    a, b, rho, m, sigma = result.x
    min_variance = a + b * sigma * np.sqrt(1 - rho ** 2)  # lowest point of the curve
    if min_variance < 0:
        print("Warning: fitted variance goes negative, fit is invalid")
    return result.x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from vol_surface.smile import load_latest, prepare, implied_forwards, keep_otm, add_implied_vols

    raw, as_of = load_latest()
    liquid = prepare(raw, as_of)
    fwd = implied_forwards(liquid)
    df = liquid.drop(columns="spot").merge(fwd[["expiry", "F", "DF"]], on="expiry")
    df = add_implied_vols(keep_otm(df))

    expiry = df.loc[(df["T"] - 30 / 365).abs().idxmin(), "expiry"]
    sub = df[df["expiry"] == expiry]
    T = sub["T"].iloc[0]
    F = sub["F"].iloc[0]
    k = np.log(sub["strike"] / F).values
    w = (sub["my_iv"] ** 2 * T).values

    params = fit_svi(k, w)
    print("SVI parameters (a, b, rho, m, sigma):", np.round(params, 4))

    k_line = np.linspace(k.min(), k.max(), 200)
    iv_fit = np.sqrt(svi_total_variance(k_line, *params) / T)

    plt.figure(figsize=(10, 5))
    plt.plot(sub["strike"], sub["my_iv"], "o", alpha=0.6, label="Market implied vol")
    plt.plot(F * np.exp(k_line), iv_fit, "-", lw=2, label="SVI fit")
    plt.axvline(F, ls="--", c="grey", label="Forward")
    plt.title(f"SVI fit, SPX expiry {expiry}")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()