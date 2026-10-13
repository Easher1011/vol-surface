import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
from vol_surface.local_vol import build_local_vol
from vol_surface.svi import svi_total_variance

MIN_DAYS = 30        # Heston can't match the very steep short-dated skew, so leave it out
K_POINTS = [-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05]  # log-moneyness points to fit
TARGET_DAYS = 90     # take the rate and dividend yield from the ~3-month expiry


def calibrate_heston(lv=None):
    """Calibrate Heston to the SVI surface. Returns parameters, a fit report, and rates."""
    if lv is None:
        lv = build_local_vol(k_range=0.25)
    fits, fwd, data_range = lv["fits"], lv["fwd"], lv["data_range"]
    spot, as_of = lv["spot"], lv["as_of"]

    # flat rate and dividend yield, implied from the ~3-month forward and discount factor
    i = (fits["T"] - TARGET_DAYS / 365).abs().idxmin()
    expiry, T_t = fits.loc[i, "expiry"], fits.loc[i, "T"]
    F_t, DF_t = fwd.loc[fwd["expiry"] == expiry, ["F", "DF"]].iloc[0]
    r = -np.log(DF_t) / T_t
    q = r - np.log(F_t / spot) / T_t

    # QuantLib setup
    today = ql.Date(as_of.day, as_of.month, as_of.year)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))

    # starting guess: v0, kappa, theta, sigma, rho
    process = ql.HestonProcess(r_ts, q_ts, spot_h, 0.02, 2.0, 0.03, 0.5, -0.7)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    # one calibration target per (expiry, strike), with vols read off your SVI fits
    helpers, targets = [], []
    for _, row in fits[fits["T"] >= MIN_DAYS / 365].iterrows():
        lo, hi = data_range.loc[row["expiry"]]
        days = int(round(row["T"] * 365))
        F_flat = spot * np.exp((r - q) * days / 365)
        for k in K_POINTS:
            if not lo <= k <= hi:  # only where the market actually has quotes
                continue
            vol = np.sqrt(svi_total_variance(k, *row["params"]) / row["T"])
            h = ql.HestonModelHelper(ql.Period(days, ql.Days), ql.NullCalendar(), spot,
                                     F_flat * np.exp(k), ql.QuoteHandle(ql.SimpleQuote(vol)),
                                     r_ts, q_ts)
            h.setPricingEngine(engine)
            helpers.append(h)
            targets.append(dict(expiry=row["expiry"], days=days, k=k, market_vol=vol))

    model.calibrate(helpers, ql.LevenbergMarquardt(), ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8))
    theta, kappa, sigma, rho, v0 = model.params()

    report = pd.DataFrame(targets)
    report["heston_vol"] = [h.impliedVolatility(h.modelValue(), 1e-6, 1000, 0.001, 4.0)
                            for h in helpers]
    report["miss"] = report["heston_vol"] - report["market_vol"]
    params = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    return params, report, dict(r=r, q=q, spot=spot)


if __name__ == "__main__":
    params, report, rates = calibrate_heston()

    print("Heston parameters:")
    print(f"  v0    = {params['v0']:.4f}   (today's vol ~ {np.sqrt(params['v0']):.1%})")
    print(f"  theta = {params['theta']:.4f}   (long-run vol ~ {np.sqrt(params['theta']):.1%})")
    print(f"  kappa = {params['kappa']:.3f}    (speed vol returns to long-run level)")
    print(f"  sigma = {params['sigma']:.3f}    (vol of vol)")
    print(f"  rho   = {params['rho']:.3f}    (market/vol correlation)")
    feller = 2 * params["kappa"] * params["theta"] > params["sigma"] ** 2
    print(f"  Feller condition (vol can't hit zero): {'met' if feller else 'NOT met'}")

    print("\nFit miss in vol points (Heston minus market), by expiry and log-moneyness:")
    table = report.pivot(index="expiry", columns="k", values="miss") * 100
    print(table.round(2).to_string())
    print(f"\nAverage absolute miss: {report['miss'].abs().mean() * 100:.2f} vol points")

    # chart: market vs Heston smiles for a few expiries
    expiries = report["expiry"].unique()
    pick = expiries[np.linspace(0, len(expiries) - 1, min(4, len(expiries))).astype(int)]
    plt.figure(figsize=(10, 5))
    for e in pick:
        g = report[report["expiry"] == e]
        line, = plt.plot(g["k"], g["market_vol"], "o-", label=f"{e} market")
        plt.plot(g["k"], g["heston_vol"], "x--", c=line.get_color(), label=f"{e} Heston")
    plt.title("Heston fit vs market (SVI) implied vols")
    plt.xlabel("Log-moneyness ln(K/F)")
    plt.ylabel("Implied vol")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()