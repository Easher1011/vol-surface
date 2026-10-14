import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

RESULTS = Path(__file__).parent / "results"
st.set_page_config(page_title="SPX Volatility Surface", layout="wide")


@st.cache_resource
def load_fig(name, height=650):
    fig = pio.read_json(RESULTS / name)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=60, b=0))
    return fig


def svi_vol(k, T, a, b, rho, m, sigma):
    """SVI total variance, converted to implied vol."""
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    return np.sqrt(w / T)


meta = json.loads((RESULTS / "barrier_meta.json").read_text())
hp = json.loads((RESULTS / "heston_params.json").read_text())
svi = pd.read_csv(RESULTS / "svi_params.csv")
prices = pd.read_csv(RESULTS / "barrier_prices.csv", index_col=0)
di = prices.loc["Down-and-in put"]

# ---------------- header ----------------
st.title("SPX volatility surface: from raw option quotes to exotic pricing")
st.caption(f"S&P 500 index options · data as of {meta['as_of']} · built by Emery Asher")

c1, c2, c3 = st.columns(3)
c1.metric("Down-and-in put: flat vol", f"{di['Flat vol']:.1f}")
c2.metric("Down-and-in put: local vol", f"{di['Local vol']:.1f}",
          f"{di['Local vol'] / di['Flat vol'] - 1:+.0%} vs flat vol", delta_color="off")
c3.metric("Down-and-in put: Heston", f"{di['Heston']:.1f}",
          f"{di['Heston'] / di['Flat vol'] - 1:+.0%} vs flat vol", delta_color="off")

st.markdown(
    "This project builds an implied volatility surface for the S&P 500 from raw option quotes, "
    "fits it with SVI, derives a Dupire local volatility model, calibrates Heston, and prices "
    "a barrier option under each model. The headline: ignoring the volatility skew badly "
    "misprices downside barrier options, while two different skew-consistent models broadly agree."
)

tabs = st.tabs(["Surface", "Smile explorer", "Local vol", "Model comparison",
                "Barrier pricing", "VIX rebuild", "Variance risk premium",
                "Index comparison", "Method & limitations"])

# ---------------- 1. surface ----------------
with tabs[0]:
    st.subheader("Raw market vols vs SVI-fitted surface")
    st.plotly_chart(load_fig("svi_surface.json"), width="stretch")
    st.markdown(
        "Left: implied vols computed from cleaned market quotes, joined point to point, so bad "
        "quotes show up as spikes. Right: each expiry fitted with SVI, a five-parameter smile. "
        "Stale quotes are removed by an outlier filter, and the fitted surface is checked for "
        "calendar arbitrage (total variance must rise with maturity) within the quoted range. "
        "Drag to rotate, scroll to zoom."
    )

# ---------------- 2. smile explorer ----------------
with tabs[1]:
    st.subheader("Fitted smiles by expiry")
    options = list(svi["expiry"])
    default = [options[i] for i in np.linspace(0, len(options) - 1, 4).astype(int)]
    chosen = st.multiselect("Expiries", options, default=default)

    fig = go.Figure()
    for e in chosen:
        r = svi[svi["expiry"] == e].iloc[0]
        k = np.linspace(r["k_min"], r["k_max"], 200)  # only where the market has quotes
        fig.add_scatter(x=k, y=svi_vol(k, r["T"], r["a"], r["b"], r["rho"], r["m"], r["sigma"]),
                        mode="lines", name=f"{e} ({r['days']}d)")
    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title="Log-moneyness ln(K/F)", yaxis_title="Implied vol")
    st.plotly_chart(fig, width="stretch")
    st.markdown("Shorter expiries have a steeper skew. Each curve covers only the strike "
                "range the market actually quotes.")
    st.dataframe(svi[svi["expiry"].isin(chosen)]
                 .drop(columns=["T", "k_min", "k_max"]).round(4), hide_index=True)

# ---------------- 3. local vol ----------------
with tabs[2]:
    st.subheader("Implied vol vs Dupire local vol")
    st.plotly_chart(load_fig("local_vol.json"), width="stretch")
    st.markdown(
        "Implied vol is the average volatility over an option's life. Local vol is the "
        "volatility at a given market level and time: the unique map that reproduces every "
        "vanilla price. Near the forward its skew is roughly twice as steep as implied vol, "
        "as theory predicts. It is computed from the SVI surface via Dupire's formula, after "
        "checking for butterfly arbitrage."
    )

# ---------------- 4. model comparison ----------------
with tabs[3]:
    st.subheader("Implied vol surface produced by each model")
    st.plotly_chart(load_fig("model_surfaces.json"), width="stretch")
    st.markdown(
        "Local vol reproduces the market surface exactly by construction. Heston, with five "
        "parameters for the whole surface, fits closely but misses at the short end and in "
        "the deep downside. Flat vol (Black-Scholes) has no skew at all. Click legend entries "
        "to show or hide surfaces."
    )
    st.dataframe(pd.DataFrame({
        "Parameter": ["v0", "theta", "kappa", "sigma", "rho"],
        "Value": [hp["v0"], hp["theta"], hp["kappa"], hp["sigma"], hp["rho"]],
        "Meaning": [f"today's variance (vol ≈ {np.sqrt(hp['v0']):.1%})",
                    f"long-run variance (vol ≈ {np.sqrt(hp['theta']):.1%})",
                    "speed vol reverts to long-run level",
                    "vol of vol",
                    "correlation between market and vol"],
    }).round(4), hide_index=True)
    st.markdown(
        f"The calibration needs very high vol of vol ({hp['sigma']:.2f}) and fast mean "
        f"reversion ({hp['kappa']:.1f}) to span a steep short-dated skew and a flatter "
        "long-dated one. The parameters fit prices but are not economically plausible: a "
        "known limitation of Heston."
    )

# ---------------- 5. barrier pricing ----------------
with tabs[4]:
    st.subheader("Pricing a down-and-in put under three models")
    st.markdown(
        f"**Option:** put on SPX, strike {meta['strike']:,.0f} (at the forward), expiry "
        f"{meta['expiry']} ({meta['days']} days). It only pays out if SPX touches the "
        f"barrier at {meta['barrier']:,.0f} (90% of spot {meta['spot']:,.0f}), checked daily. "
        f"Monte Carlo with {meta['n_paths']:,} paths."
    )
    st.dataframe(prices.round(3))
    st.markdown(
        f"**Validation:** all three models price the vanilla put close to the market "
        f"({meta['bs_vanilla']:.1f}); the Heston simulation matches Heston's exact formula "
        f"({meta['heston_exact']:.1f}); and down-and-in plus down-and-out equals vanilla in "
        "every column."
    )
    st.markdown(
        f"**Finding:** flat vol prices the down-and-in put at {di['Flat vol']:.0f}; the two "
        f"skew-consistent models give {di['Local vol']:.0f} (local vol) and "
        f"{di['Heston']:.0f} (Heston). Flat vol ignores the rise in volatility as the market "
        "falls, so it understates the chance of hitting the barrier. Local vol and Heston "
        "agree closely here because this product depends mainly on the terminal distribution, "
        "which both models are calibrated to."
    )

# ---------------- VIX rebuild ----------------
with tabs[5]:
    vix = json.loads((RESULTS / "vix.json").read_text())
    st.subheader("Rebuilding the VIX index from SPX options")
    a, b, c = st.columns(3)
    a.metric("My VIX", f"{vix['my_vix']:.2f}")
    b.metric("Official VIX", f"{vix['official']:.2f}")
    c.metric("Difference", f"{vix['my_vix'] - vix['official']:+.2f}")
    st.markdown(
        "VIX is Cboe's measure of expected S&P 500 volatility over the next 30 days, calculated "
        "from SPX option prices. This rebuilds it from the same raw quotes used for the surface, "
        "following Cboe's published method: take the two expiries either side of 30 days, sum "
        "the prices of out-of-the-money options weighted by 1/strike², and interpolate to "
        "exactly 30 days."
    )
    st.markdown(
        f"**Expiries used:** {vix['near']} ({vix['near_days']:.1f} days, {vix['near_vol']:.2f}%) "
        f"and {vix['next']} ({vix['next_days']:.1f} days, {vix['next_vol']:.2f}%). "
        f"**Snapshot:** {vix['snapshot']} New York time; the official reading available was "
        f"from {vix['official_time']}, so part of any difference reflects market moves in between."
    )

# ---------------- variance risk premium ----------------
with tabs[6]:
    s = json.loads((RESULTS / "vrp_summary.json").read_text())
    st.subheader(f"The variance risk premium, {s['start']} to {s['end']}")
    a, b, c, d = st.columns(4)
    a.metric("Average VIX", f"{s['avg_vix']:.1f}")
    b.metric("Average realised vol (next 30 days)", f"{s['avg_rv']:.1f}")
    c.metric("VIX above what followed", f"{s['pct_days_positive']:.0%} of days")
    d.metric("Worst drawdown, short variance", f"{s['worst_drawdown']:.0f} vol pts")
    st.plotly_chart(load_fig("vrp.json", height=950), width="stretch")
    st.markdown(
        f"VIX is what options charge for the next 30 days of volatility; realised vol is what "
        f"actually followed. On average VIX exceeded it by {s['avg_gap']:.1f} vol points. "
        f"Selling 30-day variance monthly (non-overlapping) was profitable in "
        f"{s['pct_months_profitable']:.0%} of {s['months']} months, with a median month of "
        f"{s['median_month']:+.1f} but an average of only {s['avg_month']:+.1f}: frequent small "
        f"gains, rare large losses. The worst losses began from low VIX levels, when the premium "
        f"collected was smallest. This measures the premium; it is not a tradable backtest "
        f"(no costs, VIX is not directly tradable)."
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Worst months for a volatility seller**")
    col1.dataframe(pd.read_csv(RESULTS / "vrp_worst_months.csv")
                   .rename(columns={"vix": "VIX", "rv": "Realised", "short_var_pnl": "P&L"})
                   .round(1), hide_index=True)
    col2.markdown("**Average by year**")
    col2.dataframe(pd.read_csv(RESULTS / "vrp_yearly.csv")
                   .rename(columns={"Date": "Year", "vix": "VIX", "rv": "Realised", "vrp": "Gap"})
                   .round(1), hide_index=True, height=400)

    # ---------------- index comparison ----------------
with tabs[7]:
    st.subheader("S&P 500 vs Nasdaq-100 vs Russell 2000")
    st.plotly_chart(load_fig("index_compare.json", height=550), width="stretch")
    m = pd.read_csv(RESULTS / "index_metrics.csv")
    st.dataframe(m.rename(columns={"index": "Index", "tenor": "Target", "expiry": "Expiry used",
                                   "days": "Days", "atm_vol": "ATM vol", "vol_90": "Vol at ~90% strike",
                                   "skew": "Skew", "skew_ratio": "Skew ratio"}).round(3),
                 hide_index=True)
    st.markdown(
        "The S&P 500 has the lowest at-the-money volatility of the three but the steepest skew "
        "relative to its level. Deep downside S&P 500 options can be priced at higher volatility "
        "than Russell 2000 options, despite the Russell being the more volatile index. This is "
        "consistent with concentrated hedging demand for S&P 500 puts."
    )
    st.markdown(
        "**Method:** the same pipeline as the SPX surface, run on each index. All three use one "
        "discount rate implied from SPX put-call parity, because the less liquid indices give "
        "unstable rates; each index's forward is the median across strikes of K + (C − P)/DF. "
        "Expiries with implausible forwards or too few quotes on either side of the forward are "
        "dropped. Skew is measured on the expiry nearest each target that is quoted down to a "
        "~90% strike, so the expiry used can differ from the target (see the Days column). "
        "Single-day snapshot."
    )
    
# ---------------- 6. method ----------------
with tabs[8]:
    st.subheader("Method")
    st.markdown("""
1. **Data:** SPX option chain from Yahoo Finance (end-of-day snapshot).
2. **Cleaning:** drop expiries under 7 days, zero bids, and bid-ask spreads over 10% of mid; keep out-of-the-money options.
3. **Forwards:** implied per expiry from put-call parity, so no assumed rate or dividend yield.
4. **Implied vols:** own Black-Scholes implementation and root-finder.
5. **SVI:** fitted per expiry, with outlier removal (fit, drop quotes missing by over 5x the median, refit).
6. **Arbitrage checks:** calendar and butterfly, within the quoted strike range.
7. **Local vol:** Dupire's formula in total-variance form, using expiries at least 10 days apart.
8. **Heston:** calibrated with QuantLib to the SVI surface (expiries of 30 days or more).
9. **Pricing:** Monte Carlo with daily barrier monitoring; Heston simulated with full truncation.
""")
    st.subheader("Limitations")
    st.markdown("""
- End-of-day Yahoo data, not live executable quotes; some stale quotes remain.
- Expiries are fitted one at a time; smoothness across maturities is not enforced, so local vol shows steps between expiries.
- Heston uses a single flat rate and dividend yield, and fits the short-dated skew poorly.
- Monte Carlo error: barrier prices carry roughly ±2 points of noise at 50,000 paths.
- Local vol reproduces today's prices but is known to misrepresent how the smile moves over time.
""")