# SPX Volatility Surface and Exotic Option Pricing

Builds an implied volatility surface for S&P 500 index options from raw market quotes, derives local volatility and Heston models from it, prices a barrier option under each model, rebuilds the VIX index, measures the variance risk premium since 1990, and compares option pricing across the S&P 500, Nasdaq-100 and Russell 2000.

**Live site:** https://spx-vol-surface.streamlit.app/

## Key results

**Barrier pricing.** A 3-month down-and-in put on SPX (strike at the forward, barrier at 90% of spot, monitored daily), priced by Monte Carlo with 50,000 paths:

| Model | 13 Oct 2026 | 14 Oct 2026 | Probability of hitting barrier (14 Oct) |
|---|---|---|---|
| Flat vol (Black-Scholes) | 77.0 | 76.9 | 9.0% |
| Local vol (Dupire) | 152.9 | 151.0 | 16.6% |
| Heston | 157.9 | 156.6 | 16.8% |

Prices in index points. Flat-vol pricing gives roughly half the value of the two skew-consistent models, which agree within about 4%. The result held on a second trading day.

**VIX rebuild.** Rebuilt from raw SPX option quotes using Cboe's published methodology: 17.06 against an official reading of 17.02. The official reading available was about an hour older than the option snapshot, so the comparison is indicative; daily same-time comparisons are now being collected.

**Variance risk premium, 1990 to 2026.** VIX exceeded the S&P 500's subsequent 30-day realised volatility on 86% of days, by 4.1 vol points on average. Selling 30-day variance monthly was profitable in 84% of 441 non-overlapping months, but the worst drawdown (-120 vol points, mostly February and March 2020) equalled about 44 average months of gains. The largest losses began from low VIX levels.

**Index comparison.** The S&P 500 has the lowest at-the-money volatility of the three indices but the steepest skew relative to its level: at 30 days, vol at a ~90% strike was 1.76 times at-the-money vol for the S&P 500, against 1.36 for the Russell 2000 on the same expiry. Below roughly 13% out of the money, S&P 500 puts were priced at higher volatility than Russell 2000 puts, consistent with concentrated hedging demand for S&P 500 protection.

## Method

**Volatility surface**
1. **Data:** option chains from Yahoo Finance via `yfinance`.
2. **Cleaning:** remove expiries under 7 days, zero bids, and bid-ask spreads wider than 10% of mid; keep out-of-the-money options only.
3. **Forwards:** for SPX, regress call minus put on strike (put-call parity) to obtain the forward and discount factor. For the less liquid indices, use the SPX-implied discount rate and take each forward as the median of K + (C − P)/DF across strikes, dropping expiries whose forward is implausible.
4. **Implied volatility:** Black-Scholes pricer inverted with Brent's method.
5. **SVI:** each expiry fitted with raw SVI in total variance; stale quotes removed by fitting, dropping quotes that miss by more than five times the median miss, and refitting.
6. **Arbitrage checks:** calendar (total variance increasing with maturity) and butterfly (Gatheral's g(k) positive), within the quoted strike range. No violations on either day.

**Models and pricing**
7. **Local volatility:** Dupire's formula in total-variance form, using SVI's analytic derivatives and expiries at least 10 days apart.
8. **Heston:** calibrated with QuantLib to the SVI surface (expiries of 30 days or more); average fit error about 0.3 vol points.
9. **Monte Carlo:** daily barrier monitoring; Heston simulated with full truncation, validated against QuantLib's analytic vanilla price.

**VIX and variance risk premium**
10. **VIX:** Cboe methodology: PM-settled weekly options, the two expiries either side of 30 days, out-of-the-money options until two consecutive zero bids, weighted by ΔK/K², interpolated to 30 days in minutes.
11. **Variance risk premium:** daily VIX against realised volatility of the S&P 500 over the following 21 trading days, annualised. Short-variance P&L per unit of vega notional, (K² − RV²)/2K, on non-overlapping monthly observations.

**Data collection**
12. A scheduled script saves SPX, Nasdaq-100 and Russell 2000 option chains, plus the official VIX, every weekday during US market hours.

## Observations

- **SPX skew:** about 30% implied vol 15% below the forward, against about 10% above it, flattening with maturity.
- **Term structure:** at-the-money vol rises from about 12% short-dated to about 15% at seven months.
- **Heston:** the calibration requires very high vol of vol (about 1.8) and fast mean reversion (about 9) to span the steep short-dated and flatter long-dated skew. These parameters fit prices but are not economically plausible.
- **Stale quotes:** outliers on SPX quarter-end expiries were identical across two days while the index moved, which identified them as stale quotes.
- **Thin markets:** for the Nasdaq-100 and Russell 2000, put-call parity regressions produced implausible discount rates (beyond ±90%) on some expiries, which distorted implied vols. Fixing the discount rate from SPX resolved this.

## Limitations

- Yahoo Finance quotes are not executable prices; some stale quotes remain after cleaning.
- Expiries are fitted independently, so local vol shows steps between maturities.
- Heston uses a single flat rate and dividend yield, and fits the short-dated skew poorly.
- Monte Carlo error on barrier prices is roughly ±2 index points.
- Local vol reproduces today's prices but misrepresents the future dynamics of the smile.
- The variance risk premium P&L is a measurement, not a tradable backtest: VIX is not directly tradable, and costs and leverage are ignored.
- The index comparison is a single-day snapshot.

## Project structure

```
src/vol_surface/
  data.py              download an option chain
  collect.py           daily snapshot of SPX, NDX, RUT chains and official VIX
  black_scholes.py     Black-Scholes pricer and implied vol solver
  smile.py             cleaning, put-call parity forwards, implied vols
  surface.py           raw 3D implied vol surface
  svi.py               SVI formula and fitting with outlier removal
  svi_surface.py       SVI fit per expiry, calendar arbitrage check
  diagnose.py          smile diagnostics for individual expiries
  local_vol.py         Dupire local vol and butterfly arbitrage check
  heston.py            Heston calibration with QuantLib
  compare_surfaces.py  implied vol surface produced by each model
  mc_barrier.py        Monte Carlo barrier pricing under three models
  vix.py               VIX rebuilt from SPX options
  vrp.py               variance risk premium since 1990
  index_compare.py     S&P 500 vs Nasdaq-100 vs Russell 2000
streamlit_app.py       the website
results/               saved outputs read by the website
```

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/Easher1011/vol-surface.git
cd vol-surface
uv sync
uv run python -m vol_surface.collect          # during US market hours
uv run python -m vol_surface.svi_surface
uv run python -m vol_surface.local_vol
uv run python -m vol_surface.compare_surfaces
uv run python -m vol_surface.mc_barrier
uv run python -m vol_surface.vix
uv run python -m vol_surface.vrp
uv run python -m vol_surface.index_compare
uv run streamlit run streamlit_app.py
```

Raw market data is not included in the repository. Results vary with market conditions.