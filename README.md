# SPX Volatility Surface and Exotic Option Pricing

Builds an implied volatility surface for S&P 500 index options from raw market quotes, derives local volatility and Heston models from it, and prices a barrier option under each model.

https://spx-vol-surface.streamlit.app/

## Key result

A 3-month down-and-in put on SPX (strike at the forward, barrier at 90% of spot, monitored daily), priced by Monte Carlo with 50,000 paths:

| Model | 13 Oct 2026 | 14 Oct 2026 | Probability of hitting barrier (14 Oct) |
|---|---|---|---|
| Flat vol (Black-Scholes) | 77.0 | 76.9 | 9.0% |
| Local vol (Dupire) | 152.9 | 151.0 | 16.6% |
| Heston | 157.9 | 156.6 | 16.8% |

Prices are in index points. All three models price the equivalent vanilla put close to the market (within about two Monte Carlo standard errors).

- Flat-vol pricing gives roughly half the value of the two skew-consistent models. It ignores the rise in volatility as the market falls, so it understates the chance of reaching the barrier.
- Local vol and Heston, calibrated to the same surface, agree within about 4%. This product depends mainly on the distribution of the final index level, which both models are fitted to. They would be expected to diverge more on products sensitive to how the smile moves over time.
- The result held when re-run on a second trading day.

## Method

1. **Data:** SPX option chain from Yahoo Finance via `yfinance`.
2. **Cleaning:** remove expiries under 7 days, zero bids, and bid-ask spreads wider than 10% of mid; keep out-of-the-money options only.
3. **Forwards:** for each expiry, regress call minus put on strike (put-call parity) to obtain the forward and discount factor, so no interest rate or dividend yield is assumed.
4. **Implied volatility:** Black-Scholes pricer inverted with a root-finder (Brent's method).
5. **SVI fit:** each expiry fitted with the raw SVI parameterisation in total variance. Stale quotes are handled by fitting once, removing quotes that miss by more than five times the median miss, and refitting.
6. **Arbitrage checks:** calendar (total variance must increase with maturity) and butterfly (Gatheral's g(k) must be positive), both checked within the quoted strike range. No violations were found on either day.
7. **Local volatility:** Dupire's formula in total-variance form, using SVI's analytic derivatives and expiries at least 10 days apart.
8. **Heston:** calibrated with QuantLib to the SVI surface for expiries of 30 days or more. Average fit error about 0.3 vol points.
9. **Pricing:** Monte Carlo with daily barrier monitoring. Heston is simulated with full truncation and four steps per day. The Heston simulation is validated against QuantLib's analytic vanilla price.

## Observations

- **Steep downside skew:** about 30% implied vol 15% below the forward, against about 10% above it, flattening with maturity.
- **Rising at-the-money term structure:** about 12% short-dated, rising to about 15% at seven months.
- **Heston's compromise:** the calibration requires very high vol of vol (about 1.8) and fast mean reversion (about 9) to span the steep short-dated skew and the flatter long-dated skew. These parameters fit prices but are not economically plausible, a known limitation of Heston.
- **Stale quotes:** outliers on the quarter-end expiries were identical across both days while the index moved, which identified them as stale quotes rather than genuine prices.

## Limitations

- Yahoo Finance quotes are not executable prices, and some stale quotes remain after cleaning.
- Expiries are fitted independently, so smoothness across maturities is not enforced; local vol shows steps between expiries.
- Heston uses a single flat rate and dividend yield, and fits the short-dated skew poorly.
- Monte Carlo error on barrier prices is roughly ±2 index points at 50,000 paths.
- Local vol reproduces today's prices but is known to misrepresent the future dynamics of the smile.

## Project structure

```
src/vol_surface/
  data.py              download the SPX option chain
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
streamlit_app.py       the website
results/               saved outputs read by the website
```

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/Easher1011/vol-surface.git
cd vol-surface
uv sync
uv run python src/vol_surface/data.py
uv run python -m vol_surface.svi_surface
uv run python -m vol_surface.local_vol
uv run python -m vol_surface.compare_surfaces
uv run python -m vol_surface.mc_barrier
uv run streamlit run streamlit_app.py
```

Raw market data is not included in the repository; `data.py` downloads a fresh snapshot. Results vary with market conditions.
