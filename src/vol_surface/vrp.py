from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

START = "1990-01-01"
WINDOW = 21   # trading days in roughly 30 calendar days, matching VIX's horizon


def load_history():
    """Daily VIX and S&P 500 closing levels."""
    vix = yf.Ticker("^VIX").history(start=START)["Close"]
    spx = yf.Ticker("^GSPC").history(start=START)["Close"]
    print(f"Downloaded {len(vix):,} VIX days and {len(spx):,} S&P 500 days")
    # VIX is stamped in Chicago time and the S&P 500 in New York time,
    # so keep just the date on each before lining them up
    vix.index = vix.index.tz_localize(None).normalize()
    spx.index = spx.index.tz_localize(None).normalize()
    return pd.DataFrame({"vix": vix, "spx": spx}).dropna()


def add_realised(df):
    """Realised vol over the NEXT 21 trading days, annualised, in vol points."""
    r = np.log(df["spx"]).diff()
    # sum of squared daily returns from tomorrow to 21 days ahead (zero mean, like a variance swap)
    future_var = (r ** 2).rolling(WINDOW).sum().shift(-WINDOW) * 252 / WINDOW
    df["rv"] = 100 * np.sqrt(future_var)
    df["vrp"] = df["vix"] - df["rv"]
    return df.dropna()


df = add_realised(load_history())

# non-overlapping months: one observation every 21 trading days
monthly = df.iloc[::WINDOW].copy()
# short variance swap P&L per unit of vega notional, in vol points: (K^2 - RV^2) / (2K)
monthly["short_var_pnl"] = (monthly["vix"] ** 2 - monthly["rv"] ** 2) / (2 * monthly["vix"])
monthly["cum_pnl"] = monthly["short_var_pnl"].cumsum()
drawdown = monthly["cum_pnl"] - monthly["cum_pnl"].cummax()

# --- headline numbers ---
print(f"Period: {df.index[0]:%b %Y} to {df.index[-1]:%b %Y} ({len(df):,} trading days)\n")
print(f"Average VIX:                       {df['vix'].mean():.1f}")
print(f"Average realised vol (next 30d):   {df['rv'].mean():.1f}")
print(f"Average gap (VIX minus realised):  {df['vrp'].mean():.1f} vol points")
print(f"Median gap:                        {df['vrp'].median():.1f} vol points")
print(f"VIX above what followed:           {(df['vrp'] > 0).mean():.0%} of days")

print("\nShort variance, monthly, non-overlapping:")
print(f"  Months:               {len(monthly)}")
print(f"  Profitable months:    {(monthly['short_var_pnl'] > 0).mean():.0%}")
print(f"  Average month:        {monthly['short_var_pnl'].mean():+.2f} vol points")
print(f"  Median month:         {monthly['short_var_pnl'].median():+.2f} vol points")
print(f"  Worst drawdown:       {drawdown.min():.1f} vol points")

print("\nFive worst months for a volatility seller:")
worst = monthly.nsmallest(5, "short_var_pnl")[["vix", "rv", "short_var_pnl"]]
print(worst.round(1).rename(columns={"vix": "VIX", "rv": "realised", "short_var_pnl": "P&L"})
      .to_string())

print("\nAverage by year:")
yearly = df.groupby(df.index.year)[["vix", "rv", "vrp"]].mean()
print(yearly.round(1).rename(columns={"vix": "VIX", "rv": "realised", "vrp": "gap"}).to_string())

# --- charts ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("VIX vs realised vol over the following 30 days",
                                    "Gap: VIX minus subsequent realised vol",
                                    "Cumulative P&L from selling variance monthly (vol points)"))
fig.add_scatter(x=df.index, y=df["vix"], name="VIX", line=dict(width=1), row=1, col=1)
fig.add_scatter(x=df.index, y=df["rv"], name="Realised (next 30d)", line=dict(width=1), row=1, col=1)
fig.add_scatter(x=df.index, y=df["vrp"], name="Gap", line=dict(width=1),
                fill="tozeroy", row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)
fig.add_scatter(x=monthly.index, y=monthly["cum_pnl"], name="Short variance P&L", row=3, col=1)
fig.update_layout(template="plotly_dark", height=950,
                  title="SPX variance risk premium since 1990")

Path("output").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
fig.write_html("output/vrp.html")
fig.write_json("results/vrp.json")
fig.show()

# save summary numbers and tables for the website
import json
summary = dict(
    start=f"{df.index[0]:%b %Y}", end=f"{df.index[-1]:%b %Y}",
    avg_vix=float(df["vix"].mean()), avg_rv=float(df["rv"].mean()),
    avg_gap=float(df["vrp"].mean()), median_gap=float(df["vrp"].median()),
    pct_days_positive=float((df["vrp"] > 0).mean()),
    months=len(monthly), pct_months_profitable=float((monthly["short_var_pnl"] > 0).mean()),
    avg_month=float(monthly["short_var_pnl"].mean()),
    median_month=float(monthly["short_var_pnl"].median()),
    worst_drawdown=float(drawdown.min()),
)
with open("results/vrp_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
worst.to_csv("results/vrp_worst_months.csv")
yearly.to_csv("results/vrp_yearly.csv")