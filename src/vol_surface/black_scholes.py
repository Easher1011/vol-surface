import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, sigma, kind="call"):
    """Black-Scholes price of a European option."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, kind="call"):
    """Find the sigma that makes bs_price equal the market price."""

    def difference(sigma):
        return bs_price(S, K, T, r, sigma, kind) - price

    return brentq(difference, 0.0001, 5.0)


if __name__ == "__main__":
    price = bs_price(S=100, K=100, T=1, r=0.04, sigma=0.20, kind="call")
    print("Call price:", price)

    iv = implied_vol(price, S=100, K=100, T=1, r=0.04, kind="call")
    print("Implied vol:", iv)

# ---------------- NOTES -----------------
#  d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
#  d2 = d1 − σ·√T
#  Call = S·N(d1) − K·e^(−rT)·N(d2)
#  Put = K·e^(−rT)·N(−d2) − S·N(−d1)
# ----------------------------------------