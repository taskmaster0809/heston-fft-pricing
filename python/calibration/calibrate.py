import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))
sys.path.append(str(Path(__file__).parent.parent))

from scipy.optimize import least_squares, brentq
import numpy as np

import heston # type: ignore

from data.market_data import MarketData

# Arbitrary grid spacing parameters
ETA = 0.1
N = 2 ** 14 # Should be a power of 2 for efficiency

ALPHA = 1.5 # Damping factor, typically set to 1.5

# Initial parameters
v0_init = 0.04; rho_init = -0.5; kappa_init = 2; theta_init = 0.04; xi_init = 0.2

data = MarketData("^SPX")

calls_data = data.get_calls()

strikes = calls_data["strike"].values
maturities = calls_data["timeToExpiry"].values
market_ivs = calls_data["impliedVol"].values

unique_maturities = calls_data["timeToExpiry"].unique()


def implied_vol_from_price(S0, K, T, r, price):
    def objective(imp_vol):
        return heston.black_scholes_price(S0, K, T, r, imp_vol) - price

    try:
        return brentq(f=objective, a=1e-6, b=5)
    except ValueError:
        return np.nan


def implied_vol_residuals(x): # x=(v0, rho, kappa, theta, xi)
    heston_prices = np.empty(len(calls_data))

    for T in unique_maturities:
        indices = np.isclose(calls_data["timeToExpiry"], T)
        this_strikes = calls_data.loc[indices, "strike"].values
        prices = heston.heston_fft_price(data.spot, this_strikes, x[0], ETA, ALPHA, data.interest_rate,
                                         x[1], x[2], x[3], x[4], T, N)

        heston_prices[indices] = prices

    heston_ivs = [
        implied_vol_from_price(data.spot, K, T, data.interest_rate, heston_price)
        for K, T, heston_price in zip(strikes, maturities, heston_prices)
    ]

    residuals = np.array(market_ivs) - np.array(heston_ivs)
    residuals[~np.isfinite(residuals)] = 1e6 # Punish bad parameter region
    return residuals

