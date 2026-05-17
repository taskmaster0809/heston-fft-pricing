import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))
sys.path.append(str(Path(__file__).parent.parent)) # Append python directory

from scipy.optimize import least_squares, brentq
import numpy as np

import heston # type: ignore
from data.market_data import MarketData

# Arbitrary grid spacing parameters
ETA = 0.1
N = 2 ** 11 # Should be a power of 2 for efficiency
ALPHA = 1.5 # Damping factor, typically set to 1.5


def implied_vol_from_price(S0, K, T, r, price):
    def objective(imp_vol):
        return heston.black_scholes_price(S0, K, T, r, imp_vol) - price

    try:
        return brentq(f=objective, a=1e-6, b=5)
    except ValueError:
        return np.nan


def calibrate_params(market_data: MarketData, calls_data, v0_init=0.03, rho_init = -0.5, kappa_init = 10,
                     theta_init = 0.04, xi_init = 0.2, lower_bound=None, upper_bound=None):
    strikes = calls_data["strike"].values
    maturities = calls_data["timeToExpiry"].values
    market_ivs = calls_data["impliedVol"].values
    vega = np.array([
        market_data.vega(K, T, market_data.interest_rate, imp_vol)
        for K, T, imp_vol in zip(strikes, maturities, market_ivs)
    ])
    weights = vega / vega.sum()

    unique_maturities = calls_data["timeToExpiry"].unique()

    def implied_vol_residuals(x):  # x=(v0, rho, kappa, theta, xi)
        heston_prices = np.empty(len(calls_data))

        for T in unique_maturities:
            indices = np.isclose(calls_data["timeToExpiry"], T)
            this_strikes = calls_data.loc[indices, "strike"].values
            prices = heston.heston_fft_price(market_data.spot, this_strikes, x[0], ETA, ALPHA,
                                             market_data.interest_rate,
                                             x[1], x[2], x[3], x[4], T, N)
            heston_prices[indices] = prices

        heston_ivs = [
            implied_vol_from_price(market_data.spot, K, T, market_data.interest_rate, heston_price)
            for K, T, heston_price in zip(strikes, maturities, heston_prices)
        ]

        residuals = np.array(market_ivs) - np.array(heston_ivs)
        residuals[~np.isfinite(residuals)] = 1e6  # Punish bad parameter region
        residuals = residuals * np.sqrt(weights)  # Weight residuals by normalized vega
        return residuals

    if lower_bound is None:
        lower_bound=[0.01, -0.95, 0.5, 0.01, 0.05]
    if upper_bound is None:
        upper_bound = [1.0, -0.2, 20.0, 1.0, 1.0]

    result = least_squares(fun=implied_vol_residuals, x0=(v0_init, rho_init, kappa_init, theta_init, xi_init),
                           bounds=(lower_bound, upper_bound))
    return result.x
