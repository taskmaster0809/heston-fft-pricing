import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))

import heston
from math import sqrt
import time

S0=100; K=100; v0=0.04; r=0.05; rho=-0.7; kappa=2; theta=0.04; xi=0.000001; T=1; num_paths=1_000_000
# Heston price converges to Black-Scholes price as xi tends to 0 and v0 tends to theta

bs_price = heston.black_scholes_price(S0, K, T, r, sqrt(v0)) # Black-Scholes function takes sigma, i.e., sqrt(v0)

start = time.time()
heston_price = heston.heston_mc_price(S0, K, v0, r, rho, kappa, theta, xi, T, num_paths=num_paths)
print(f"Time taken by heston_price for {num_paths} paths: {(time.time() - start):.4f} seconds")

print(f"Black Scholes and Heston model discrepancy: {(abs(bs_price - heston_price)):.5f}")
