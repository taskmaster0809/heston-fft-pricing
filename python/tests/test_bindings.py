import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))

import heston # type: ignore
from math import sqrt
import time

S0=100; K=100; v0=0.04; r=0.1; rho=-0.5; kappa=2; theta=0.04; xi=1e-2; T=1
num_paths = 100_000

alpha = 1.5
eta = 0.1
N = 2 ** 14 

bs_price = heston.black_scholes_price(S0, K, T, r, sqrt(v0)) # Black-Scholes function takes sigma, i.e., sqrt(v0)

print("Performing Monte Carlo simulations...")
start = time.time()
heston_mc_price = heston.heston_mc_price(S0, K, v0, r, rho, kappa, theta, xi, T, num_paths=num_paths)
end = time.time()
print(f"Time taken by Monte Carlo method for {num_paths} paths: {(end - start):.4f} seconds")
print(f"Monte Carlo price: {heston_mc_price:.3f}\n")

start = time.time()
heston_fft_price = heston.heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N)
end = time.time()
print(f"Time taken by FFT method for {N} grid FFT: {(end - start):.4f} seconds")
print(f"FFT price: {heston_fft_price:.3f}\n") 

# For v0 = theta and small xi, Heston model converges to the Black Scholes model
print(f"Black Scholes and Heston model relative error via Monte Carlo: "
      f"{(abs(bs_price - heston_mc_price) / max(bs_price, heston_mc_price)) * 100:.5f}%")

print(f"Black Scholes and Heston model relative error via FFT: "
      f"{(abs(bs_price - heston_fft_price) / max(bs_price, heston_fft_price)) * 100:.5f}%")

# Fast Fourier Transform is much faster than Monte Carlo with good accuracy. However, FFT is limited to vanilla
# European options where CF can be calculated analytically. Moreover, Monte Carlo can surpass FFT in accuracy with
# enough computations (for example, try using a million paths) but is much slower
