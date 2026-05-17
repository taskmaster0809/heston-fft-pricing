import sys
from pathlib import Path
from math import sqrt
import time

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))

import heston # type: ignore

S0=100; K=[90, 100, 110]; v0=0.04; r=0.1; rho=-0.5; kappa=2; theta=0.04; xi=1e-2; T=1
num_paths = 100_000

alpha = 1.5
eta = 0.1
N = 2 ** 14 

heston_mc_prices = []
print("Performing Monte Carlo simulations...")
start = time.time()
for i in range(len(K)):
      heston_mc_prices.append(heston.heston_mc_price(S0, K[i], v0, r, rho, kappa, theta, xi, T, num_paths=num_paths))
      print(f"Monte Carlo price for K = {K[i]}: {heston_mc_prices[i]:.3f}")
end = time.time()

print(f"Time taken by Monte Carlo method for {num_paths} paths and {len(K)} strikes: {end - start:.4f} seconds\n")

start = time.time()
heston_fft_price = heston.heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N)
end = time.time()
for i in range(len(K)): 
      print(f"FFT price for K = {K[i]}: {heston_fft_price[i]:.3f}") 
print(f"Time taken by FFT method for {N} grid FFT for {len(K)} strikes: {(end - start):.4f} seconds\n")


# For v0 = theta and small xi, Heston model converges to the Black Scholes model
mc_error = 0; fft_error = 0
for i in range(len(K)):
      bs_price = heston.black_scholes_price(S0, K[i], T, r, sqrt(v0)) # Black-Scholes function takes sigma, i.e., sqrt(v0)
      mc_error += abs(bs_price - heston_mc_prices[i]) / max(bs_price, heston_mc_prices[i]) * 100
      fft_error += (abs(bs_price - heston_fft_price[i]) / max(bs_price, heston_fft_price[i])) * 100

print(f"Average Black Scholes and Heston model relative error via Monte Carlo: "
      f"{mc_error / len(K):.5f}%")

print(f"Average Black Scholes and Heston model relative error via FFT: "
      f"{fft_error / len(K):.5f}%")

# Fast Fourier Transform is much faster than Monte Carlo with good accuracy and is appropriate if we have multiple options. 
# However, FFT is limited to vanilla European options where CF can be calculated analytically. Moreover, Monte Carlo can 
# surpass FFT in accuracy with enough computations (for example, try using a million paths) but is much slower
