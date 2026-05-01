#include "heston_fft.h"
#include <iostream>

#include <cmath>     // For std::exp, std::sqrt and std::log
#include <complex>   // For std::complex
#include <stdexcept> // For std::invalid_argument

using std::exp, std::log, std::sqrt;

std::complex<double> heston_cf(std::complex<double> u, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T)
{
    if(S0 <= 0 || v0 < 0 || kappa <= 0 || theta < 0 || xi <= 0 || T <= 0 || (rho < -1 || rho > 1)){
        throw std::invalid_argument("Invalid range for parameters");
    }

    constexpr std::complex<double> i { 0.0, 1.0 };
    const std::complex<double> x = kappa - rho * xi * i * u;
    const std::complex<double> d = sqrt(x * x + xi * xi * (i * u + u * u));
    const std::complex<double> g = (x - d) / (x + d);
    const std::complex<double> C = i * u * r * T + (kappa * theta / (xi * xi)) * ((x - d) * T - 2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)));
    const std::complex<double> D = ((x - d) / (xi * xi)) * ((1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T)));

    return exp(i * u * log(S0) + C + D * v0);
}

std::complex<double> heston_fourier_transform(double v, double alpha, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T)
{
    if(alpha <= 0){
        throw std::invalid_argument("alpha must be positive");
    }

    constexpr std::complex<double> i { 0.0, 1.0 };
    const std::complex<double> phi = heston_cf(v - i * (alpha + 1), S0, v0, r, rho, kappa, theta, xi, T);
    
    return exp(-r * T) * phi / (alpha * alpha + alpha - v * v + i * (2 * alpha + 1) * v);
}
