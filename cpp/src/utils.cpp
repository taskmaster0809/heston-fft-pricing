#include "../include/utils.h"
#include <cmath>
#include <stdexcept>

using std::exp, std::log, std::sqrt;

double norm_cdf(double value)
{
    const double inv_sqrt2 = 1 / sqrt(2.0);
    return 0.5 * std::erfc(-value * inv_sqrt2);
}

double black_scholes_price(double S, double K, double T, double r, double sigma)
{
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0){
        throw std::invalid_argument("S, K, T and sigma must be strictly positive");
    }

    double d1 = (log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    return S * norm_cdf(d1) - norm_cdf(d2) * K * exp(-r * T);
}
