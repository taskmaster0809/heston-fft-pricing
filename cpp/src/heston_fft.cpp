#include <fftw3.h>

#include "heston_fft.h"

#include <cmath>     // For std::exp, std::sqrt, std::log
#include <complex>   // For std::complex
#include <stdexcept> // For std::invalid_argument
#include <vector>    // For std::vector
#include <algorithm> // For std::lower_bound, std::any_of
#include <iostream>  // For std::cerr

using std::exp, std::log, std::sqrt;
using complex = std::complex<double>;
using vector = std::vector<double>;

complex heston_cf(complex u, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T)
{
    constexpr complex i { 0.0, 1.0 };
    const complex x = kappa - rho * xi * i * u;
    const complex d = sqrt(x * x + xi * xi * (i * u + u * u));
    const complex g = (x - d) / (x + d);
    const complex C = i * u * r * T + (kappa * theta / (xi * xi)) * ((x - d) * T - 2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)));
    const complex D = ((x - d) / (xi * xi)) * ((1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T)));

    return exp(i * u * log(S0) + C + D * v0);
}

complex heston_fourier_transform(double v, double alpha, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T)
{
    constexpr complex i { 0.0, 1.0 };
    const complex phi = heston_cf(v - i * (alpha + 1), S0, v0, r, rho, kappa, theta, xi, T);
    
    return exp(-r * T) * phi / (alpha * alpha + alpha - v * v + i * (2 * alpha + 1) * v);
}

vector heston_fft_price(double S0, const vector& K, double v0, double eta, double alpha, double r, double rho, double kappa, double theta, double xi, double T, int N)
{
    if(alpha <= 0){
        throw std::invalid_argument("alpha must be positive");
    }

    if(S0 <= 0 || v0 < 0 || kappa <= 0 || theta < 0 || xi <= 0 || T <= 0 || (rho < -1 || rho > 1) || N <= 0 || 
       std::any_of(K.begin(), K.end(), [](double n){ return n <= 0; }))
    {
        throw std::invalid_argument("Invalid range for parameters");
    }

    if(2 * kappa * theta <= xi * xi) {
        std::cerr << "Warning: Feller condition is not satisfied\n";
    }

    constexpr double pi { 3.14159265358979323846264338327950288 }; 
    constexpr complex i { 0.0, 1.0 };

    const double lambda { (2 * pi) / (N * eta) };
    const double b { pi / eta };
    vector log_strikes(K.size());
    std::transform(K.begin(), K.end(), log_strikes.begin(), [](double k){ return log(k); });

    vector log_strike_grid(N);

    for(int u = 0; u < N; ++u){
        log_strike_grid[u] = -b + lambda * u;
    }

    if(std::any_of(log_strikes.begin(), log_strikes.end(), [&](double log_strike){ return log_strike <= log_strike_grid[0] 
        || log_strike > log_strike_grid[N-1]; }))
    {
        throw std::out_of_range("Strike outside FFT grid range. Adjust N and/or eta");
    }

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    double v_j {};
    complex psi {};
    bool delta {};
    double w_j {}; // Simpson weight
    complex integrand {};

    for(int j = 0; j < N; ++j){
        v_j = eta * j;
        psi = heston_fourier_transform(v_j, alpha, S0, v0, r, rho, kappa, theta, xi, T);
        delta = (j == 0 ? 1 : 0);
        w_j = (eta / 3) * (3 + (j % 2 == 0 ? -1 : 1) - delta);
        integrand = exp(i * b * v_j) * psi * w_j;
        in[j][0] = integrand.real();
        in[j][1] = integrand.imag(); 
    }

    fftw_execute(p); // Execute FFT plan
    
    vector call_price_grid(N);
    vector call_price(K.size());

    for(int u = 0; u < N; ++u){
        call_price_grid[u] = (exp(-alpha * log_strike_grid[u]) / pi) * out[u][0];
    }

    for(size_t j = 0; j < log_strikes.size(); ++j){
        auto it = std::lower_bound(log_strike_grid.begin(), log_strike_grid.end(), log_strikes[j]);
        int idx = it - log_strike_grid.begin();
        if(idx >= N) idx = N-1;
        if(idx <= 0) idx = 1;

        double lower_call_price = call_price_grid[idx - 1];
        double upper_call_price = call_price_grid[idx];

        // Interpolating call price in the call price grid
        call_price[j] = lower_call_price + ((log_strikes[j] - log_strike_grid[idx - 1]) / lambda) * (upper_call_price - lower_call_price);
    }

    fftw_destroy_plan(p);
    fftw_free(in); 
    fftw_free(out);

    return call_price;
}
