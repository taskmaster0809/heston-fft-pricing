#ifndef HESTON_FFT_H
#define HESTON_FFT_H

#include <complex> // For std::complex
#include <vector>  // For std::vector

using complex = std::complex<double>;
using vector = std::vector<double>;

complex heston_cf(complex u, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T);
complex heston_fourier_transform(double v, double alpha, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T);
vector heston_fft_price(double S0, const vector& K, double v0, double eta, double alpha, double r, double rho, double kappa, double theta, double xi, double T, int N);

#endif
