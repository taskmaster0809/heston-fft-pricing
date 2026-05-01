#include <complex> // For std::complex

std::complex<double> heston_cf(std::complex<double> u, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T);
std::complex<double> heston_fourier_transform(double v, double alpha, double S0, double v0, double r, double rho, double kappa, double theta, double xi, double T);
