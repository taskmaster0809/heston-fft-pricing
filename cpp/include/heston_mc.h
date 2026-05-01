#ifndef HESTON_MC_H
#define HESTON_MC_H

double heston_single_path_payoff(double S0, double K, double v0, double r, double rho, double kappa, double theta, double xi, double T, int N=252);
double hestonMCPrice(double S0, double K, double v0, double r, double rho, double kappa, double theta, double xi, double T, int N=252, int num_paths=1000);
void set_seed(unsigned int seed);

#endif