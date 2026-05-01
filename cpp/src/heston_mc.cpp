#include "heston_mc.h"

#include <iostream>  // For std::cerr
#include <cmath>     // For std::exp and std::sqrt
#include <random>    // For std::mt19937 and std::normal_distribution
#include <stdexcept> // For std::invalid_argument
#include <algorithm> // For std::max
#include <thread>    // For multi-threading
#include <vector>    // For std::vector
#include <numeric>   // For std::reduce

thread_local std::mt19937 rng{ std::random_device{}() };
thread_local std::normal_distribution<double> dist{0.0, 1.0};

using std::exp, std::sqrt;

double heston_single_path_payoff(double S0, double K, double v0, double r, double rho, double kappa, double theta, double xi, double T, int N)
{
    const double dt { T / N };

    double old_S { S0 };
    double old_v { v0 };

    double new_S { old_S };
    double new_v { old_v };

    double Z1 {};
    double Z2 {};

    double vol {};
    const double sqrt_one_minus_rho_sq { sqrt(1 - rho * rho) };

    for(int i = 0; i < N; ++i){
        Z1 = dist(rng);
        Z2 = rho * Z1 + sqrt_one_minus_rho_sq * dist(rng);

        vol = sqrt(std::max(old_v, 0.0) * dt);

        new_S = old_S * exp((r - 0.5 * old_v) * dt + vol * Z1);
        new_v = old_v + kappa * (theta - old_v) * dt + xi * vol * Z2;
        new_v = std::max(new_v, 0.0); // Truncating variance to avoid negative variance

        old_S = new_S;
        old_v = new_v;
    }

    return std::max(new_S - K, 0.0);
}

double hestonMCPrice(double S0, double K, double v0, double r, double rho, double kappa, double theta, double xi, double T, int N, int num_paths)
{
    if(num_paths <= 0){
        throw std::invalid_argument("Number of paths must be positive");
    }

    if(S0 <= 0 || v0 < 0 || K <= 0 || kappa <= 0 || theta < 0 || xi <= 0 || T <= 0 || N <= 0 || (rho < -1 || rho > 1)){
        throw std::invalid_argument("Invalid range for parameters");
    }

    if(2 * kappa * theta <= xi * xi) {
        std::cerr << "Warning: Feller condition is not satisfied\n";
    }

    int num_threads = std::thread::hardware_concurrency(); //Multi-threading for optimal performance
    if (num_threads == 0) num_threads = 2;                 //Default to 2 threads if number of threads is indeterminate 

    std::vector<double> partial_payoff_sum(num_threads, 0.0);
    std::vector<std::thread> threads;
    int work {};

    for(int i = 0; i < num_threads; ++i){
        work = (i == num_threads - 1 ? num_paths / num_threads + num_paths % num_threads : num_paths / num_threads);
        threads.emplace_back(
            [&, i, work](){
            double local_sum {};

            for(int j = 0; j < work; ++j){
                local_sum += heston_single_path_payoff(S0, K, v0, r, rho, kappa, theta, xi, T, N);
            }
            partial_payoff_sum[i] = local_sum;
        }
        );
    }

    for(auto& t : threads){
        t.join();
    }    

    double payoff_sum = std::reduce(partial_payoff_sum.begin(), partial_payoff_sum.end(), 0.0);
    double average_payoff = payoff_sum / num_paths;

    return exp(-r * T) * average_payoff;
}
