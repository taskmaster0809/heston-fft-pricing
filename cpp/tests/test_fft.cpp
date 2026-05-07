#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "heston_fft.h"
#include "heston_mc.h"
#include "utils.h"

#include <cmath>    // For std::sqrt
#include <iostream> // For std::cout

constexpr double S0 { 100 }, v0 { 0.04 }, eta { 0.1 }, alpha { 1.5 }, r { 0.1 }, rho { -0.5 }, kappa { 2 }, theta { 0.04 }, xi { 0.2 }, T { 1 };
const int N { static_cast<int>(pow(2, 14)) }; // Choose a power of 2 for efficiency

// Simple test case
TEST_CASE("Heston price is less than spot [FFT]")
{
    auto K = GENERATE(90, 100, 110); // Different strikes for ITM, OTM and ATM options
    double heston_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    REQUIRE( (heston_price < S0 && heston_price > 0) );
}

// Validating FFT pricer against Monte Carlo pricer with 5% tolerance
TEST_CASE("Heston FFT price is close to Heston Monte Carlo price [FFT]")
{
    auto K = GENERATE(90, 100, 110); 

    double fft_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    double mc_price = heston_mc_price(S0, K, v0, r, rho, kappa, theta, xi, T, 252, 10000);
    std::cout << "K=" << K 
    << " FFT=" << fft_price 
    << " MC=" << mc_price
    << " rel err=" << std::abs(fft_price - mc_price) / std::max(fft_price, mc_price)
    << "\n";

    REQUIRE_THAT( fft_price, Catch::Matchers::WithinRel(mc_price, 0.05) );
}

// Validating FFT pricer against Black-Scholes pricer with 1% tolerance
TEST_CASE("Heston converges to Black-Scholes when xi -> 0 and v0 = theta [FFT]")
{
    auto K = GENERATE(90.0, 100.0, 110.0);

    double small_xi = 1e-2; // Cannot take xi too small as that blows up heston cf and leads to numerical instability

    double fft_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, small_xi, T, N);
    double bs_price = black_scholes_price(S0, K, T, r, std::sqrt(v0));

    std::cout << "K=" << K 
    << " FFT=" << fft_price 
    << " BS=" << bs_price
    << " rel err=" << std::abs(fft_price - bs_price) / std::max(fft_price, bs_price)
    << "\n";

    REQUIRE_THAT(fft_price, Catch::Matchers::WithinRel(bs_price, 0.01));
}

TEST_CASE("Higher strike implies lower price [FFT]")
{
    double high_k { 110.0 }, low_k { 90.0 }, lowest_k { 70.0 };

    double high_k_price = heston_fft_price(S0, high_k, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    double low_k_price = heston_fft_price(S0, low_k, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    double lowest_k_price = heston_fft_price(S0, lowest_k, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);

    REQUIRE( (high_k_price < low_k_price && low_k_price < lowest_k_price) );
}

TEST_CASE("Higher volatility implies higher price [FFT]")
{
    auto K = GENERATE(90, 100, 110);
    double high_v { 0.08 }, low_v { 0.04 }, high_xi { 0.09 }, low_xi { 0.05 }, high_theta { 0.04 }, low_theta { 0.02 };

    double high_v_price = heston_fft_price(S0, K, high_v, eta, alpha, r, rho, kappa, high_theta, high_xi, T, N);
    double low_v_price = heston_fft_price(S0, K, low_v, eta, alpha, r, rho, kappa, low_theta, low_xi, T, N);

    REQUIRE( high_v_price > low_v_price );
}

