#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "heston_fft.h"
#include "heston_mc.h"

#include <cmath>    // For std::sqrt
#include <iostream> // For std::cout

constexpr double S0 { 100 }, v0 { 0.04 }, eta { 0.1 }, alpha { 1.5 }, r { 0.1 }, rho { -0.5 }, kappa { 2 }, theta { 0.04 }, xi { 0.2 }, T { 1 };
constexpr int N { 16384 }; // Choose a power of 2 for efficiency

// Simple test case
TEST_CASE("Heston price is less than spot [FFT]")
{
    auto K = GENERATE(90, 100, 110); // Different strikes for ITM, OTM and ATM options
    double heston_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    REQUIRE( (heston_price < S0 && heston_price > 0) );
}

// Validating FFT pricer against Monte Carlo pricer with 5% tolerance
TEST_CASE("Heston price is close to Heston Monte Carlo price [FFT]")
{
    auto K = GENERATE(90, 100, 110); 

    double heston_price_fft = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    double heston_price_mc = heston_mc_price(S0, K, v0, r, rho, kappa, theta, xi, T, 252, 10000);
        std::cout << "K=" << K 
          << " FFT=" << heston_price_fft 
          << " MC=" << heston_price_mc
          << " rel err=" << std::abs(heston_price_fft - heston_price_mc) / std::max(heston_price_fft, heston_price_mc)
          << "\n";

    REQUIRE_THAT( heston_price_fft, Catch::Matchers::WithinRel(heston_price_mc, 0.05) );
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

