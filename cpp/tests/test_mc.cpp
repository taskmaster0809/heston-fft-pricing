#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "heston_mc.h"
#include "utils.h"

#include <cmath>     // For std::sqrt
#include <iostream>  // For std::cout
#include <algorithm> // For std::max

// Change parameters to test different parameters
constexpr double S0 { 100 }, v0 { 0.04 }, r { 0.1 }, rho { -0.5 }, kappa { 2 }, theta { 0.04 }, xi { 0.2 }, T { 1 };
constexpr int N { 252 }, num_paths { 10000 };

// Simple test case
TEST_CASE("Heston price is less than spot [MC]")
{
    auto K = GENERATE(90, 100, 110); // Different strikes for ITM, OTM and ATM options
    double heston_price = heston_mc_price(S0, K, v0, r, rho, kappa, theta, xi, T);
    REQUIRE( (heston_price < S0 && heston_price > 0) );
}

// Validating MC pricer against Black Scholes
TEST_CASE("Heston price is close to Black-Scholes price for xi approx 0 and v0 = theta [MC]")
{
    auto K = GENERATE(90, 100, 110); 

    double heston_price = heston_mc_price(S0, K, v0, r, rho, kappa, theta, 0.000000001, T, N, num_paths);
    double bs_price = black_scholes_price(S0, K, T, r, std::sqrt(v0));
    std::cout << "K=" << K 
          << " MC=" << heston_price 
          << " BS=" << bs_price 
          << " rel err=" << std::abs(heston_price - bs_price) / std::max(heston_price, bs_price)
          << "\n";

    double mc_error = 3.0 / std::sqrt(num_paths);
    double tol { mc_error };

    // Increase tolerance for OTM and ATM options
    if(std::abs(K - S0) < 1e-12) tol *= 1.4;
    else if(K > S0) tol *= 1.8;

    REQUIRE_THAT( heston_price, Catch::Matchers::WithinRel(bs_price, tol) );
}

TEST_CASE("Higher strike implies lower price [MC]")
{
    double high_k { 110.0 }, low_k { 90.0 }, lowest_k { 70.0 };

    double high_k_price = heston_mc_price(S0, high_k, v0, r, rho, kappa, theta, xi, T, N, num_paths);
    double low_k_price = heston_mc_price(S0, low_k, v0, r, rho, kappa, theta, xi, T, N, num_paths);
    double lowest_k_price = heston_mc_price(S0, lowest_k, v0, r, rho, kappa, theta, xi, T, N, num_paths);

    REQUIRE( (high_k_price < low_k_price && low_k_price < lowest_k_price) );
}

TEST_CASE("Higher volatility implies higher price [MC]")
{
    auto K = GENERATE(90, 100, 110);
    double high_v { 0.08 }, low_v { 0.04 }, high_xi { 0.09 }, low_xi { 0.05 }, high_theta { 0.04 }, low_theta { 0.02 };

    double high_v_price = heston_mc_price(S0, K, high_v, r, rho, kappa, high_theta, high_xi, T, N, num_paths);
    double low_v_price = heston_mc_price(S0, K, low_v, r, rho, kappa, low_theta, low_xi, T, N, num_paths);

    REQUIRE( high_v_price > low_v_price );
}
