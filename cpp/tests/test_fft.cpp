#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "heston_fft.h"
#include "heston_mc.h"
#include "utils.h"

#include <cmath>     // For std::sqrt
#include <vector>    // For std::vector
#include <algorithm> // For std::any_of
#include <iostream>  // For std::cout

using dvec = std::vector<double>;

constexpr double S0 { 100 }, v0 { 0.04 }, eta { 0.1 }, alpha { 1.5 }, r { 0.1 }, rho { -0.5 }, kappa { 2 }, theta { 0.04 }, xi { 0.2 }, T { 1 };
const int N { static_cast<int>(pow(2, 14)) }; // Choose a power of 2 for efficiency

// Simple test case
TEST_CASE("Heston price is less than spot [FFT]")
{  
    dvec K { 90, 100, 110 }; // Different strikes for ITM, OTM and ATM options
    dvec heston_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);
    REQUIRE( std::all_of(heston_price.begin(), heston_price.end(), [](double price){ return price < S0 && price > 0; }) );
}

// Validating FFT pricer against Monte Carlo pricer with 5% tolerance
TEST_CASE("Heston FFT price is close to Heston Monte Carlo price [FFT]")
{ 
    dvec K { 90, 100, 110 };

    dvec fft_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);

    for(size_t i = 0; i < K.size(); ++i){
        double mc_price = heston_mc_price(S0, K[i], v0, r, rho, kappa, theta, xi, T, 252, 10000);
        std::cout << "K=" << K[i]
        << " FFT=" << fft_price[i]
        << " MC=" << mc_price
        << " rel err=" << std::abs(fft_price[i] - mc_price) / std::max(fft_price[i], mc_price)
        << "\n";

        REQUIRE_THAT( fft_price[i], Catch::Matchers::WithinRel(mc_price, 0.05) );
    }
}

// Validating FFT pricer against Black-Scholes pricer with 1% tolerance
TEST_CASE("Heston converges to Black-Scholes when xi -> 0 and v0 = theta [FFT]")
{
    dvec K { 90, 100, 110 };

    double small_xi { 1e-2 }; // Cannot take xi too small as that blows up heston cf and leads to numerical instability

    dvec fft_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, small_xi, T, N);

    for(size_t i = 0; i < K.size(); ++i){
        double bs_price = black_scholes_price(S0, K[i], T, r, std::sqrt(v0));
        std::cout << "K=" << K[i] 
        << " FFT=" << fft_price[i] 
        << " BS=" << bs_price
        << " rel err=" << std::abs(fft_price[i] - bs_price) / std::max(fft_price[i], bs_price)
        << "\n";

        REQUIRE_THAT(fft_price[i], Catch::Matchers::WithinRel(bs_price, 0.01));
    }
}

TEST_CASE("Higher strike implies lower price [FFT]")
{
    dvec K { 70, 80, 90, 100, 110, 120 };
    dvec heston_price = heston_fft_price(S0, K, v0, eta, alpha, r, rho, kappa, theta, xi, T, N);

    for(size_t i = 0; i < K.size() - 1; ++i){
        REQUIRE( heston_price[i] >= heston_price[i+1] );
    }
}

TEST_CASE("Higher volatility implies higher price [FFT]")
{
    dvec K = { 90, 100, 110 };
    double high_v { 0.08 }, low_v { 0.04 }, high_xi { 0.09 }, low_xi { 0.05 }, high_theta { 0.04 }, low_theta { 0.02 };

    dvec high_v_price = heston_fft_price(S0, K, high_v, eta, alpha, r, rho, kappa, high_theta, high_xi, T, N);
    dvec low_v_price = heston_fft_price(S0, K, low_v, eta, alpha, r, rho, kappa, low_theta, low_xi, T, N);

    for(size_t i = 0; i < K.size(); ++i){
        REQUIRE(high_v_price[i] > low_v_price[i]);
    }
}
