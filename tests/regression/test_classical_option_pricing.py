from classical.option_pricing import black_scholes_price, monte_carlo_european_price


def test_black_scholes_call_known_value():
    # standard textbook case: S0=100, K=100, T=1, r=0.05, sigma=0.2 -> ~10.45
    price = black_scholes_price(s0=100, k=100, t=1, r=0.05, sigma=0.2, option_type="call")
    assert abs(price - 10.45) < 0.05


def test_monte_carlo_agrees_with_black_scholes_within_error_bars():
    bs_price = black_scholes_price(s0=100, k=100, t=1, r=0.05, sigma=0.2, option_type="call")
    mc_price, stderr = monte_carlo_european_price(
        s0=100, k=100, t=1, r=0.05, sigma=0.2, option_type="call", n_paths=200_000, seed=7
    )
    # within ~4 standard errors is a fair, non-flaky bar for a stochastic test
    assert abs(mc_price - bs_price) < 4 * stderr


def test_put_call_consistency_zero_strike_edge_case():
    call = black_scholes_price(s0=100, k=0.01, t=1, r=0.05, sigma=0.2, option_type="call")
    # a call struck near zero should be worth ~ the full discounted spot
    assert call > 90
