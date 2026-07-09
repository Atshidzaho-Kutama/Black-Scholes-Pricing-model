# Black-Scholes Option Pricing Model

A single-file Python implementation of the Black-Scholes framework for
pricing European-style options, exploring how prices respond to changing
market conditions, and validating the pricing logic against known
theoretical properties.

## What it does

- Prices European call and put options given the five standard Black-Scholes
  inputs: underlying price, strike price, time to maturity, risk-free rate,
  and volatility.
- Computes the option Greeks (delta, gamma, vega, theta, rho) — the
  sensitivities of the option price to each input.
- Validates the implementation against theoretical properties that must
  hold if the pricing is correct: put-call parity, analytical vs.
  finite-difference delta, and no-arbitrage price bounds.
- Sweeps parameters one at a time to show how price and sensitivities move
  as market conditions change (e.g. price vs. volatility, price vs. time to
  maturity), printed as text tables.
- Sweeps parameters two at a time and renders the result as heatmaps
  (`black_scholes_heatmaps.png`), showing joint relationships such as price
  vs. spot & volatility, or price vs. strike & risk-free rate.

## How it works

The script is organized into independent, composable pieces:

1. **Numerical utilities** — the standard normal PDF/CDF, built on
   `math.erf` so no external dependency is needed for the core math.
2. **`BlackScholesOption`** — a frozen dataclass holding the five inputs and
   the option type (`call`/`put`). Its `price()` method implements the
   closed-form Black-Scholes formula; `delta()`, `gamma()`, `vega()`,
   `theta()`, and `rho()` implement the corresponding analytical Greeks.
   Degenerate cases (zero volatility, zero time to maturity) fall back to
   the discounted intrinsic payoff instead of dividing by zero.
3. **Validation routines** — `run_validation_suite()` checks put-call parity
   across several scenarios, compares the analytical delta to a
   finite-difference estimate (bumping the spot price up and down), and
   confirms prices stay within no-arbitrage bounds.
4. **`ScenarioEngine`** — takes a base-case option and re-prices it while
   varying one parameter (`sweep()`) or two parameters jointly
   (`sweep_2d()`), using `dataclasses.replace()` so the base case is never
   mutated. This is what makes the "vary one input, hold the rest constant"
   analysis possible without duplicating pricing logic.
5. **Heatmap visualization** — `plot_price_heatmaps()` takes the 2-D sweeps
   from `sweep_2d()` and renders them with matplotlib as a grid of
   heatmaps, one hue (blue) running light-to-dark to encode price
   magnitude, with the base-case scenario marked on each panel. Matplotlib
   is only needed for this step — the rest of the script runs on the
   standard library alone, and the script degrades gracefully to text-only
   output if matplotlib isn't installed.
6. **`main()`** — runs all of the above in sequence: prices the base case,
   prints its Greeks, runs the validation suite, prints one-parameter
   sensitivity tables, and generates the heatmap figure.

## Usage

```bash
pip install matplotlib numpy   # only needed for the heatmap output
python black_scholes.py
```

This prints the base-case price and Greeks, validation results, sensitivity
tables for each parameter, and saves/opens `black_scholes_heatmaps.png`.

## Extending it

The pricing engine, validation, scenario sweeps, and plotting are
deliberately decoupled. To price options under a different model (e.g. a
binomial tree, Monte Carlo simulation, or one that accounts for dividends),
implement a class with the same `price()`/Greeks interface as
`BlackScholesOption` and it will work with the existing `ScenarioEngine` and
heatmap code unchanged.
