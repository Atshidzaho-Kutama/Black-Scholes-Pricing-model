"""
Black-Scholes Option Pricing Model
==================================

A comprehensive Python implementation of the Black-Scholes quantitative
pricing framework for valuing European-style options under the standard
assumptions of continuous-time financial markets (lognormal asset dynamics,
constant volatility and risk-free rate, no arbitrage, frictionless trading).

The script is organised into modular, reusable components:

    1. Numerical utilities        - standard normal PDF / CDF (via math.erf)
    2. BlackScholesOption         - core pricing engine + Greeks (sensitivities)
    3. Validation routines        - put-call parity and boundary checks
    4. ScenarioEngine             - systematic parameter sweeps across market
                                    scenarios (spot, vol, rates, maturity, strike)
    5. Heatmap visualisation      - 2-D sweeps rendered as heatmaps showing how
                                    the option price responds to pairs of
                                    parameters (requires matplotlib)
    6. Demonstration / analysis   - run from the command line

The design deliberately keeps the pricing engine independent of the scenario
engine so that alternative models (e.g. binomial trees, Monte Carlo, models
with dividends or stochastic volatility) can later be dropped in behind the
same interface for quantitative risk analysis.

The pricing and scenario engines use only the Python standard library;
matplotlib is needed only for the heatmap output (the script degrades
gracefully to text-only if it is missing).

Usage:
    python black_scholes.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Sequence


# ---------------------------------------------------------------------------
# 1. Numerical utilities
# ---------------------------------------------------------------------------

def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function (via the error function)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 2. Core pricing engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlackScholesOption:
    """
    A European option priced under the Black-Scholes framework.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate (annualised).
    sigma : float
        Volatility of the underlying asset (annualised).
    option_type : str
        'call' or 'put'.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"

    def __post_init__(self) -> None:
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Underlying price S and strike K must be positive.")
        if self.T < 0:
            raise ValueError("Time to maturity T cannot be negative.")
        if self.sigma < 0:
            raise ValueError("Volatility sigma cannot be negative.")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

    # -- d1 / d2 terms ------------------------------------------------------

    @property
    def d1(self) -> float:
        num = math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T
        return num / (self.sigma * math.sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * math.sqrt(self.T)

    # -- Pricing ------------------------------------------------------------

    def price(self) -> float:
        """Black-Scholes fair value of the option."""
        # Degenerate cases: expired option or zero volatility -> intrinsic /
        # discounted-forward payoff.
        if self.T == 0 or self.sigma == 0:
            forward_payoff = self.S - self.K * math.exp(-self.r * self.T)
            if self.option_type == "call":
                return max(forward_payoff, 0.0)
            return max(-forward_payoff, 0.0)

        d1, d2 = self.d1, self.d2
        discounted_strike = self.K * math.exp(-self.r * self.T)

        if self.option_type == "call":
            return self.S * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
        return discounted_strike * norm_cdf(-d2) - self.S * norm_cdf(-d1)

    # -- Greeks (analytical sensitivities) -----------------------------------

    def delta(self) -> float:
        """Sensitivity of the option price to the underlying price."""
        if self.option_type == "call":
            return norm_cdf(self.d1)
        return norm_cdf(self.d1) - 1.0

    def gamma(self) -> float:
        """Second-order sensitivity to the underlying price (same for calls/puts)."""
        return norm_pdf(self.d1) / (self.S * self.sigma * math.sqrt(self.T))

    def vega(self) -> float:
        """Sensitivity to volatility, per 1% change in sigma."""
        return self.S * norm_pdf(self.d1) * math.sqrt(self.T) / 100.0

    def theta(self) -> float:
        """Time decay, expressed per calendar day."""
        term1 = -(self.S * norm_pdf(self.d1) * self.sigma) / (2.0 * math.sqrt(self.T))
        if self.option_type == "call":
            term2 = -self.r * self.K * math.exp(-self.r * self.T) * norm_cdf(self.d2)
        else:
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm_cdf(-self.d2)
        return (term1 + term2) / 365.0

    def rho(self) -> float:
        """Sensitivity to the risk-free rate, per 1% change in r."""
        if self.option_type == "call":
            return self.K * self.T * math.exp(-self.r * self.T) * norm_cdf(self.d2) / 100.0
        return -self.K * self.T * math.exp(-self.r * self.T) * norm_cdf(-self.d2) / 100.0

    def greeks(self) -> Dict[str, float]:
        """All first- and second-order sensitivities in one dictionary."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }


# ---------------------------------------------------------------------------
# 3. Validation of theoretical pricing behaviour
# ---------------------------------------------------------------------------

def put_call_parity_gap(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Put-call parity states:  C - P = S - K * exp(-rT).
    Returns the absolute deviation from parity (should be ~0 for a
    correct implementation).
    """
    call = BlackScholesOption(S, K, T, r, sigma, "call").price()
    put = BlackScholesOption(S, K, T, r, sigma, "put").price()
    return abs((call - put) - (S - K * math.exp(-r * T)))


def numerical_delta(option: BlackScholesOption, bump: float = 1e-4) -> float:
    """
    Finite-difference delta, used to cross-validate the analytical Greek.
    """
    up = replace(option, S=option.S * (1 + bump)).price()
    down = replace(option, S=option.S * (1 - bump)).price()
    return (up - down) / (2 * option.S * bump)


def run_validation_suite() -> List[str]:
    """
    Validate theoretical pricing behaviour across a grid of market conditions.
    Returns a list of human-readable results.
    """
    results: List[str] = []
    test_points = [
        (100, 100, 1.0, 0.05, 0.20),
        (100, 120, 0.5, 0.03, 0.35),
        (100, 80, 2.0, 0.07, 0.15),
        (50, 55, 0.25, 0.01, 0.45),
    ]

    max_parity_gap = max(put_call_parity_gap(*p) for p in test_points)
    results.append(
        f"Put-call parity: max deviation across {len(test_points)} scenarios "
        f"= {max_parity_gap:.2e}  ({'PASS' if max_parity_gap < 1e-10 else 'FAIL'})"
    )

    opt = BlackScholesOption(100, 100, 1.0, 0.05, 0.20, "call")
    gap = abs(opt.delta() - numerical_delta(opt))
    results.append(
        f"Analytical vs finite-difference delta: |error| = {gap:.2e}  "
        f"({'PASS' if gap < 1e-6 else 'FAIL'})"
    )

    # A call must be worth at least its discounted intrinsic value and at
    # most the underlying itself (no-arbitrage bounds).
    price = opt.price()
    lower = max(opt.S - opt.K * math.exp(-opt.r * opt.T), 0.0)
    in_bounds = lower <= price <= opt.S
    results.append(
        f"No-arbitrage bounds ({lower:.4f} <= {price:.4f} <= {opt.S:.2f}): "
        f"{'PASS' if in_bounds else 'FAIL'}"
    )
    return results


# ---------------------------------------------------------------------------
# 4. Scenario engine: systematic parameter sweeps
# ---------------------------------------------------------------------------

def linspace(start: float, stop: float, num: int) -> List[float]:
    """Evenly spaced grid of `num` points on [start, stop] (stdlib-only)."""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


class ScenarioEngine:
    """
    Evaluates option prices across a range of market scenarios by
    systematically varying one financial parameter at a time while holding
    the others at their base-case values.
    """

    SWEEPABLE = ("S", "K", "T", "r", "sigma")

    def __init__(self, base_option: BlackScholesOption):
        self.base = base_option

    def sweep(self, parameter: str, values: Iterable[float]) -> List[Dict[str, float]]:
        """
        Re-price the option for each value of `parameter`, returning a list of
        records with the parameter value, price, and key sensitivities.
        """
        if parameter not in self.SWEEPABLE:
            raise ValueError(f"parameter must be one of {self.SWEEPABLE}")

        records = []
        for value in values:
            option = replace(self.base, **{parameter: value})
            records.append({parameter: value, "price": option.price(), **option.greeks()})
        return records

    def sensitivity_report(
        self, sweeps: Dict[str, Sequence[float]]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Run several parameter sweeps and collect the results."""
        return {param: self.sweep(param, values) for param, values in sweeps.items()}

    def sweep_2d(
        self,
        param_x: str,
        values_x: Sequence[float],
        param_y: str,
        values_y: Sequence[float],
        metric: Callable[[BlackScholesOption], float] = BlackScholesOption.price,
    ) -> List[List[float]]:
        """
        Jointly vary two parameters and evaluate `metric` (the option price by
        default) at every grid point. Returns a matrix indexed [row=y][col=x],
        suitable for rendering as a heatmap.
        """
        for p in (param_x, param_y):
            if p not in self.SWEEPABLE:
                raise ValueError(f"parameters must be among {self.SWEEPABLE}")
        if param_x == param_y:
            raise ValueError("param_x and param_y must differ")

        return [
            [metric(replace(self.base, **{param_x: x, param_y: y})) for x in values_x]
            for y in values_y
        ]


# ---------------------------------------------------------------------------
# 5. Heatmap visualisation of pairwise parameter relationships
# ---------------------------------------------------------------------------

# Single-hue sequential ramp (light -> dark blue): darker cell = higher price.
_SEQUENTIAL_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"

PARAMETER_LABELS = {
    "S": "Underlying price S",
    "K": "Strike price K",
    "T": "Maturity T (years)",
    "r": "Risk-free rate r",
    "sigma": "Volatility σ",
}


def plot_price_heatmaps(
    engine: ScenarioEngine,
    pairs: Sequence[tuple],
    grid_points: int = 15,
    output_path: str = "black_scholes_heatmaps.png",
    show: bool = True,
) -> str:
    """
    Render heatmaps of the option price over 2-D parameter grids.

    Each entry of `pairs` is (param_x, (x_lo, x_hi), param_y, (y_lo, y_hi));
    the remaining parameters are held at the engine's base-case values. The
    figure is written to `output_path` and optionally shown interactively.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("sequential_blue", _SEQUENTIAL_BLUES)

    n = len(pairs)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), facecolor=_SURFACE
    )
    axes = [ax for row in ([axes] if nrows == 1 else axes) for ax in row]

    base = engine.base
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, (px, (x_lo, x_hi), py, (y_lo, y_hi)) in zip(axes, pairs):
        xs = linspace(x_lo, x_hi, grid_points)
        ys = linspace(y_lo, y_hi, grid_points)
        prices = engine.sweep_2d(px, xs, py, ys)

        ax.set_facecolor(_SURFACE)
        mesh = ax.pcolormesh(
            xs, ys, prices, cmap=cmap, shading="nearest",
            edgecolors=_SURFACE, linewidth=0.8,  # thin surface gap between cells
        )

        # Mark the base-case scenario if it falls inside this grid.
        bx, by = getattr(base, px), getattr(base, py)
        if x_lo <= bx <= x_hi and y_lo <= by <= y_hi:
            ax.plot(bx, by, "o", markersize=7, markerfacecolor="none",
                    markeredgecolor=_INK_PRIMARY, markeredgewidth=1.4)
            ax.annotate("base case", (bx, by), xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color=_INK_SECONDARY)

        ax.set_xlabel(PARAMETER_LABELS[px], fontsize=10, color=_INK_SECONDARY)
        ax.set_ylabel(PARAMETER_LABELS[py], fontsize=10, color=_INK_SECONDARY)
        ax.set_title(
            f"{PARAMETER_LABELS[py]} vs {PARAMETER_LABELS[px]}",
            fontsize=11, color=_INK_PRIMARY, pad=10,
        )
        ax.tick_params(colors=_INK_MUTED, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(mesh, ax=ax, shrink=0.9)
        cbar.set_label(f"{base.option_type.capitalize()} price", fontsize=9,
                       color=_INK_SECONDARY)
        cbar.ax.tick_params(colors=_INK_MUTED, labelsize=8)
        cbar.outline.set_visible(False)

    fig.suptitle(
        "Black-Scholes option price across pairs of market parameters",
        fontsize=13, color=_INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 6. Demonstration and analysis
# ---------------------------------------------------------------------------

def format_table(records: List[Dict[str, float]], columns: Sequence[str]) -> str:
    """Render a list of records as a fixed-width text table."""
    header = "".join(f"{c:>12}" for c in columns)
    lines = [header, "-" * len(header)]
    for rec in records:
        lines.append("".join(f"{rec[c]:>12.4f}" for c in columns))
    return "\n".join(lines)


def main() -> None:
    print("=" * 72)
    print("BLACK-SCHOLES EUROPEAN OPTION PRICING MODEL")
    print("=" * 72)

    # ---- Base-case market scenario ----------------------------------------
    base = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
    print("\nBase-case parameters:")
    for name, value in base.items():
        print(f"    {name:<6} = {value}")

    call = BlackScholesOption(**base, option_type="call")
    put = BlackScholesOption(**base, option_type="put")

    print(f"\nCall price: {call.price():.4f}")
    print(f"Put  price: {put.price():.4f}")

    print("\nGreeks (call / put):")
    call_greeks, put_greeks = call.greeks(), put.greeks()
    for greek in call_greeks:
        print(f"    {greek:<6} {call_greeks[greek]:>10.4f} / {put_greeks[greek]:>10.4f}")

    # ---- Validation of theoretical behaviour -------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION OF THEORETICAL PRICING BEHAVIOUR")
    print("=" * 72)
    for line in run_validation_suite():
        print("  " + line)

    # ---- Systematic scenario analysis --------------------------------------
    print("\n" + "=" * 72)
    print("SCENARIO ANALYSIS: OPTION PRICE ACROSS MARKET CONDITIONS (CALL)")
    print("=" * 72)

    engine = ScenarioEngine(call)
    sweeps = {
        "S": linspace(60, 140, 9),        # underlying asset price
        "sigma": linspace(0.05, 0.60, 9),  # volatility
        "r": linspace(0.00, 0.10, 9),      # risk-free interest rate
        "T": linspace(0.10, 2.00, 9),      # time to maturity
        "K": linspace(60, 140, 9),         # strike price
    }
    labels = {
        "S": "Underlying asset price",
        "sigma": "Volatility",
        "r": "Risk-free interest rate",
        "T": "Time to maturity (years)",
        "K": "Strike price",
    }

    report = engine.sensitivity_report(sweeps)
    for param, records in report.items():
        print(f"\n--- Varying {labels[param]} ({param}) ---")
        print(format_table(records, [param, "price", "delta", "gamma", "vega", "theta"]))

    # ---- Interpretation -----------------------------------------------------
    print("\n" + "=" * 72)
    print("OBSERVED PRICING BEHAVIOUR (consistent with theory)")
    print("=" * 72)
    print("""  * Call value increases monotonically with the underlying price (S),
    volatility (sigma), the risk-free rate (r), and time to maturity (T),
    and decreases with the strike price (K).
  * Delta tends to 1 deep in-the-money and 0 deep out-of-the-money.
  * Gamma and vega peak near-the-money, where the option's moneyness is
    most uncertain.
  * Theta is negative: option value decays as expiry approaches.""")

    # ---- Heatmaps: pairwise parameter relationships -------------------------
    print("\n" + "=" * 72)
    print("HEATMAPS: PAIRWISE PARAMETER RELATIONSHIPS")
    print("=" * 72)
    pairs = [
        ("S", (60.0, 140.0), "sigma", (0.05, 0.60)),
        ("S", (60.0, 140.0), "T", (0.10, 2.00)),
        ("T", (0.10, 2.00), "sigma", (0.05, 0.60)),
        ("K", (60.0, 140.0), "r", (0.00, 0.10)),
    ]
    try:
        path = plot_price_heatmaps(engine, pairs)
    except ImportError:
        print("  matplotlib is not installed - skipping heatmaps.")
        print("  Install it with:  pip install matplotlib")
    else:
        print(f"  Heatmap figure saved to: {path}")
        print("  Darker blue = higher call price. The open circle marks the")
        print("  base-case scenario on each panel.")


if __name__ == "__main__":
    main()
