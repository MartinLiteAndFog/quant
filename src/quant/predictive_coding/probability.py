from __future__ import annotations
import math

_SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def compute_probabilities(
    mu: dict[int, float],
    sigma: dict[int, float],
    price: float,
) -> dict[int, dict[str, float]]:
    """Compute p_up, price levels, and ±1σ bands per horizon."""
    out: dict[int, dict[str, float]] = {}
    for h in mu:
        m = mu[h]
        s = sigma[h]
        z = m / s if s > 1e-15 else 0.0
        p_up = _normal_cdf(z)
        out[h] = {
            "mu": m,
            "sigma": s,
            "z": z,
            "p_up": p_up,
            "price_level": price * math.exp(m),
            "price_upper": price * math.exp(m + s),
            "price_lower": price * math.exp(m - s),
        }
    return out
