"""Exact event-parameter arithmetic for csg_rays.

Every event parameter (ray parameter t of a leaf boundary) is either

* a ``fractions.Fraction`` -- exact (all box events, rational sphere roots,
  tangent points), or
* a :class:`Root` -- one exact root of a quadratic with Fraction coefficients
  whose discriminant is positive but not a rational square.

Comparisons are exact wherever algebra allows:

* Fraction vs Fraction: direct.
* Fraction vs Root: decided exactly from the sign of the quadratic at the
  Fraction (exact Fraction arithmetic, no rounding).
* Root vs Root: equality is decided exactly (two quadratics over Q share an
  irrational root iff they are proportional).  Ordering of distinct roots
  uses Decimal with escalating precision (60 -> 120 -> 240 digits); if the
  gap still cannot be certified, :class:`CertificationBudgetExhausted` is
  raised instead of silently guessing.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cmp_to_key

from ._types import CertificationBudgetExhausted

#: Decimal precision tiers used to certify the order of distinct roots.
PRECISION_TIERS = (60, 120, 240)


def frac_to_decimal(value):
    """Convert a Fraction to a Decimal under the current context precision."""
    return Decimal(value.numerator) / Decimal(value.denominator)


class Root:
    """Exact root of ``a*t**2 + b*t + c = 0`` (Fractions, a > 0).

    ``sign`` is -1 for the smaller root and +1 for the larger one.  The
    discriminant must be positive and not a rational square, so the root is
    irrational and never equal to any Fraction.
    """

    __slots__ = ("a", "b", "c", "sign", "_cache")

    def __init__(self, a, b, c, sign):
        self.a = a
        self.b = b
        self.c = c
        self.sign = sign
        self._cache = {}

    def __repr__(self):
        return f"Root(a={self.a}, b={self.b}, c={self.c}, sign={self.sign:+d})"

    def monic(self):
        """(b/a, c/a): identifies the minimal polynomial over Q."""
        return (self.b / self.a, self.c / self.a)

    def decimal(self, prec):
        """Decimal approximation computed at ``prec`` digits (cached)."""
        value = self._cache.get(prec)
        if value is None:
            with localcontext() as ctx:
                ctx.prec = prec
                da = frac_to_decimal(self.a)
                db = frac_to_decimal(self.b)
                dc = frac_to_decimal(self.c)
                sqrt_disc = (db * db - 4 * da * dc).sqrt()
                value = (-db + (sqrt_disc if self.sign > 0 else -sqrt_disc)) / (2 * da)
            self._cache[prec] = value
        return value


def _cmp_fraction_fraction(x, y):
    return (x > y) - (x < y)


def _cmp_fraction_root(f, r):
    """Compare Fraction f against Root r; returns -1/1 (never 0)."""
    # s = a*(f - r-)*(f - r+); s != 0 because r is irrational.
    s = r.a * f * f + r.b * f + r.c
    vertex = -r.b / (2 * r.a)
    if r.sign < 0:
        # f < r-  iff  f is left of the vertex and outside (f-r-)(f-r+) > 0
        less = (s > 0) and (f < vertex)
    else:
        # f < r+  iff  f is left of the vertex, or between the roots
        less = (f < vertex) or (s < 0)
    return -1 if less else 1


def _cmp_root_root(r1, r2):
    if r1.monic() == r2.monic():
        # Same minimal polynomial: roots are r- < r+ (or identical).
        return (r1.sign > r2.sign) - (r1.sign < r2.sign)
    # Distinct irreducible quadratics over Q share no root, so r1 != r2;
    # only the order remains to be certified numerically.
    for prec in PRECISION_TIERS:
        d1 = r1.decimal(prec + 10)
        d2 = r2.decimal(prec + 10)
        if d1 == d2:
            continue  # indistinguishable at this precision: escalate
        gap = abs(d1 - d2)
        magnitude = max(abs(d1), abs(d2), Decimal(1))
        if gap > magnitude * (Decimal(10) ** (8 - prec)):
            return -1 if d1 < d2 else 1
        # gap too small to certify at this precision: escalate
    raise CertificationBudgetExhausted(
        "cannot certify the order of two distinct quadratic roots within the "
        f"{PRECISION_TIERS[-1]}-digit Decimal budget "
        f"(polynomials {r1.a}t^2+{r1.b}t+{r1.c} and {r2.a}t^2+{r2.b}t+{r2.c})"
    )


def cmp_tval(x, y):
    """Total order over event parameters (Fraction | Root).  May raise
    CertificationBudgetExhausted for pathologically close distinct roots."""
    x_frac = isinstance(x, Fraction)
    y_frac = isinstance(y, Fraction)
    if x_frac and y_frac:
        return _cmp_fraction_fraction(x, y)
    if x_frac:
        return _cmp_fraction_root(x, y)
    if y_frac:
        return -_cmp_fraction_root(y, x)
    return _cmp_root_root(x, y)


def sorted_unique(tvals):
    """Sort event parameters and merge exact duplicates (coincident events)."""
    memo = {}

    def cmp(x, y):
        key = (id(x), id(y))
        cached = memo.get(key)
        if cached is None:
            cached = cmp_tval(x, y)
            memo[key] = cached
            memo[(id(y), id(x))] = -cached
        return cached

    ordered = sorted(tvals, key=cmp_to_key(cmp))
    unique = []
    for tval in ordered:
        if unique and cmp_tval(unique[-1], tval) == 0:
            continue  # coincident event parameters: one shared cluster
        unique.append(tval)
    return unique


def tval_to_float(tval):
    """Single rounding at the output boundary (see README error policy)."""
    if isinstance(tval, Fraction):
        return float(tval)
    return float(tval.decimal(80))
