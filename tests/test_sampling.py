"""Randomized membership tests against an independent reference classifier.

The reference (`ref_inside`) decides point membership of the *closed* CSG
solid directly from the tree definition with exact Fraction arithmetic.  It
shares no code with the implementation under test; expected values are never
produced by calling `ray_intervals`.

Per the task contract, sampling only asserts membership of interval interiors
(and non-membership of gaps / zero-measure touches); exact endpoint values
are verified analytically in test_analytic.py.
"""

import random
import unittest
from fractions import Fraction

from csg_rays import (
    Box,
    Difference,
    Intersection,
    Rotate,
    Ray,
    Scale,
    Sphere,
    Transform,
    Translate,
    Union,
    ray_intervals,
)


# ---------------------------------------------------------------------------
# independent reference: exact point membership of the closed CSG solid
# ---------------------------------------------------------------------------

def _in_sphere(node, p, closed):
    d2 = sum((p[i] - node.center[i]) ** 2 for i in range(3))
    r2 = node.radius**2
    return d2 <= r2 if closed else d2 < r2


def _in_box(node, p, closed):
    if closed:
        return all(node.mins[i] <= p[i] <= node.maxs[i] for i in range(3))
    return all(node.mins[i] < p[i] < node.maxs[i] for i in range(3))


def ref_inside(node, p, closed=False):
    """True iff point p (Fraction triple) lies in the solid.

    ``closed=False`` gives the strict interior; ``closed=True`` the closed
    solid (used for subtractors, matching regularized difference).
    """
    if isinstance(node, Sphere):
        return _in_sphere(node, p, closed)
    if isinstance(node, Box):
        return _in_box(node, p, closed)
    if isinstance(node, Union):
        return any(ref_inside(c, p, closed) for c in node.children)
    if isinstance(node, Intersection):
        return all(ref_inside(c, p, closed) for c in node.children)
    if isinstance(node, Difference):
        if not ref_inside(node.positive, p, closed):
            return False
        return not any(ref_inside(s, p, closed=True) for s in node.subtractors)
    if isinstance(node, Transform):
        # Independent evaluation: map the world point into child
        # coordinates with a hand-rolled exact 3x3 solve (this reference
        # never calls the implementation's inverse cache).
        q = tuple(p[i] - node.translation[i] for i in range(3))
        local = _solve3(node.linear, q)
        return ref_inside(node.child, local, closed)
    raise AssertionError("unknown node")


def _det3(m):
    return (
        m[0][0] * (m[0+1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve3(m, q):
    """Exact solution of m @ x = q via Cramer's rule (m is invertible)."""
    det = _det3(m)

    def rep(col, i):
        rows = [list(m[r]) for r in range(3)]
        for r in range(3):
            rows[r][col] = q[r]
        return _det3(rows) / det

    return (rep(0, 0), rep(1, 1), rep(2, 2))


def _point_at(origin, direction, t_float):
    t = Fraction(t_float)
    return tuple(Fraction(origin[i]) + t * Fraction(direction[i]) for i in range(3))


# Exact nudge applied to interior samples: membership is sampled at generic
# points only.  A sample that lands exactly on a measure-zero event (e.g. a
# tangent touch swallowed inside a solid) is on the boundary by definition,
# so the strict-interior reference cannot be checked there.  1e-7 is far
# below any gap between distinct events in these small-integer scenes and
# far above any rounding effect (all reference arithmetic is exact).
_NUDGE = Fraction(1, 10**7)


def _interior_point(origin, direction, t_float):
    t = Fraction(t_float) + _NUDGE
    return tuple(Fraction(origin[i]) + t * Fraction(direction[i]) for i in range(3))


# ---------------------------------------------------------------------------
# deterministic random scene generation
# ---------------------------------------------------------------------------

def _random_tree(rng, depth, id_counter):
    if depth == 0 or rng.random() < 0.45:
        leaf_id = f"L{next(id_counter)}"
        if rng.random() < 0.5:
            center = tuple(rng.randint(-3, 3) for _ in range(3))
            return Sphere(center, rng.randint(1, 3), leaf_id)
        mins = tuple(rng.randint(-3, 2) for _ in range(3))
        maxs = tuple(m + rng.randint(1, 3) for m in mins)
        return Box(mins, maxs, leaf_id)
    op = rng.choice("uid")
    count = rng.randint(2, 3)
    subs = [_random_tree(rng, depth - 1, id_counter) for _ in range(count)]
    if op == "u":
        return Union(subs)
    if op == "i":
        return Intersection(subs)
    return Difference(subs[0], *subs[1:])


def _random_transform(rng, child):
    kind = rng.randrange(4)
    if kind == 0:
        offset = tuple(rng.randint(-3, 3) for _ in range(3))
        return Translate(child, offset)
    if kind == 1:
        factors = tuple(rng.choice((-2, -1, 1, 2, 3)) for _ in range(3))
        return Scale(child, factors)
    if kind == 2:
        # 90-degree rotations about a coordinate axis (exact rationals).
        axis = rng.randrange(3)
        sign = rng.choice((-1, 1))
        rows = [[0, 0, 0] for _ in range(3)]
        rows[axis][axis] = 1
        a, b = (axis + 1) % 3, (axis + 2) % 3
        rows[a][a] = 0
        rows[a][b] = -sign
        rows[b][a] = sign
        rows[b][b] = 0
        return Rotate(child, tuple(tuple(row) for row in rows))
    # shear (non-orthogonal, still invertible)
    axis = rng.randrange(3)
    others = [x for x in range(3) if x != axis]
    i, j = rng.choice(others), rng.choice(others)
    rows = [[1 if a == b else 0 for b in range(3)] for a in range(3)]
    rows[i][j] = rng.choice((-2, -1, 1, 2))
    return Transform(child, linear=tuple(tuple(row) for row in rows))


def _maybe_wrap(rng, tree, depth):
    if depth > 0 and rng.random() < 0.5:
        return _maybe_wrap(rng, _random_transform(rng, tree), depth - 1)
    return tree


def _random_ray(rng):
    origin = tuple(rng.randint(-5, 5) for _ in range(3))
    while True:
        direction = tuple(rng.randint(-3, 3) for _ in range(3))
        if any(direction):
            return Ray(origin, direction)


class TestRandomizedMembership(unittest.TestCase):
    def test_membership_against_independent_reference(self):
        rng = random.Random(20260922)
        cases = 0
        for _ in range(300):
            id_counter = iter(range(1000))
            tree = _random_tree(rng, depth=3, id_counter=id_counter)
            tree = _maybe_wrap(rng, tree, 2)
            ray = _random_ray(rng)
            res = ray_intervals(tree, ray)
            cases += 1

            # intervals sorted, non-empty and pairwise disjoint
            for iv in res.intervals:
                self.assertLess(iv.t0, iv.t1)
            for prev, nxt in zip(res.intervals, res.intervals[1:]):
                self.assertLess(prev.t1, nxt.t0)

            # interval interiors are solid (sampled at generic interior points)
            for iv in res.intervals:
                for frac in (0.19, 0.5, 0.83):
                    t = iv.t0 + (iv.t1 - iv.t0) * frac
                    p = _interior_point(ray.origin, ray.direction, t)
                    self.assertTrue(
                        ref_inside(tree, p),
                        f"interior sample t={t} not inside; interval={iv}",
                    )

            # gaps between/around intervals are not solid interior
            gap_ts = []
            if res.intervals:
                gap_ts.append(res.intervals[0].t0 - 1.31)
                gap_ts.append(res.intervals[-1].t1 + 1.77)
                for prev, nxt in zip(res.intervals, res.intervals[1:]):
                    gap_ts.append((prev.t1 + nxt.t0) / 2.0)
            else:
                gap_ts.append(0.0)
                gap_ts.append(-2.5)
            for t in gap_ts:
                p = _point_at(ray.origin, ray.direction, t)
                self.assertFalse(
                    ref_inside(tree, p), f"gap sample t={t} unexpectedly inside"
                )

            # touches are zero-measure: their points are not solid interior
            for touch in res.touches:
                for t in {touch.t0, touch.t1, (touch.t0 + touch.t1) / 2.0}:
                    p = _point_at(ray.origin, ray.direction, t)
                    self.assertFalse(
                        ref_inside(tree, p), f"touch t={t} unexpectedly interior"
                    )

            # every boundary carries sources; representative is lexicographically first
            for boundary in res.boundaries:
                self.assertTrue(boundary.sources)
                keys = [(s.leaf_id, s.face) for s in boundary.sources]
                self.assertEqual(keys, sorted(keys))
                self.assertIs(boundary.source, boundary.sources[0])

            # nearest_boundary is the smallest boundary with t > 0
            forward = [b for b in res.boundaries if b.t > 0.0]
            if forward:
                self.assertIsNotNone(res.nearest_boundary)
                self.assertEqual(
                    res.nearest_boundary.t, min(b.t for b in forward)
                )
            # (when no boundary has t > 0 the analytic tests cover None)

        self.assertEqual(cases, 300)


if __name__ == "__main__":
    unittest.main()
