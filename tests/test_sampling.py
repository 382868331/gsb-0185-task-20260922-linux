"""Randomized membership tests against an independent reference classifier.

The reference (`ref_inside`) decides point membership of the *closed* CSG
solid directly from the tree definition with exact Fraction arithmetic.  It
shares no code with the implementation under test; expected values are never
produced by calling `ray_intervals`.

Per the task contract, sampling only asserts membership of interval interiors
(and non-membership of gaps / zero-measure touches); exact endpoint values
are verified analytically in test_analytic.py.
"""

import math
import random
import unittest
from fractions import Fraction

from csg_rays import (
    Box,
    Difference,
    Instance,
    Intersection,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)


# ---------------------------------------------------------------------------
# independent exact affine helpers (not imported from the package core)
# ---------------------------------------------------------------------------

def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inv3(m):
    d = _det3(m)

    def minor(i, j):
        rows = [r for r in range(3) if r != i]
        cols = [c for c in range(3) if c != j]
        return (
            m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
            - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]
        )

    return [
        [((-1) ** (i + j)) * minor(j, i) / d for j in range(3)]
        for i in range(3)
    ]


def _mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


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


def ref_inside(node, p, closed=False, inverse_cache=None):
    """True iff point p (Fraction triple) lies in the solid.

    ``closed=False`` gives the strict interior; ``closed=True`` the closed
    solid (used for subtractors, matching regularized difference).  Instance
    references map the point through the exact inverse affine.
    """
    if inverse_cache is None:
        inverse_cache = {}
    if isinstance(node, Sphere):
        return _in_sphere(node, p, closed)
    if isinstance(node, Box):
        return _in_box(node, p, closed)
    if isinstance(node, Union):
        return any(ref_inside(c, p, closed, inverse_cache) for c in node.children)
    if isinstance(node, Intersection):
        return all(ref_inside(c, p, closed, inverse_cache) for c in node.children)
    if isinstance(node, Difference):
        if not ref_inside(node.positive, p, closed, inverse_cache):
            return False
        return not any(
            ref_inside(s, p, True, inverse_cache) for s in node.subtractors
        )
    if isinstance(node, Instance):
        inv = inverse_cache.get(id(node))
        if inv is None:
            inv = (_inv3(node.matrix), node.translation_inv)
            inverse_cache[id(node)] = inv
        b, v = inv
        p_local = [v[i] + sum(b[i][j] * p[j] for j in range(3)) for i in range(3)]
        return ref_inside(node.child, p_local, closed, inverse_cache)
    raise AssertionError("unknown node")


def _point_at(origin, direction, t_float):
    t = Fraction(t_float)
    return tuple(Fraction(origin[i]) + t * Fraction(direction[i]) for i in range(3))


def _hit_point(ray, t_float):
    """World hit point as floats (for independent numeric checks)."""
    return tuple(
        float(ray.origin[i]) + t_float * float(ray.direction[i]) for i in range(3)
    )


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


# ---------------------------------------------------------------------------
# randomized scenes containing Instance references
# ---------------------------------------------------------------------------

# A fixed pool of small-integer invertible matrices (scale / shear / mirror /
# 90-degree rotations / composites).  Determinants are all +-1 or +-2.
_MATRIX_POOL = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 3, 0), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 1, 1), (0, 0, 1)),
]


def _random_matrix(rng):
    return _MATRIX_POOL[rng.randrange(len(_MATRIX_POOL))]


def _random_subtree(rng, id_counter):
    leaf_id = f"L{next(id_counter)}"
    kind = rng.random()
    if kind < 0.4:
        center = tuple(rng.randint(-2, 2) for _ in range(3))
        return Sphere(center, rng.randint(1, 2), leaf_id)
    if kind < 0.8:
        mins = tuple(rng.randint(-2, 1) for _ in range(3))
        maxs = tuple(m + rng.randint(1, 2) for m in mins)
        return Box(mins, maxs, leaf_id)
    s1 = Sphere(tuple(rng.randint(-1, 1) for _ in range(3)), 1,
                f"L{next(id_counter)}")
    s2 = Sphere(tuple(rng.randint(-1, 1) for _ in range(3)), 1,
                f"L{next(id_counter)}")
    op = rng.choice("ud")
    return Difference(s1, s2) if op == "d" else Union([s1, s2])


def _random_instance_scene(rng, id_counter):
    shared = _random_subtree(rng, id_counter)
    n_refs = rng.randint(1, 3)
    refs = []
    for k in range(n_refs):
        m = _random_matrix(rng)
        t = tuple(rng.randint(-6, 6) for _ in range(3))
        refs.append(Instance(shared, m, t, f"r{k}"))
    if len(refs) == 1:
        return refs[0]
    wrapper = rng.choice(("union", "diff"))
    if wrapper == "union" or rng.random() < 0.5:
        return Union(refs)
    # Difference needs an explicit positive: first reference minus the rest
    return Difference(refs[0], *refs[1:])


def _follow_path(tree, path):
    """Independently walk to the referenced leaf via the instance path;
    return (leaf, composed B rows, composed v, polarity)."""
    def walk(node, b, v, pol, depth):
        if isinstance(node, Instance):
            b2 = _inv3(node.matrix)
            v2 = [
                -sum(b2[i][j] * node.translation[j] for j in range(3))
                for i in range(3)
            ]
            bn = [
                [sum(b2[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]
            vn = [v2[i] + sum(b2[i][k] * v[k] for k in range(3)) for i in range(3)]
            assert node.instance_id == path[depth]
            return walk(node.child, bn, vn, pol, depth + 1)
        if isinstance(node, Union):
            for c in node.children:
                try:
                    return walk(c, b, v, pol, depth)
                except AssertionError:
                    pass
            raise AssertionError("path not found")
        if isinstance(node, Intersection):
            for c in node.children:
                try:
                    return walk(c, b, v, pol, depth)
                except AssertionError:
                    pass
            raise AssertionError("path not found")
        if isinstance(node, Difference):
            try:
                return walk(node.positive, b, v, pol, depth)
            except AssertionError:
                pass
            for s in node.subtractors:
                try:
                    return walk(s, b, v, -pol, depth)
                except AssertionError:
                    pass
            raise AssertionError("path not found")
        if depth == len(path) and isinstance(node, (Sphere, Box)):
            return node, b, v, pol
        raise AssertionError("path not found")

    eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return walk(tree, eye, [0, 0, 0], 1, 0)


class TestRandomizedInstanceMembership(unittest.TestCase):
    def test_instances_against_independent_reference(self):
        rng = random.Random(20260925)
        cases = 0
        for _ in range(150):
            id_counter = iter(range(10000))
            tree = _random_instance_scene(rng, id_counter)
            origin = tuple(rng.randint(-8, 8) for _ in range(3))
            direction = tuple(rng.randint(-2, 2) for _ in range(3))
            if not any(direction):
                direction = (1, 0, 0)
            ray = Ray(origin, direction)
            res = ray_intervals(tree, ray)
            cases += 1

            for iv in res.intervals:
                self.assertLess(iv.t0, iv.t1)
                for frac in (0.23, 0.57, 0.84):
                    t = iv.t0 + (iv.t1 - iv.t0) * frac
                    p = _interior_point(ray.origin, ray.direction, t)
                    self.assertTrue(
                        ref_inside(tree, p),
                        f"instance interior sample t={t} not inside",
                    )
            for prev, nxt in zip(res.intervals, res.intervals[1:]):
                self.assertLess(prev.t1, nxt.t0)
                p = _point_at(
                    ray.origin, ray.direction, (prev.t1 + nxt.t0) / 2.0
                )
                self.assertFalse(ref_inside(tree, p))
            if res.intervals:
                p = _point_at(ray.origin, ray.direction,
                              res.intervals[0].t0 - 1.5)
                self.assertFalse(ref_inside(tree, p))
            for touch in res.touches:
                p = _point_at(ray.origin, ray.direction, touch.t0)
                self.assertFalse(ref_inside(tree, p))

            # independent normal-direction check: numeric central difference
            # of the transformed implicit surface at every boundary hit
            h = 1e-6
            for b in res.boundaries:
                for s in b.sources:
                    leaf, bm, vm, pol = _follow_path(tree, s.path)
                    if isinstance(leaf, Sphere):
                        # F(p) = |B p + v - c|^2 - r^2
                        def surf(pf):
                            q = [
                                vm[i]
                                + sum(bm[i][j] * pf[j] for j in range(3))
                                - leaf.center[i]
                                for i in range(3)
                            ]
                            return (
                                float(sum(x * x for x in q))
                                - float(leaf.radius) ** 2
                            )

                        pf = _hit_point(ray, b.t)
                        grad = []
                        for axis in range(3):
                            pp = list(pf)
                            pm = list(pf)
                            pp[axis] += h
                            pm[axis] -= h
                            grad.append((surf(pp) - surf(pm)) / (2 * h))
                        gn = math.sqrt(sum(g * g for g in grad))
                        expected = tuple(g / gn * pol for g in grad)
                        dot = sum(
                            s.normal[i] * expected[i] for i in range(3)
                        )
                        self.assertGreater(
                            dot, 0.9999,
                            f"sphere normal disagrees with numeric gradient "
                            f"at t={b.t} path={s.path}",
                        )
                    self.assertAlmostEqual(
                        sum(c * c for c in s.normal), 1.0, delta=1e-12
                    )

            # sources from two references are distinguishable by path
            for b in res.boundaries:
                keys = [(s.path, s.leaf_id, s.face) for s in b.sources]
                self.assertEqual(keys, sorted(keys))
                self.assertEqual(len(keys), len(set(keys)))

        self.assertEqual(cases, 150)

    def test_nested_instances_against_reference(self):
        rng = random.Random(424242)
        id_counter = iter(range(1000))
        cases = 0
        for _ in range(60):
            shared = _random_subtree(rng, id_counter)
            inner = Instance(shared, _random_matrix(rng),
                             tuple(rng.randint(-3, 3) for _ in range(3)), "in")
            outer = Instance(inner, _random_matrix(rng),
                             tuple(rng.randint(-5, 5) for _ in range(3)), "out")
            ray = Ray(
                tuple(rng.randint(-8, 8) for _ in range(3)),
                (1, rng.randint(-1, 1), rng.randint(-1, 1)),
            )
            res = ray_intervals(outer, ray)
            cases += 1
            for iv in res.intervals:
                for frac in (0.3, 0.7):
                    t = iv.t0 + (iv.t1 - iv.t0) * frac
                    p = _interior_point(ray.origin, ray.direction, t)
                    self.assertTrue(ref_inside(outer, p))
            for b in res.boundaries:
                for s in b.sources:
                    self.assertEqual(s.path, ("out", "in"))
        self.assertEqual(cases, 60)


if __name__ == "__main__":
    unittest.main()
