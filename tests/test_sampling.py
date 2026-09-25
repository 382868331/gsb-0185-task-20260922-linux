"""Randomized membership tests against an independent reference classifier.

The reference (`ref_inside`) decides point membership of the *closed* CSG
solid directly from the tree definition with exact Fraction arithmetic.  It
shares no code with the implementation under test; expected values are never
produced by calling `ray_intervals`.

Per the task contract, sampling only asserts membership of interval interiors
(and non-membership of gaps / zero-measure touches); exact endpoint values
are verified analytically in test_analytic.py and test_instances.py.
"""

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
    if isinstance(node, Instance):
        q = _ref_apply_inverse(node, p)
        return ref_inside(node.child, q, closed)
    raise AssertionError("unknown node")


def _ref_apply_inverse(inst, p):
    """Independent exact q = L^-1 (p - t) for the test oracle only."""
    L = inst.linear
    t = inst.translation
    v = (p[0] - t[0], p[1] - t[1], p[2] - t[2])
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = L
    c00 = a11*a22 - a12*a21
    c01 = a12*a20 - a10*a22
    c02 = a10*a21 - a11*a20
    det = a00*c00 + a01*c01 + a02*c02
    adj = (c00, a02*a21-a01*a22, a01*a12-a02*a11,
           c01, a00*a22-a02*a20, a02*a10-a00*a12,
           c02, a01*a20-a00*a21, a00*a11-a01*a10)
    return (
        (adj[0]*v[0] + adj[1]*v[1] + adj[2]*v[2]) / det,
        (adj[3]*v[0] + adj[4]*v[1] + adj[5]*v[2]) / det,
        (adj[6]*v[0] + adj[7]*v[1] + adj[8]*v[2]) / det,
    )


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


def _random_ray(rng):
    origin = tuple(rng.randint(-5, 5) for _ in range(3))
    while True:
        direction = tuple(rng.randint(-3, 3) for _ in range(3))
        if any(direction):
            return Ray(origin, direction)


def _collect_leaf_frames(node, A=None, b=None, path=(), polarity=1, out=None):
    """All leaf occurrences as {(path, leaf_id): (node, A, b, polarity)}.

    Independent test oracle: threads the composed affine frame and the
    Difference polarity itself, sharing no code with the implementation.
    """
    if A is None:
        A = _IDENT
        b = _ZERO
        out = {}
    if isinstance(node, (Box, Sphere)):
        out[(path, node.leaf_id)] = (node, A, b, polarity)
    elif isinstance(node, (Union, Intersection)):
        for child in node.children:
            _collect_leaf_frames(child, A, b, path, polarity, out)
    elif isinstance(node, Difference):
        _collect_leaf_frames(node.positive, A, b, path, polarity, out)
        for sub in node.subtractors:
            _collect_leaf_frames(sub, A, b, path, -polarity, out)
    elif isinstance(node, Instance):
        A2 = tuple(
            sum(A[3*r+k] * node.linear[k*3+c] for k in range(3))
            for r in range(3) for c in range(3)
        )
        At = tuple(
            sum(A[3*r+k] * node.translation[k] for k in range(3))
            for r in range(3)
        )
        b2 = tuple(At[i] + b[i] for i in range(3))
        _collect_leaf_frames(
            node.child, A2, b2, path + (node.instance_id,), polarity, out
        )
    return out


def _collect_box_frames(node, A=None, b=None, out=None):
    """Backwards-compatible (box, A, b) tuples for the face-ride filter."""
    frames = _collect_leaf_frames(node)
    return [(leaf, A2, b2) for (leaf, A2, b2, _pol) in frames.values()
            if isinstance(leaf, Box)]


_IDENT = tuple(1 if i // 3 == i % 3 else 0 for i in range(9))
_ZERO = (Fraction(0), Fraction(0), Fraction(0))


def _inverse3(A):
    a = A
    c00, c01, c02 = a[4]*a[8]-a[5]*a[7], a[5]*a[6]-a[3]*a[8], a[3]*a[7]-a[4]*a[6]
    det = a[0]*c00 + a[1]*c01 + a[2]*c02
    adj = (c00, a[2]*a[7]-a[1]*a[8], a[1]*a[5]-a[2]*a[4],
           c01, a[0]*a[8]-a[2]*a[6], a[2]*a[3]-a[0]*a[5],
           c02, a[1]*a[6]-a[0]*a[7], a[0]*a[4]-a[1]*a[3])
    return tuple(v / det for v in adj)


def _ray_rides_box_face(tree, ray):
    """True if the ray lies exactly in a (transformed) box face plane.

    Samples on such a ray stay on a measure-zero face for every t, so the
    t-nudge cannot move them into generic interior; the strict-interior
    reference cannot classify them.  Random tests reject these rays.
    """
    for box, A, b in _collect_box_frames(tree):
        Ainv = _inverse3(A)
        shifted = tuple(ray.origin[i] - b[i] for i in range(3))
        o = tuple(
            sum(Ainv[3*r+c] * shifted[c] for c in range(3)) for r in range(3)
        )
        d = tuple(
            sum(Ainv[3*r+c] * ray.direction[c] for c in range(3))
            for r in range(3)
        )
        for i in range(3):
            if d[i] == 0 and (o[i] == box.mins[i] or o[i] == box.maxs[i]):
                return True
    return False


def _random_generic_ray(rng, tree):
    for _ in range(100):
        ray = _random_ray(rng)
        if not _ray_rides_box_face(tree, ray):
            return ray
    # give up on filtering only if geometry is extremely degenerate
    return _random_ray(rng)


# ---------------------------------------------------------------------------
# deterministic random DAG generation with shared instanced subtrees
# ---------------------------------------------------------------------------

def _random_invertible_matrix(rng):
    """Random small integer invertible matrix (sometimes a reflection)."""
    for _ in range(20):
        rows = [[rng.randint(-2, 2) for _ in range(3)] for _ in range(3)]
        a = [v for row in rows for v in row]
        det = (
            a[0]*(a[4]*a[8]-a[5]*a[7])
            - a[1]*(a[3]*a[8]-a[5]*a[6])
            + a[2]*(a[3]*a[7]-a[4]*a[6])
        )
        if det != 0:
            return rows
    # fallback: diagonal with non-zero small entries
    diag = [rng.choice((-2, -1, 1, 2)) for _ in range(3)]
    return [[diag[0], 0, 0], [0, diag[1], 0], [0, 0, diag[2]]]


def _random_plain_tree(rng, depth, id_counter):
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
    subs = [_random_plain_tree(rng, depth - 1, id_counter) for _ in range(count)]
    if op == "u":
        return Union(subs)
    if op == "i":
        return Intersection(subs)
    return Difference(subs[0], *subs[1:])


def _random_instanced_scene(rng):
    """Build a root tree that references shared subtree prototypes via
    Instances (unique leaf/instance ids, acyclic by construction)."""
    id_counter = iter(range(10**6))
    inst_counter = iter(range(10**6))

    # shared library: 2 plain prototypes and one prototype that itself
    # instances the first two (nested, but strictly lower level -> acyclic)
    proto0 = _random_plain_tree(rng, 2, id_counter)
    proto1 = _random_plain_tree(rng, 2, id_counter)
    nested_parts = [
        Instance(proto0, _random_invertible_matrix(rng),
                 tuple(rng.randint(-2, 2) for _ in range(3)),
                 f"lib{next(inst_counter)}"),
        Instance(proto1, _random_invertible_matrix(rng),
                 tuple(rng.randint(-2, 2) for _ in range(3)),
                 f"lib{next(inst_counter)}"),
    ]
    proto2 = Union(nested_parts)
    library = [proto0, proto1, proto2]

    def gen(depth):
        if depth == 0 or rng.random() < 0.4:
            return _random_plain_tree(rng, 1, id_counter)
        if rng.random() < 0.45:
            proto = rng.choice(library)
            return Instance(
                proto,
                _random_invertible_matrix(rng),
                tuple(rng.randint(-3, 3) for _ in range(3)),
                f"i{next(inst_counter)}",
            )
        op = rng.choice("uid")
        count = rng.randint(2, 3)
        subs = [gen(depth - 1) for _ in range(count)]
        if op == "u":
            return Union(subs)
        if op == "i":
            return Intersection(subs)
        return Difference(subs[0], *subs[1:])

    return gen(3)


class TestRandomizedMembership(unittest.TestCase):
    def test_membership_against_independent_reference(self):
        rng = random.Random(20260922)
        cases = 0
        for _ in range(300):
            id_counter = iter(range(1000))
            tree = _random_tree(rng, depth=3, id_counter=id_counter)
            ray = _random_generic_ray(rng, tree)
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


class TestRandomizedInstancedMembership(unittest.TestCase):
    def test_instanced_scenes_against_independent_reference(self):
        rng = random.Random(20260925)
        cases = 0
        for _ in range(120):
            tree = _random_instanced_scene(rng)
            ray = _random_generic_ray(rng, tree)
            res = ray_intervals(tree, ray)
            cases += 1

            for iv in res.intervals:
                self.assertLess(iv.t0, iv.t1)
            for prev, nxt in zip(res.intervals, res.intervals[1:]):
                self.assertLess(prev.t1, nxt.t0)

            for iv in res.intervals:
                for frac in (0.19, 0.5, 0.83):
                    t = iv.t0 + (iv.t1 - iv.t0) * frac
                    p = _interior_point(ray.origin, ray.direction, t)
                    self.assertTrue(
                        ref_inside(tree, p),
                        f"interior sample t={t} not inside; interval={iv}",
                    )

            gap_ts = []
            if res.intervals:
                gap_ts.append(res.intervals[0].t0 - 1.31)
                gap_ts.append(res.intervals[-1].t1 + 1.77)
                for prev, nxt in zip(res.intervals, res.intervals[1:]):
                    gap_ts.append((prev.t1 + nxt.t0) / 2.0)
            else:
                gap_ts += [0.0, -2.5, 4.0]
            for t in gap_ts:
                p = _point_at(ray.origin, ray.direction, t)
                self.assertFalse(
                    ref_inside(tree, p), f"gap sample t={t} unexpectedly inside"
                )

            for touch in res.touches:
                for t in {touch.t0, touch.t1, (touch.t0 + touch.t1) / 2.0}:
                    p = _point_at(ray.origin, ray.direction, t)
                    self.assertFalse(
                        ref_inside(tree, p), f"touch t={t} unexpectedly interior"
                    )

            # source ordering and path qualification invariants
            frames = _collect_leaf_frames(tree)
            for boundary in res.boundaries:
                self.assertTrue(boundary.sources)
                keys = [(s.path, s.leaf_id, s.face) for s in boundary.sources]
                self.assertEqual(keys, sorted(keys))
                self.assertIs(boundary.source, boundary.sources[0])
                for s in boundary.sources:
                    # independently rebuild the world normal from the frame
                    leaf, A, b, polarity = frames[(s.path, s.leaf_id)]
                    Ainv = _inverse3(A)
                    p = [Fraction(ray.origin[i])
                         + Fraction(str(boundary.t)) * Fraction(ray.direction[i])
                         for i in range(3)]
                    q = [
                        sum(Ainv[3*r+c] * (p[c] - b[c]) for c in range(3))
                        for r in range(3)
                    ]
                    if isinstance(leaf, Sphere):
                        nl = [(q[i] - leaf.center[i]) / leaf.radius
                              for i in range(3)]
                    else:
                        axis, side = s.face[0], s.face[2:]
                        nl = [Fraction(0), Fraction(0), Fraction(0)]
                        nl["xyz".index(axis)] = Fraction(
                            1 if side == "hi" else -1
                        )
                    nw = [sum(Ainv[3*c+r] * nl[c] for c in range(3))
                          for r in range(3)]  # Ainv^T n_l
                    length = sum(c*c for c in nw) ** 0.5
                    expected = [(-c if polarity < 0 else c) / length
                                for c in map(float, nw)]
                    for got, want in zip(s.normal, expected):
                        self.assertAlmostEqual(got, want, places=9)
                    # orientation convention: enter faces the incoming ray
                    # (n.d < 0), exit faces away (n.d > 0)
                    dot = sum(s.normal[i] * float(ray.direction[i])
                              for i in range(3))
                    if boundary.kind == "enter":
                        self.assertLess(dot, 0.0)
                    else:
                        self.assertGreater(dot, 0.0)

            forward = [b for b in res.boundaries if b.t > 0.0]
            if forward:
                self.assertIsNotNone(res.nearest_boundary)
                self.assertEqual(
                    res.nearest_boundary.t, min(b.t for b in forward)
                )

        self.assertEqual(cases, 120)


if __name__ == "__main__":
    unittest.main()
