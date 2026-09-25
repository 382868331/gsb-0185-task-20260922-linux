"""Tests for instanced subtrees under invertible affine transforms.

Covers the task's required cases:

* scaled sphere vs an independently derived analytic ellipsoid (exact
  rational endpoints, tangent touch, implicit-equation back-substitution and
  gradient-based normal checks);
* sheared box (inverse-transpose normals, face-riding contact);
* reflection (negative determinant): enter/exit must never be swapped, and a
  mirrored difference with correctly composed flip levels;
* nested instances (frame composition) and the same shared subtree
  referenced twice (sources distinguished by instance path; coincident
  events still follow the same-t batch rule);
* singular matrices and reference cycles are rejected.

Expected values are derived analytically in the tests; the implementation
under test is never called to produce them.
"""

import math
import unittest
from fractions import Fraction

from csg_rays import (
    MAX_LEAF_OCCURRENCES,
    Box,
    Difference,
    GeometryError,
    Instance,
    Intersection,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)

IDENT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _fr_mat(m):
    return [[Fraction(v) for v in row] for row in m]


def _mv(A, v):
    return [sum(A[r][c] * v[c] for c in range(3)) for r in range(3)]


def _inv3(A):
    """Independent Fraction 3x3 inverse (used only by the test oracle)."""
    a = [A[r // 3][r % 3] for r in range(9)]
    c00, c01, c02 = a[4]*a[8]-a[5]*a[7], a[5]*a[6]-a[3]*a[8], a[3]*a[7]-a[4]*a[6]
    det = a[0]*c00 + a[1]*c01 + a[2]*c02
    cof = [c00, c01, c02,
           a[2]*a[7]-a[1]*a[8], a[0]*a[8]-a[2]*a[6], a[1]*a[6]-a[0]*a[7],
           a[1]*a[5]-a[2]*a[4], a[2]*a[3]-a[0]*a[5], a[0]*a[4]-a[1]*a[3]]
    adj = [cof[0], cof[3], cof[6], cof[1], cof[4], cof[7], cof[2], cof[5], cof[8]]
    return [[adj[3*r+c] / det for c in range(3)] for r in range(3)]


def _unit(v):
    n = math.sqrt(sum(c*c for c in v))
    return [c / n for c in v]


class TestScaledSphereEllipsoid(unittest.TestCase):
    """Unit sphere scaled by A=diag(2,1,1): ellipsoid x^2/4+y^2+z^2=1."""

    def setUp(self):
        self.A = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.tree = Instance(
            Sphere((0, 0, 0), 1, "ball"), self.A, (0, 0, 0), "ell"
        )

    def test_axis_endpoints_exact(self):
        res = ray_intervals(self.tree, Ray((-10, 0, 0), (1, 0, 0)))
        # (t-10)^2/4 = 1 -> t = 8, 12 exactly
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(8.0, 12.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.path, ("ell",))
        self.assertEqual(enter.source.leaf_id, "ball")
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))

    def test_offset_ray_matches_analytic_roots(self):
        # independent analytic: (t-5)^2/4 + 0.09 = 1
        res = ray_intervals(self.tree, Ray((-5, 0.3, 0), (1, 0, 0)))
        t0, t1 = res.intervals[0].t0, res.intervals[0].t1
        self.assertAlmostEqual(t0, 5 - math.sqrt(3.64), places=12)
        self.assertAlmostEqual(t1, 5 + math.sqrt(3.64), places=12)

    def test_hit_points_satisfy_implicit_equation(self):
        # F(p) = p^T A^-T A^-1 p - 1; checked at every reported boundary.
        Ainv = _inv3(_fr_mat(self.A))
        G = [[sum(Ainv[k][r] * Ainv[k][c] for k in range(3)) for c in range(3)]
             for r in range(3)]
        for origin, direction in [
            ((-7, 0.2, 0.1), (1, 0, 0)),
            ((2, -6, 1), (0.3, 1, 0.0)),
        ]:
            res = ray_intervals(self.tree, Ray(origin, direction))
            for b in res.boundaries:
                p = [origin[i] + b.t * direction[i] for i in range(3)]
                f = sum(G[r][c] * p[r] * p[c] for r in range(3) for c in range(3)) - 1
                self.assertAlmostEqual(f, 0.0, places=10, msg=f"{b} {p}")

    def test_normals_match_analytic_gradient(self):
        # outward normal of F is grad F / |grad F| = G p / |G p|
        Ainv = _inv3(_fr_mat(self.A))
        G = [[sum(Ainv[k][r] * Ainv[k][c] for k in range(3)) for c in range(3)]
             for r in range(3)]
        res = ray_intervals(self.tree, Ray((-7, 0.2, 0.1), (1, 0, 0)))
        for b in res.boundaries:
            p = [(-7) + b.t * 1.0, 0.2, 0.1]
            expected = _unit([sum(G[r][c] * p[c] for c in range(3)) for r in range(3)])
            for got, want in zip(b.source.normal, expected):
                self.assertAlmostEqual(got, want, places=10)
            # normal must be perpendicular to the ray-plane tangent error
            self.assertAlmostEqual(
                sum(c * c for c in b.source.normal), 1.0, places=12
            )

    def test_tangent_ray_is_touch_with_correct_normal(self):
        # ray y=1 grazes the ellipsoid tangentially at p=(0,1,0), t=10
        res = ray_intervals(self.tree, Ray((-10, 1, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (10.0, 10.0))
        self.assertEqual(touch.source.normal, (0.0, 1.0, 0.0))
        self.assertEqual(touch.source.kind, "touch")
        self.assertIsNone(res.nearest_boundary)


class TestShearedBox(unittest.TestCase):
    """Unit cube [0,1]^3 under shear p_w = (x+y, y, z)."""

    def setUp(self):
        self.A = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        self.tree = Instance(
            Box((0, 0, 0), (1, 1, 1), "cube"), self.A, (0, 0, 0), "sh"
        )

    def test_axis_ray_endpoints_and_normals(self):
        # world ray (-2,.5,.5)+t(1,0,0): local ray (-2.5,.5,.5)+t(1,0,0),
        # enters x_l=0 at t=2.5 (world (.5,.5,.5)), exits x_l=1 at t=3.5.
        res = ray_intervals(self.tree, Ray((-2, 0.5, 0.5), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(2.5, 3.5)])
        enter, exit_ = res.boundaries
        # face plane x_w - y_w = 0; x_lo outward normal (-1,1,0)/sqrt2
        for got, want in zip(enter.source.normal,
                             _unit([-1.0, 1.0, 0.0])):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(exit_.source.normal,
                             _unit([1.0, -1.0, 0.0])):
            self.assertAlmostEqual(got, want, places=12)

    def test_face_riding_contact_normal(self):
        # world ray (-2,0,-2)+t(1,0,1): rides the local y_lo face, which the
        # shear maps to the plane y_w=0; zero-measure touch run t in [2,3].
        res = ray_intervals(self.tree, Ray((-2, 0, -2), (1, 0, 1)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (2.0, 3.0))
        normals = {s.face: s.normal for s in touch.sources}
        self.assertEqual(normals["y_lo"], (0.0, -1.0, 0.0))

    def test_hit_points_inside_transformed_slab_oracle(self):
        # independent membership: p inside iff A^-1 p in [0,1]^3
        Ainv = _inv3(_fr_mat(self.A))
        res = ray_intervals(self.tree, Ray((-4, 0.3, 0.7), (1, 0, 0)))
        iv = res.intervals[0]
        for f in (0.2, 0.5, 0.8):
            t = iv.t0 + f * (iv.t1 - iv.t0)
            p = [-4 + t, 0.3, 0.7]
            q = _mv(Ainv, [Fraction(c) for c in p])
            for c in q:
                self.assertTrue(0 < c < 1)
        for t in (iv.t0 - 0.5, iv.t1 + 0.5):
            p = [-4 + t, 0.3, 0.7]
            q = _mv(Ainv, [Fraction(c) for c in p])
            self.assertFalse(all(0 < c < 1 for c in q))


class TestReflection(unittest.TestCase):
    """Negative determinant: mirror must not swap enter/exit intervals."""

    def test_mirrored_sphere_intervals_and_normals(self):
        # local unit sphere mirrored to center x=10: x_w = 10 - x_l.
        mir = Instance(
            Sphere((0, 0, 0), 1, "ball"),
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], (10, 0, 0), "mir",
        )
        res = ray_intervals(mir, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(9.0, 11.0)])
        self.assertEqual([b.kind for b in res.boundaries], ["enter", "exit"])
        self.assertEqual(res.boundaries[0].source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(res.boundaries[1].source.normal, (1.0, 0.0, 0.0))

    def test_mirrored_sphere_ray_travelling_backwards(self):
        # ray travelling in -x: local direction becomes +x; still enter first.
        mir = Instance(
            Sphere((0, 0, 0), 1, "ball"),
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], (10, 0, 0), "mir",
        )
        res = ray_intervals(mir, Ray((20, 0, 0), (-1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(9.0, 11.0)])
        self.assertEqual([b.kind for b in res.boundaries], ["enter", "exit"])
        # enter at world x=11: local x=-1, -e_x pushed through diag(-1,1,1)
        self.assertEqual(res.boundaries[0].source.normal, (1.0, 0.0, 0.0))

    def test_mirrored_difference_flip_levels(self):
        # big box [0,4]^3 minus the unit cube mirrored in z (translation 4,
        # A z-flip): cavity occupies world z in [3,4] at x,y in [0,1].
        sub = Instance(
            Box((0, 0, 0), (1, 1, 1), "b"),
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]], (0, 0, 4), "m",
        )
        tree = Difference(Box((0, 0, 0), (4, 4, 4), "big"), sub)
        res = ray_intervals(tree, Ray((0.5, 0.5, -5), (0, 0, 1)))
        # solid z in (0,3) (boundary at z=4 collapses: outer face coincides
        # with the cavity back wall; forcing either side leaves the point OUT)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(5.0, 8.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.leaf_id, "big")
        self.assertFalse(enter.source.flipped)
        self.assertEqual(exit_.source.leaf_id, "b")
        self.assertEqual(exit_.source.face, "z_hi")
        self.assertTrue(exit_.source.flipped)
        # local z_hi normal +e_z -> A^-T gives -e_z -> Difference flip: +e_z
        self.assertEqual(exit_.source.normal, (0.0, 0.0, 1.0))


class TestSharingAndNesting(unittest.TestCase):
    def test_same_subtree_two_references_paths_distinguish_sources(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        tree = Union([
            Instance(ball, IDENT, (0, 0, 0), "a"),
            Instance(ball, IDENT, (5, 0, 0), "b"),
        ])
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(-1.0, 1.0), (4.0, 6.0)],
        )
        for b in res.boundaries:
            self.assertEqual(len(b.sources), 1)
        self.assertEqual(res.boundaries[0].source.path, ("a",))
        self.assertEqual(res.boundaries[2].source.path, ("b",))

    def test_coincident_instanced_faces_share_one_boundary(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        tree = Union([
            Instance(ball, IDENT, (0, 0, 0), "a"),
            Instance(ball, IDENT, (0, 0, 0), "b"),  # geometrically identical
        ])
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])
        for b in res.boundaries:
            self.assertEqual(len(b.sources), 2)  # same-t batch, both paths
            self.assertEqual(b.source.path, ("a",))  # path sorts before b

    def test_identical_instanced_difference_is_empty(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        tree = Difference(
            Instance(ball, IDENT, (0, 0, 0), "a"),
            Instance(ball, IDENT, (0, 0, 0), "b"),
        )
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.boundaries, ())
        self.assertEqual(res.touches, ())

    def test_shared_subtree_used_statically_and_as_instance(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        tree = Union([ball, Instance(ball, IDENT, (5, 0, 0), "copy")])
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(-1.0, 1.0), (4.0, 6.0)],
        )
        paths = {b.source.path for b in res.boundaries}
        self.assertEqual(paths, {(), ("copy",)})

    def test_nested_instances_compose_frames(self):
        # inner: p -> p + (10,0,0); outer scales by 2: world center 20, r 2
        leaf = Sphere((0, 0, 0), 1, "s")
        inner = Instance(leaf, IDENT, (10, 0, 0), "in")
        outer = Instance(inner, [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                         (0, 0, 0), "out")
        res = ray_intervals(outer, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals],
                         [(18.0, 22.0)])
        self.assertEqual(res.boundaries[0].source.path, ("out", "in"))

    def test_shared_child_is_not_mutated_by_reference(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        moved = Instance(ball, IDENT, (100, 0, 0), "moved")
        # referencing the same subtree under two different transforms leaves
        # both references intact (and the leaf object itself unchanged)
        tree = Union([
            Instance(ball, IDENT, (0, 0, 0), "a"),
            moved,
        ])
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(4.0, 6.0), (104.0, 106.0)],
        )
        self.assertEqual(ball.center, (Fraction(0), Fraction(0), Fraction(0)))

    def test_polarity_through_nested_instance_in_subtractor(self):
        # subtractor subtree contains an instance; its sources must be flipped
        ball = Sphere((0, 0, 0), 1, "ball")
        tree = Difference(
            Box((-5, -5, -5), (5, 5, 5), "big"),
            Instance(ball, IDENT, (0, 0, 0), "ref"),
        )
        res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
        cavity = [b for b in res.boundaries if b.source.leaf_id == "ball"]
        self.assertEqual(len(cavity), 2)
        for b in cavity:
            self.assertTrue(b.source.flipped)
            self.assertEqual(b.source.path, ("ref",))


class TestInstanceValidation(unittest.TestCase):
    def test_singular_matrix_rejected(self):
        with self.assertRaises(GeometryError) as ctx:
            Instance(
                Sphere((0, 0, 0), 1, "s"),
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], (0, 0, 0), "sing",
            )
        self.assertIn("linear", str(ctx.exception))
        self.assertIn("singular", str(ctx.exception))

    def test_bad_matrix_shape_rejected(self):
        with self.assertRaises(GeometryError):
            Instance(Sphere((0, 0, 0), 1, "s"),
                     [[1, 0, 0], [0, 1, 0]], (0, 0, 0), "bad")
        with self.assertRaises(GeometryError):
            Instance(Sphere((0, 0, 0), 1, "s"),
                     ((1, 0, 0), (0, 1, "x"), (0, 0, 1)), (0, 0, 0), "bad")
        with self.assertRaises(GeometryError):
            Instance(Sphere((0, 0, 0), 1, "s"),
                     IDENT, (0, float("inf"), 0), "bad")

    def test_instance_id_rules(self):
        with self.assertRaises(GeometryError):
            Instance(Sphere((0, 0, 0), 1, "s"), IDENT, (0, 0, 0), "")
        with self.assertRaises(GeometryError):
            Instance("not a node", IDENT, (0, 0, 0), "x")

    def test_duplicate_instance_id_rejected(self):
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(
                Union([
                    Instance(Sphere((0, 0, 0), 1, "a"), IDENT, (0, 0, 0), "i"),
                    Instance(Sphere((5, 0, 0), 1, "b"), IDENT, (0, 0, 0), "i"),
                ]),
                Ray((0, 0, 0), (1, 0, 0)),
            )
        self.assertIn("duplicate instance_id", str(ctx.exception))

    def test_reference_cycle_rejected(self):
        # Instance uses slots (no frozen dataclass), so a back-edge can be
        # constructed by mutating .child after creation.
        top = Instance(Sphere((0, 0, 0), 1, "z"), IDENT, (0, 0, 0), "top")
        wrap = Union([top])
        top.child = wrap
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(wrap, Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn("cycle", str(ctx.exception))

    def test_plain_sharing_without_instance_rejected(self):
        shared = Box((0, 0, 0), (1, 1, 1), "b")
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(Union([shared, shared]), Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn("Instance", str(ctx.exception))

    def test_leaf_occurrence_budget(self):
        # Each level doubles occurrences for 3 new declared nodes; 2048
        # occurrences need far fewer than MAX_TREE_NODES declared nodes.
        node = Sphere((0, 0, 0), 1, "seed")
        for level in range(11):
            node = Union([
                Instance(node, IDENT, (0, 0, 10 * level), f"l{level}a"),
                Instance(node, IDENT, (0, 0, -10 * level), f"l{level}b"),
            ])
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(node, Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn(str(MAX_LEAF_OCCURRENCES), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
