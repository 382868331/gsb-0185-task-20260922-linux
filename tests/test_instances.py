"""Tests for Instance leaves: a shared CSG subtree under an invertible affine.

Every expected value here is derived independently of the implementation
(closed-form quadrics / slab inequalities / the inverse-transpose normal
rule).  Hit points are substituted back into the analytic implicit
equations; expected normals are computed from the analytic gradients.
"""

import math
import unittest

from csg_rays import (
    MAX_TREE_NODES,
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

_ID3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_ZERO3 = (0, 0, 0)


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _hit_point(ray, t):
    return tuple(ray.origin[i] + t * ray.direction[i] for i in range(3))


class TestScaledSphereEllipsoid(unittest.TestCase):
    """Unit sphere under diag(2, 3, 1) -> ellipsoid x^2/4 + y^2/9 + z^2 = 1."""

    def setUp(self):
        self.ball = Sphere((0, 0, 0), 1, "ball")
        self.ellipsoid = Instance(
            self.ball, ((2, 0, 0), (0, 3, 0), (0, 0, 1)), _ZERO3, "ell"
        )

    def test_axis_ray_exact_endpoints_and_normals(self):
        ray = Ray((-6, 0, 0), (1, 0, 0))
        res = ray_intervals(self.ellipsoid, ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 8.0)])
        enter, exit_ = res.boundaries
        # path identifies the reference; a directly placed leaf has path ()
        self.assertEqual(enter.source.path, ("ell",))
        self.assertEqual(enter.source.leaf_id, "ball")
        self.assertFalse(enter.source.flipped)
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))

    def test_offaxis_ray_hits_satisfy_implicit_equation(self):
        ray = Ray((-6, 1, 0), (1, 0, 0))  # y=1: (t-6)^2/4 = 8/9
        res = ray_intervals(self.ellipsoid, ray)
        self.assertEqual(len(res.intervals), 1)
        t0, t1 = res.intervals[0].t0, res.intervals[0].t1
        self.assertAlmostEqual(t0, 6 - 4 * math.sqrt(2.0) / 3, places=12)
        self.assertAlmostEqual(t1, 6 + 4 * math.sqrt(2.0) / 3, places=12)
        for b in res.boundaries:
            p = _hit_point(ray, b.t)
            # independent implicit equation F = x^2/4 + y^2/9 + z^2 - 1
            f = p[0] ** 2 / 4 + p[1] ** 2 / 9 + p[2] ** 2 - 1
            self.assertAlmostEqual(f, 0.0, delta=1e-11)
            # independent normal: grad F = (x/2, 2y/9, 2z), normalized
            expected = _unit((p[0] / 2, 2 * p[1] / 9, 2 * p[2]))
            for got, want in zip(b.source.normal, expected):
                self.assertAlmostEqual(got, want, delta=1e-11)

    def test_interior_start_reports_both_line_halves(self):
        res = ray_intervals(self.ellipsoid, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-2.0, 2.0)])
        self.assertEqual(res.nearest_boundary.source.normal, (1.0, 0.0, 0.0))

    def test_tangent_ray_stays_zero_measure_touch(self):
        # y=3 grazes the ellipsoid at (0, 3): exactly one tangent point t=6.
        res = ray_intervals(self.ellipsoid, Ray((-6, 3, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (6.0, 6.0))
        self.assertEqual(touch.source.kind, "touch")
        p = _hit_point(Ray((-6, 3, 0), (1, 0, 0)), 6.0)
        self.assertAlmostEqual(p[1] ** 2 / 9 - 1, 0.0, delta=1e-12)
        self.assertIsNone(res.nearest_boundary)

    def test_shared_node_is_not_modified(self):
        # The same ball is referenced twice at two places; the node itself
        # must keep its original frame when intersected on its own.
        moved_a = Instance(self.ball, _ID3, (10, 0, 0), "a")
        moved_b = Instance(self.ball, _ID3, (-10, 0, 0), "b")
        res = ray_intervals(Union([moved_a, moved_b]), Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals], [(-11.0, -9.0), (9.0, 11.0)]
        )
        plain = ray_intervals(self.ball, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in plain.intervals], [(-1.0, 1.0)])
        self.assertEqual(self.ball.center, (0, 0, 0))


class TestShearedBox(unittest.TestCase):
    """Unit cube under A=[[1,1,0],[0,1,0],[0,0,1]]: slab -1 <= x-y <= 1."""

    def setUp(self):
        self.cube = Box((-1, -1, -1), (1, 1, 1), "cube")
        self.shear = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
        self.sheared = Instance(self.cube, self.shear, _ZERO3, "sh")

    def test_axis_ray_endpoints_and_inverse_transpose_normals(self):
        ray = Ray((-4, 0, 0), (1, 0, 0))  # -1 <= t-4 <= 1 -> t in (3, 5)
        res = ray_intervals(self.sheared, ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(3.0, 5.0)])
        enter, exit_ = res.boundaries
        # x_lo face x-y=-1 outward normal (-1,+1,0)/sqrt(2);
        # x_hi face x-y=+1 outward normal (+1,-1,0)/sqrt(2).
        for got, want in zip(enter.source.normal, _unit((-1, 1, 0))):
            self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertEqual(enter.source.face, "x_lo")
        for got, want in zip(exit_.source.normal, _unit((1, -1, 0))):
            self.assertAlmostEqual(got, want, delta=1e-12)
        for b in res.boundaries:
            p = _hit_point(ray, b.t)
            self.assertTrue(-1.0 - 1e-12 <= p[0] - p[1] <= 1.0 + 1e-12)
            self.assertAlmostEqual(abs(p[0] - p[1]), 1.0, delta=1e-12)

    def test_face_riding_contact_segment(self):
        # World ray p(t)=(-2,-1,0)+t(1,1,0) rides the transformed face
        # q_x = x-y = -1 for q_y in [-1,1], i.e. t in [0, 2]: zero measure.
        ray = Ray((-2, -1, 0), (1, 1, 0))
        res = ray_intervals(self.sheared, ray)
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (0.0, 2.0))
        faces = {s.face for s in touch.sources if s.path == ("sh",)}
        self.assertEqual(faces, {"x_lo", "y_lo", "y_hi"})
        self.assertIsNone(res.nearest_boundary)


class TestRotatedBox(unittest.TestCase):
    """A 90-degree rotation (orthogonal): inverse transpose equals R."""

    def test_rotated_cube_endpoints_normals(self):
        rot = ((0, -1, 0), (1, 0, 0), (0, 0, 1))  # q -> (-qy, qx, 0)
        cube = Box((-1, -1, -1), (1, 1, 1), "cube")
        tree = Instance(cube, rot, _ZERO3, "rot")
        # p=(t-3, 1/2); q=(y,-x)=(1/2, 3-t) inside [-1,1]^3 -> t in (2,4)
        ray = Ray((-3, 0.5, 0), (1, 0, 0))
        res = ray_intervals(tree, ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(2.0, 4.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.face, "y_hi")  # q_y=1 entered first
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.face, "y_lo")
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))


class TestMirroredDifference(unittest.TestCase):
    """Negative determinant must not swap enter/exit; difference flip stacks."""

    def setUp(self):
        mirror = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.hole = Sphere((3, 0, 0), 1, "hole")
        mirrored_hole = Instance(self.hole, mirror, _ZERO3, "mirror")
        self.tree = Difference(Sphere((0, 0, 0), 5, "shell"), mirrored_hole)

    def test_mirrored_subtractor_intervals(self):
        # cavity center moves from +3 to -3; ray along x: solid (-5,5)
        # minus (-4,-2) -> (-5,-4) and (-2,5).
        res = ray_intervals(self.tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals], [(-5.0, -4.0), (-2.0, 5.0)]
        )
        kinds = [b.kind for b in res.boundaries]
        self.assertEqual(kinds, ["enter", "exit", "enter", "exit"])
        b_minus5, b_minus4, b_minus2, b5 = res.boundaries
        # shell boundaries: unflipped
        self.assertEqual(b_minus5.source.leaf_id, "shell")
        self.assertFalse(b_minus5.source.flipped)
        # cavity walls: subtractor face, world normal flipped to point into
        # the cavity; mirroring is already in the geometric normal.
        self.assertEqual(b_minus4.source.leaf_id, "hole")
        self.assertEqual(b_minus4.source.path, ("mirror",))
        self.assertTrue(b_minus4.source.flipped)
        self.assertEqual(b_minus4.source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(b_minus2.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(b5.source.normal, (1.0, 0.0, 0.0))

    def test_plain_mirror_does_not_swap_enter_exit(self):
        # The mirrored subtractor alone is a ball centered at -3.
        mirror = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
        tree = Instance(Sphere((3, 0, 0), 1, "s"), mirror, _ZERO3, "m")
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-4.0, -2.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.kind, "enter")  # not swapped under det < 0
        self.assertEqual(exit_.kind, "exit")
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))
        for b in res.boundaries:  # both hit points lie on |x + 3| = 1
            p = _hit_point(Ray((0, 0, 0), (1, 0, 0)), b.t)
            self.assertAlmostEqual(abs(p[0] + 3.0), 1.0, delta=1e-12)

    def test_mirrored_difference_sampled_membership(self):
        # Independent check along a generic ray against closed-form
        # membership: |p| < 5 and not |p - (-3,0,0)| <= 1.
        ray = Ray((-7, 2, 1), (2, -1, 1))
        res = ray_intervals(self.tree, ray)
        for iv in res.intervals:
            for f in (0.2, 0.5, 0.81):
                t = iv.t0 + (iv.t1 - iv.t0) * f + 1e-7
                p = _hit_point(ray, t)
                inside_shell = sum(c * c for c in p) < 25.0
                inside_cavity = (p[0] + 3.0) ** 2 + p[1] ** 2 + p[2] ** 2 <= 1.0
                self.assertTrue(inside_shell and not inside_cavity)


class TestNestedInstances(unittest.TestCase):
    def test_nested_composition_and_path(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        big = Instance(ball, ((2, 0, 0), (0, 2, 0), (0, 0, 2)), _ZERO3, "big")
        moved = Instance(big, _ID3, (10, 0, 0), "move")  # ball r=2 at x=10
        res = ray_intervals(moved, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(8.0, 12.0)])
        for b in res.boundaries:
            self.assertEqual(b.source.path, ("move", "big"))
            self.assertEqual(b.source.leaf_id, "ball")

    def test_two_references_coincident_batch_sources(self):
        ball = Sphere((0, 0, 0), 1, "ball")
        i1 = Instance(ball, _ID3, (5, 0, 0), "i1")
        i2 = Instance(ball, _ID3, (5, 0, 0), "i2")  # identical placement
        res = ray_intervals(Union([i1, i2]), Ray((0, 0, 0), (1, 0, 0)))
        # same-t batch rule: one interval, no phantom boundary at t=4/6
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])
        for b in res.boundaries:
            paths = [s.path for s in b.sources]
            self.assertEqual(paths, [("i1",), ("i2",)])  # path-sorted
            self.assertEqual(b.source.path, ("i1",))      # representative

    def test_two_references_of_difference_subtree_independent(self):
        subtree = Difference(
            Box((-2, -2, -2), (2, 2, 2), "block"),
            Sphere((0, 0, 0), 1, "core"),
        )
        tree = Union([
            Instance(subtree, _ID3, (10, 0, 0), "left"),
            Instance(subtree, _ID3, (-10, 0, 0), "right"),
        ])
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        # right block at -10: (-12,-8) minus core (-11,-9);
        # left block at +10: (8,12) minus core (9,11).
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(-12.0, -11.0), (-9.0, -8.0), (8.0, 9.0), (11.0, 12.0)],
        )
        # flipped core normals carry the instance path of their reference
        core_sources = [
            s for b in res.boundaries for s in b.sources if s.leaf_id == "core"
        ]
        self.assertTrue(all(s.flipped for s in core_sources))
        self.assertTrue(any(s.path == ("left",) for s in core_sources))
        self.assertTrue(any(s.path == ("right",) for s in core_sources))


class TestSingularAndCycles(unittest.TestCase):
    def test_singular_matrix_rejected_at_construction(self):
        with self.assertRaises(GeometryError) as ctx:
            Instance(
                Sphere((0, 0, 0), 1, "s"),
                ((1, 2, 3), (4, 5, 6), (7, 8, 9)),  # determinant 0
                _ZERO3,
                "bad",
            )
        self.assertIn("singular", str(ctx.exception))

    def test_bad_transform_inputs(self):
        good = Sphere((0, 0, 0), 1, "s")
        with self.assertRaises(GeometryError):
            Instance(good, ((1, 0, 0), (0, 1, 0)), _ZERO3, "x")  # 2 rows
        with self.assertRaises(GeometryError):
            Instance(good, _ID3, (0, 0), "x")                     # 2-vector
        with self.assertRaises(GeometryError):
            Instance(42, _ID3, _ZERO3, "x")                       # not a node
        with self.assertRaises(GeometryError):
            Instance(good, _ID3, _ZERO3, "")                      # empty id
        with self.assertRaises(GeometryError) as ctx:
            Instance(good, ((1, True, 0), (0, 1, 0), (0, 0, 1)),
                     _ZERO3, "x")                                 # bool rejected
        self.assertIn("bool", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Instance(good, _ID3, (float("inf"), 0, 0), "x")

    def test_self_reference_cycle_rejected(self):
        inst = Instance(Sphere((0, 0, 0), 1, "s"), _ID3, _ZERO3, "cyc")
        wrapper = Union([inst, Sphere((9, 0, 0), 1, "other")])
        object.__setattr__(inst, "child", wrapper)  # cycles cannot be built
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(inst, Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn("cycle", str(ctx.exception))

    def test_mutual_reference_cycle_rejected(self):
        a = Instance(Sphere((0, 0, 0), 1, "sa"), _ID3, _ZERO3, "a")
        b = Instance(Sphere((0, 0, 0), 1, "sb"), _ID3, _ZERO3, "b")
        object.__setattr__(a, "child", Union([Sphere((1, 0, 0), 1, "x"), b]))
        object.__setattr__(b, "child", Union([Sphere((2, 0, 0), 1, "y"), a]))
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(a, Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn("cycle", str(ctx.exception))

    def test_structural_sharing_outside_references_still_rejected(self):
        leaf = Sphere((0, 0, 0), 1, "s")
        with self.assertRaises(GeometryError):
            ray_intervals(Union([leaf, leaf]), Ray((0, 0, 0), (1, 0, 0)))

    def test_node_budget_counts_reference_occurrences(self):
        ball = Sphere((0, 0, 0), 1, "s")
        ok = Union([Instance(ball, _ID3, (i, 0, 0), f"r{i}") for i in range(49)])
        res = ray_intervals(ok, Ray((0, 0, 0), (1, 0, 0)))
        self.assertTrue(res.intervals)
        too_many = Union(
            [Instance(ball, _ID3, (i, 0, 0), f"q{i}") for i in range(50)]
        )  # 1 union + 50 instances + 50 leaf occurrences = 101 > 100
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(too_many, Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn(str(MAX_TREE_NODES), str(ctx.exception))

    def test_duplicate_instance_id_on_same_path_rejected(self):
        s1 = Sphere((0, 0, 0), 1, "a")
        s2 = Sphere((5, 0, 0), 1, "b")
        tree = Union([
            Instance(s1, _ID3, (0, 0, 0), "k"),
            Instance(s2, _ID3, (0, 0, 0), "k"),  # same path ("k",)
        ])
        with self.assertRaises(GeometryError):
            ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))

    def test_subtree_placed_directly_and_referenced(self):
        # The same subtree may be used directly and behind a reference; the
        # occurrences are independent and distinguished by path.
        sub = Difference(
            Box((-1, -1, -1), (1, 1, 1), "blk"),
            Sphere((0, 0, 0), 0.5, "core"),
        )
        tree = Union([sub, Instance(sub, _ID3, (5, 0, 0), "copy")])
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        # direct occurrence: (-1,-0.5) and (0.5,1); copy: (4,4.5),(5.5,6)
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(-1.0, -0.5), (0.5, 1.0), (4.0, 4.5), (5.5, 6.0)],
        )
        paths = {
            s.path for b in res.boundaries for s in b.sources
        } | {s.path for tch in res.touches for s in tch.sources}
        self.assertIn((), paths)
        self.assertIn(("copy",), paths)


if __name__ == "__main__":
    unittest.main()
