"""Analytic tests: exact endpoints, coincident faces, tangents, interior starts.

Expected values here are derived by hand from closed-form geometry, never by
calling the implementation under test.
"""

import math
import unittest

from csg_rays import (
    MAX_TREE_NODES,
    Box,
    CertificationBudgetExhausted,
    CSGError,
    Difference,
    GeometryError,
    Intersection,
    Ray,
    Rotate,
    Scale,
    Sphere,
    Transform,
    Translate,
    Union,
    ray_intervals,
)


class TestHollowSphereExact(unittest.TestCase):
    """Concentric hollow sphere: Difference(outer r=5, inner r=2)."""

    def setUp(self):
        self.hollow = Difference(
            Sphere((0, 0, 0), 5, "outer"), Sphere((0, 0, 0), 2, "inner")
        )

    def test_exact_endpoints_axis_ray(self):
        # Ray (-10,0,0) + t*(1,0,0): crosses |x|=5 at t=5,15 and |x|=2 at t=8,12.
        res = ray_intervals(self.hollow, Ray((-10, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(5.0, 8.0), (12.0, 15.0)])
        self.assertEqual(res.touches, ())
        kinds = [b.kind for b in res.boundaries]
        self.assertEqual(kinds, ["enter", "exit", "enter", "exit"])
        # cavity walls come from the subtractor with flipped normals
        b5, b8, b12, b15 = res.boundaries
        self.assertEqual((b5.source.leaf_id, b5.source.flipped), ("outer", False))
        self.assertEqual(b5.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual((b8.source.leaf_id, b8.source.flipped), ("inner", True))
        self.assertEqual(b8.source.normal, (1.0, 0.0, 0.0))   # into the cavity
        self.assertEqual((b12.source.leaf_id, b12.source.flipped), ("inner", True))
        self.assertEqual(b12.source.normal, (-1.0, 0.0, 0.0))  # into the cavity
        self.assertEqual((b15.source.leaf_id, b15.source.flipped), ("outer", False))
        self.assertEqual(b15.source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(res.nearest_boundary.t, 5.0)
        self.assertEqual(res.nearest_boundary.kind, "enter")

    def test_interior_start_inside_cavity(self):
        # Origin (0,0,0) lies in the cavity: solid is (-5,-2) and (2,5).
        res = ray_intervals(self.hollow, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-5.0, -2.0), (2.0, 5.0)])
        # nearest forward boundary is the cavity wall at t=2 (entering solid)
        self.assertEqual(res.nearest_boundary.t, 2.0)
        self.assertEqual(res.nearest_boundary.kind, "enter")
        self.assertEqual(res.nearest_boundary.source.leaf_id, "inner")
        self.assertEqual(res.nearest_boundary.source.normal, (-1.0, 0.0, 0.0))

    def test_tangent_to_cavity_is_swallowed(self):
        # Ray (-10,2,0) + t*(1,0,0) grazes the inner sphere (tangent at t=10)
        # while crossing the outer one at t = 10 - sqrt(21), 10 + sqrt(21).
        res = ray_intervals(self.hollow, Ray((-10, 2, 0), (1, 0, 0)))
        self.assertEqual(len(res.intervals), 1)
        self.assertEqual(res.touches, ())  # tangent inside solid: no puncture
        iv = res.intervals[0]
        root = math.sqrt(21.0)
        self.assertAlmostEqual(iv.t0, 10.0 - root, places=12)
        self.assertAlmostEqual(iv.t1, 10.0 + root, places=12)
        # exact algebraic identities of the quadratic roots
        self.assertAlmostEqual(iv.t0 + iv.t1, 20.0, places=12)
        self.assertAlmostEqual(iv.t0 * iv.t1, 79.0, places=9)


class TestSphereBasics(unittest.TestCase):
    def test_miss_is_empty_not_error(self):
        res = ray_intervals(Sphere((0, 0, 0), 2, "s"), Ray((0, 5, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.touches, ())
        self.assertEqual(res.boundaries, ())
        self.assertIsNone(res.nearest_boundary)

    def test_tangent_ray_is_zero_measure_touch(self):
        # Ray at distance exactly r from the center: tangent at t=5.
        res = ray_intervals(Sphere((0, 0, 0), 2, "s"), Ray((-5, 2, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (5.0, 5.0))
        self.assertEqual(touch.source.leaf_id, "s")
        self.assertEqual(touch.source.kind, "touch")
        self.assertEqual(touch.source.normal, (0.0, 1.0, 0.0))
        self.assertIsNone(res.nearest_boundary)  # a touch is not a solid boundary

    def test_ray_pointing_away_still_reports_line_interval(self):
        # The ray is a full line; nearest_boundary filters to t > 0.
        res = ray_intervals(Sphere((0, 0, 0), 2, "s"), Ray((5, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-7.0, -3.0)])
        self.assertIsNone(res.nearest_boundary)

    def test_interior_origin(self):
        res = ray_intervals(Sphere((0, 0, 0), 2, "s"), Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-2.0, 2.0)])
        self.assertEqual(res.nearest_boundary.t, 2.0)
        self.assertEqual(res.nearest_boundary.kind, "exit")


class TestBoxFaces(unittest.TestCase):
    def test_generic_crossing_faces(self):
        res = ray_intervals(Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, 1, 1), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(1.0, 3.0)])
        enter, exit_ = res.boundaries
        self.assertEqual([s.face for s in enter.sources], ["x_lo"])
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual([s.face for s in exit_.sources], ["x_hi"])

    def test_corner_hit_lists_all_coincident_faces(self):
        res = ray_intervals(Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, -1, -1), (1, 1, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(1.0, 3.0)])
        enter, exit_ = res.boundaries
        self.assertEqual([s.face for s in enter.sources], ["x_lo", "y_lo", "z_lo"])
        self.assertEqual([s.face for s in exit_.sources], ["x_hi", "y_hi", "z_hi"])

    def test_face_riding_ray_is_zero_measure_touch_run(self):
        # Ray travels exactly on the face y=0 of the box.
        res = ray_intervals(Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, 0, 1), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (1.0, 3.0))
        faces = sorted(s.face for s in touch.sources)
        self.assertEqual(faces, ["x_hi", "x_lo", "y_lo"])  # edge endpoints + face
        self.assertIsNone(res.nearest_boundary)

    def test_parallel_outside_misses(self):
        res = ray_intervals(Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, 5, 1), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.touches, ())


class TestCoincidentEvents(unittest.TestCase):
    def setUp(self):
        self.a = Box((0, 0, 0), (2, 2, 2), "a")
        self.b = Box((2, 0, 0), (4, 2, 2), "b")
        self.ray = Ray((-1, 1, 1), (1, 0, 0))

    def test_union_coincident_faces_no_phantom_boundary(self):
        res = ray_intervals(Union([self.a, self.b]), self.ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(1.0, 5.0)])
        # only the outer boundaries: the shared face at t=3 creates none
        self.assertEqual([b.t for b in res.boundaries], [1.0, 5.0])
        self.assertEqual(res.touches, ())

    def test_difference_shared_face(self):
        res = ray_intervals(Difference(self.a, self.b), self.ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(1.0, 3.0)])
        exit_boundary = res.boundaries[-1]
        self.assertEqual(exit_boundary.kind, "exit")
        # A exits and B enters at the same t=3: both faces are valid sources
        # of the shared boundary; the representative is lexicographic ("a").
        self.assertEqual([s.leaf_id for s in exit_boundary.sources], ["a", "b"])
        self.assertEqual(exit_boundary.source.leaf_id, "a")
        by_leaf = {s.leaf_id: s for s in exit_boundary.sources}
        self.assertFalse(by_leaf["a"].flipped)
        self.assertEqual(by_leaf["a"].face, "x_hi")
        self.assertTrue(by_leaf["b"].flipped)  # subtractor face normal flipped
        self.assertEqual(by_leaf["b"].face, "x_lo")
        self.assertEqual(by_leaf["b"].normal, (1.0, 0.0, 0.0))

    def test_intersection_shared_face_is_empty(self):
        res = ray_intervals(Intersection([self.a, self.b]), self.ray)
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.boundaries, ())

    def test_identical_leaves_union_keeps_all_sources(self):
        s1 = Sphere((0, 0, 0), 2, "a")
        s2 = Sphere((0, 0, 0), 2, "b")
        res = ray_intervals(Union([s1, s2]), Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(3.0, 7.0)])
        for boundary in res.boundaries:
            self.assertEqual([s.leaf_id for s in boundary.sources], ["a", "b"])
            # representative chosen by leaf-id lexicographic order
            self.assertEqual(boundary.source.leaf_id, "a")

    def test_identical_leaves_difference_is_empty(self):
        res = ray_intervals(
            Difference(Sphere((0, 0, 0), 2, "a"), Sphere((0, 0, 0), 2, "b")),
            Ray((-5, 0, 0), (1, 0, 0)),
        )
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.boundaries, ())
        self.assertEqual(res.touches, ())


class TestTransforms(unittest.TestCase):
    def test_translate_subtree_without_rewriting_leaf(self):
        tree = Translate(Box((0, 0, 0), (1, 1, 1), "b"), (10, 0, 0))
        res = ray_intervals(tree, Ray((-1, 0.5, 0.5), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(11.0, 12.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))

    def test_identity_transform_preserves_intervals_and_boundaries(self):
        ray = Ray((-10, 0, 0), (1, 0, 0))
        tree = Difference(
            Sphere((0, 0, 0), 5, "outer"),
            Sphere((0, 0, 0), 2, "inner"),
        )
        plain = ray_intervals(tree, ray)
        transformed = ray_intervals(
            Transform(
                tree,
                matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                translation=(0, 0, 0),
            ),
            ray,
        )
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in transformed.intervals],
            [(iv.t0, iv.t1) for iv in plain.intervals],
        )
        self.assertEqual(
            [b.t for b in transformed.boundaries],
            [b.t for b in plain.boundaries],
        )
        self.assertEqual(
            [s.normal for b in transformed.boundaries for s in b.sources],
            [s.normal for b in plain.boundaries for s in b.sources],
        )

    def test_quarter_rotation(self):
        # Local box x in [0,2] rotates to world y in [0,2]; local y width 1
        # becomes world x in [-1,0].
        tree = Rotate(Box((0, 0, 0), (2, 1, 1), "b"), math.pi / 2, (0, 0, 1))
        res = ray_intervals(tree, Ray((-5, 0.5, 0.5), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 5.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))

    def test_nonuniform_scale_ellipsoid_normals(self):
        tree = Scale(Sphere((0, 0, 0), 1, "s"), (2, 1, 1))
        res = ray_intervals(tree, Ray((0, -5, 0), (0, 1, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])
        self.assertEqual(res.boundaries[0].source.normal, (0.0, -1.0, 0.0))
        self.assertEqual(res.boundaries[1].source.normal, (0.0, 1.0, 0.0))

        # Ray x=y through ellipsoid x^2/4+y^2=1: 5t^2/4=1, t=2/sqrt(5).
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 1, 0)))
        endpoint = 2.0 / math.sqrt(5.0)
        self.assertAlmostEqual(res.intervals[0].t0, -endpoint, places=12)
        self.assertAlmostEqual(res.intervals[0].t1, endpoint, places=12)
        normal = res.boundaries[1].source.normal
        expected = (1.0 / math.sqrt(17.0), 4.0 / math.sqrt(17.0), 0.0)
        for got, wanted in zip(normal, expected):
            self.assertAlmostEqual(got, wanted, places=12)

    def test_nested_transforms_compose(self):
        # Translate the unit box to local y [2,3], then rotate by +90 around
        # z: it occupies world x in [-3,-2].
        tree = Rotate(
            Translate(Box((0, 0, 0), (1, 1, 1), "b"), (0, 2, 0)),
            math.pi / 2,
            (0, 0, 1),
        )
        res = ray_intervals(tree, Ray((0, 0.5, 0.5), (-1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(2.0, 3.0)])

    def test_transformed_subtractor_normal_is_flipped(self):
        tree = Difference(
            Sphere((0, 0, 0), 5, "outer"),
            Translate(Sphere((0, 0, 0), 1, "inner"), (3, 0, 0)),
        )
        res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(5.0, 12.0), (14.0, 15.0)])
        cavity_exit = res.boundaries[1]
        cavity_enter = res.boundaries[2]
        self.assertTrue(cavity_exit.source.flipped)
        self.assertTrue(cavity_enter.source.flipped)
        self.assertEqual(cavity_exit.source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(cavity_enter.source.normal, (-1.0, 0.0, 0.0))

    def test_explicit_affine_matrix_and_singular_rejection(self):
        tree = Transform(
            Box((0, 0, 0), (1, 1, 1), "b"),
            matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
            translation=(0, 5, 0),
        )
        res = ray_intervals(tree, Ray((1, 0, 0.5), (0, 1, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(5.0, 6.0)])
        with self.assertRaises(GeometryError) as ctx:
            Transform(
                Box((0, 0, 0), (1, 1, 1), "b"),
                matrix=((1, 0, 0), (0, 1, 0), (1, 1, 0)),
            )
        self.assertIn("singular", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Scale(Sphere((0, 0, 0), 1, "s"), (1, 0, 1))


class TestInputValidation(unittest.TestCase):
    def test_nan_and_inf_rejected(self):
        with self.assertRaises(GeometryError) as ctx:
            Sphere((0, 0, float("nan")), 1, "s")
        self.assertIn("center[2]", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), float("inf"), "s")
        with self.assertRaises(GeometryError):
            Ray((0, 0, 0), (1, 0, float("nan")))

    def test_bool_rejected_as_number(self):
        with self.assertRaises(GeometryError) as ctx:
            Sphere((0, 0, 0), True, "s")
        self.assertIn("bool", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Ray((0, True, 0), (1, 0, 0))

    def test_degenerate_geometry_rejected(self):
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 0, "s")
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), -1, "s")
        with self.assertRaises(GeometryError) as ctx:
            Box((0, 0, 0), (1, 0, 1), "b")
        self.assertIn("maxs[1]", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Ray((0, 0, 0), (0, 0, 0))

    def test_leaf_id_rules(self):
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, "")
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, 7)
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(
                Union([Sphere((0, 0, 0), 1, "dup"), Sphere((5, 0, 0), 1, "dup")]),
                Ray((0, 0, 0), (1, 0, 0)),
            )
        self.assertIn("dup", str(ctx.exception))

    def test_tree_structure_rules(self):
        shared = Sphere((0, 0, 0), 1, "s")
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(Union([shared, shared]), Ray((0, 0, 0), (1, 0, 0)))
        self.assertIn("shared", str(ctx.exception))
        with self.assertRaises(GeometryError):
            ray_intervals(42, Ray((0, 0, 0), (1, 0, 0)))
        with self.assertRaises(GeometryError):
            ray_intervals(Sphere((0, 0, 0), 1, "s"), "not a ray")
        with self.assertRaises(GeometryError):
            Union([])
        with self.assertRaises(GeometryError):
            Difference(Sphere((0, 0, 0), 1, "s"))

    def test_node_budget(self):
        leaves = [Sphere((0, 0, 0), 1, f"L{i}") for i in range(MAX_TREE_NODES - 1)]
        tree = Union(leaves)  # exactly MAX_TREE_NODES nodes: accepted
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(len(res.intervals), 1)
        too_many = Union(
            [Sphere((0, 0, 0), 1, f"M{i}") for i in range(MAX_TREE_NODES)]
        )
        with self.assertRaises(GeometryError) as ctx:
            ray_intervals(too_many, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertIn(str(MAX_TREE_NODES), str(ctx.exception))

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(GeometryError, CSGError))
        self.assertTrue(issubclass(CertificationBudgetExhausted, CSGError))
        self.assertTrue(issubclass(CSGError, ValueError))


if __name__ == "__main__":
    unittest.main()
