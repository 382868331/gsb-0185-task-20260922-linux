"""Analytic tests for nestable translate / rotate / scale transforms.

Expected endpoints and normals are derived by hand; the randomized
membership cross-check (including shears) lives in test_sampling.py.
"""

import math
import unittest

from csg_rays import (
    MAX_TREE_NODES,
    Box,
    Difference,
    GeometryError,
    Ray,
    Rotate,
    Scale,
    Sphere,
    Transform,
    Translate,
    Union,
    ray_intervals,
)


class TestTranslate(unittest.TestCase):
    def test_translated_sphere_endpoints_and_normals(self):
        # unit sphere moved to center (10, 0, 0); x-axis ray from origin
        tree = Translate(Sphere((0, 0, 0), 2, "s"), (10, 0, 0))
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(8.0, 12.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(res.nearest_boundary.t, 8.0)

    def test_nested_translation_accumulates(self):
        tree = Translate(
            Translate(Sphere((0, 0, 0), 1, "s"), (1, 2, 3)),
            (10, 0, 0),
        )
        res = ray_intervals(tree, Ray((11, 2, 0), (0, 0, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(2.0, 4.0)])

    def test_translate_whole_union(self):
        # two unit spheres centered at (0,0,0) and (4,0,0), shifted by (-10,5,0)
        tree = Translate(
            Union([
                Sphere((0, 0, 0), 1, "a"),
                Sphere((4, 0, 0), 1, "b"),
            ]),
            (-10, 5, 0),
        )
        res = ray_intervals(tree, Ray((0, 5, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(-11.0, -9.0), (-7.0, -5.0)],
        )


class TestNonUniformScale(unittest.TestCase):
    def test_ellipsoid_endpoints(self):
        # unit sphere scaled by (2, 4, 1): ellipsoid x^2/4 + y^2/16 + z^2 <= 1
        tree = Scale(Sphere((0, 0, 0), 1, "e"), (2, 4, 1))
        res = ray_intervals(tree, Ray((-10, 1.0, 0), (1, 0, 0)))
        root = math.sqrt(4.0 * (1.0 - 1.0 / 16.0))
        self.assertAlmostEqual(res.intervals[0].t0, 10.0 - root, places=12)
        self.assertAlmostEqual(res.intervals[0].t1, 10.0 + root, places=12)

    def test_scaled_box_interval(self):
        # [-1,1]^3 box scaled by (2, 4, 1) -> x in [-2,2], y in [-4,4]
        tree = Scale(Box((-1, -1, -1), (1, 1, 1), "b"), (2, 4, 1))
        res = ray_intervals(tree, Ray((0, -10, 0), (0, 1, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(6.0, 14.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.face, "y_lo")
        self.assertEqual(enter.source.normal, (0.0, -1.0, 0.0))
        self.assertEqual(exit_.source.normal, (0.0, 1.0, 0.0))

    def test_slanted_normal_under_shear(self):
        # shear maps local x face to a slanted world face:
        # L = [[1,1,0],[0,1,0],[0,0,1]], inverse = [[1,-1,0],[0,1,0],[0,0,1]]
        # local x_lo normal (1,0,0) -> L^{-T} (1,0,0) = (1,0,0) (unchanged);
        # local y_lo normal (0,1,0) -> (-1,1,0)/sqrt(2).
        shear = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
        tree = Transform(Box((0, 0, 0), (1, 1, 1), "b"), linear=shear)
        # world point (c,-10+t) maps to local (c-(-10+t), -10+t). With
        # c=3/2 the solid overlap is t in [10.5, 11]; the t=11 end is the
        # slanted local y_hi face.
        res = ray_intervals(tree, Ray((1.5, -10, 0.5), (0, 1, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(10.5, 11.0)])
        enter, exit_ = res.boundaries
        # enter crosses the slanted local x_hi face; exit is local y_hi
        # (0,1,0), which stays vertical under this shear.
        self.assertEqual(enter.source.face, "x_hi")
        n = enter.source.normal
        self.assertAlmostEqual(n[0], 1.0 / math.sqrt(2.0), places=14)
        self.assertAlmostEqual(n[1], -1.0 / math.sqrt(2.0), places=14)
        self.assertEqual(n[2], 0.0)
        self.assertEqual(exit_.source.face, "y_hi")
        self.assertEqual(exit_.source.normal, (0.0, 1.0, 0.0))

    def test_negative_scale_is_reflection(self):
        # scale x by -1: sphere centered at local (1,0,0) maps to (-1,0,0)
        tree = Scale(Sphere((1, 0, 0), 1, "s"), (-1, 1, 1))
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-2.0, 0.0)])
        enter = res.boundaries[0]
        self.assertAlmostEqual(enter.source.normal[0], -1.0, places=14)


class TestRotate(unittest.TestCase):
    def test_quarter_turn_box(self):
        # 90 deg about z turns box x in [0,2], y in [0,2] into
        # world x in [-2,0], y in [0,2].
        r = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        tree = Rotate(Box((0, 0, -1), (2, 2, 1), "b"), r)
        res = ray_intervals(tree, Ray((-1, -10, 0), (0, 1, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(10.0, 12.0)])
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.face, "x_lo")
        self.assertEqual(enter.source.normal, (0.0, -1.0, 0.0))
        self.assertEqual(exit_.source.face, "x_hi")
        self.assertEqual(exit_.source.normal, (0.0, 1.0, 0.0))

    def test_float_rotation_matrix_accepted(self):
        c, s = math.cos(0.7), math.sin(0.7)
        r = ((c, -s, 0), (s, c, 0), (0, 0, 1))
        tree = Rotate(Sphere((0, 0, 0), 2, "s"), r)
        res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
        self.assertAlmostEqual(res.intervals[0].t0, 8.0, places=10)
        self.assertAlmostEqual(res.intervals[0].t1, 12.0, places=10)
        # rotated sphere normals lie along world x at the axis crossings
        self.assertAlmostEqual(res.boundaries[0].source.normal[0], -1.0, places=10)


class TestTransformsWithBoolean(unittest.TestCase):
    def test_inner_surface_normal_of_transformed_cavity(self):
        hollow = Difference(
            Sphere((0, 0, 0), 5, "outer"),
            Sphere((0, 0, 0), 2, "inner"),
        )
        # scale x by 2, then translate to (0, 100, 0)
        tree = Translate(Scale(hollow, (2, 1, 1)), (0, 100, 0))
        res = ray_intervals(tree, Ray((-20, 100, 0), (1, 0, 0)))
        # outer x crossings at +/-10, cavity walls at +/-4
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(10.0, 16.0), (24.0, 30.0)],
        )
        b10, b16, b24, b30 = res.boundaries
        self.assertEqual(b16.source.leaf_id, "inner")
        self.assertTrue(b16.source.flipped)
        # inward-facing cavity wall: normal points +x (into the cavity)
        self.assertEqual(b16.source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(b24.source.normal, (-1.0, 0.0, 0.0))


class TestIdentityPreservesResults(unittest.TestCase):
    def test_identity_transform_matches_plain_tree(self):
        plain = Difference(
            Box((0, 0, 0), (2, 2, 2), "a"),
            Box((2, 0, 0), (4, 2, 2), "b"),
        )
        ray = Ray((-1, 1, 1), (1, 0, 0))
        res_plain = ray_intervals(plain, ray)
        res_wrap = ray_intervals(Transform(plain), ray)
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res_plain.intervals],
            [(iv.t0, iv.t1) for iv in res_wrap.intervals],
        )
        for bp, bw in zip(res_plain.boundaries, res_wrap.boundaries):
            self.assertEqual(bp.kind, bw.kind)
            self.assertEqual(bp.source.normal, bw.source.normal)
            self.assertEqual(
                [s.face for s in bp.sources], [s.face for s in bw.sources]
            )


class TestTransformValidation(unittest.TestCase):
    def test_singular_transforms_rejected_before_computation(self):
        with self.assertRaises(GeometryError) as ctx:
            Scale(Sphere((0, 0, 0), 1, "s"), (1, 0, 2))
        self.assertIn("non-zero", str(ctx.exception))
        singular = ((1, 2, 3), (4, 5, 6), (7, 8, 9))  # det == 0
        with self.assertRaises(GeometryError) as ctx:
            Transform(Sphere((0, 0, 0), 1, "s"), linear=singular)
        self.assertIn("singular", str(ctx.exception))

    def test_bad_child_and_matrix_shape_rejected(self):
        with self.assertRaises(GeometryError):
            Translate(5, (0, 0, 0))
        with self.assertRaises(GeometryError):
            Rotate(Sphere((0, 0, 0), 1, "s"), ((1, 0, 0), (0, 1, 0)))
        with self.assertRaises(GeometryError):
            Transform(
                Sphere((0, 0, 0), 1, "s"),
                linear=((1, 0, 0), (0, "x", 0), (0, 0, 1)),
            )
        with self.assertRaises(GeometryError):
            Translate(Sphere((0, 0, 0), 1, "s"), (1, 2, float("nan")))

    def test_transform_nodes_count_toward_budget(self):
        # 1 leaf + (MAX-1) transforms == MAX nodes: accepted
        tree = Sphere((0, 0, 0), 1, "s")
        for _ in range(MAX_TREE_NODES - 1):
            tree = Translate(tree, (0, 0, 0))
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(len(res.intervals), 1)
        tree = Translate(tree, (1, 0, 0))  # one node over the limit
        with self.assertRaises(GeometryError):
            ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))

    def test_duplicate_leaf_ids_under_transform_still_rejected(self):
        tree = Translate(
            Union([Sphere((0, 0, 0), 1, "dup"), Sphere((5, 0, 0), 1, "dup")]),
            (0, 0, 0),
        )
        with self.assertRaises(GeometryError):
            ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
