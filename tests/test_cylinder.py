"""Analytic tests for the finite z-aligned Cylinder leaf.

Expected values are derived by hand from the implicit equation
(x-cx)^2 + (y-cy)^2 = r^2 clipped to z_lo <= z <= z_hi, never by calling
the implementation under test.  Membership cross-checks substitute hit
points back into the implicit form and sample open-interval midpoints.
"""

import unittest
from fractions import Fraction

from csg_rays import (
    Box,
    Cylinder,
    Difference,
    GeometryError,
    Intersection,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)


def _cyl_membership(cyl, origin, direction, t):
    """Closed-solid membership of origin + t*direction via the implicit form."""
    t = Fraction(t)
    x = Fraction(origin[0]) + t * Fraction(direction[0])
    y = Fraction(origin[1]) + t * Fraction(direction[1])
    z = Fraction(origin[2]) + t * Fraction(direction[2])
    radial = (x - cyl.center[0]) ** 2 + (y - cyl.center[1]) ** 2
    return radial <= cyl.radius**2 and cyl.z_lo <= z <= cyl.z_hi


class TestCylinderBasics(unittest.TestCase):
    def setUp(self):
        # radius 1, caps at z=-2 and z=+3 (asymmetric heights)
        self.cyl = Cylinder((0, 0, 0), 1, -2, 3, "c")

    def test_side_crossing(self):
        res = ray_intervals(self.cyl, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])
        enter, exit_ = res.boundaries
        self.assertEqual([s.face for s in enter.sources], ["side"])
        self.assertEqual(enter.source.normal, (-1.0, 0.0, 0.0))
        self.assertEqual([s.face for s in exit_.sources], ["side"])
        self.assertEqual(exit_.source.normal, (1.0, 0.0, 0.0))

    def test_cap_crossing_along_axis(self):
        # ray along the axis, inside the tube: only the caps are crossed
        res = ray_intervals(self.cyl, Ray((0, 0, -10), (0, 0, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(8.0, 13.0)])
        self.assertEqual([s.face for s in res.boundaries[0].sources], ["z_lo"])
        self.assertEqual(res.boundaries[0].source.normal, (0.0, 0.0, -1.0))
        self.assertEqual([s.face for s in res.boundaries[1].sources], ["z_hi"])
        self.assertEqual(res.boundaries[1].source.normal, (0.0, 0.0, 1.0))

    def test_axis_ray_outside_tube_misses(self):
        res = ray_intervals(self.cyl, Ray((2, 0, 0), (0, 0, 1)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.touches, ())

    def test_oblique_ray_clipped_by_cap(self):
        # origin (-5,0,-3), dir (1,0,1): radial roots at t=4 and t=6,
        # z=3 cap at t=6 -> exit lists side and z_hi at the same t.
        res = ray_intervals(self.cyl, Ray((-5, 0, -3), (1, 0, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])
        exit_sources = res.boundaries[-1].sources
        self.assertEqual(sorted(s.face for s in exit_sources), ["side", "z_hi"])
        for s in exit_sources:
            self.assertEqual(s.kind, "exit")

    def test_implicit_form_and_midpoint_membership(self):
        # every boundary point satisfies the implicit side/cap equation and
        # every open-interval midpoint is strictly inside the closed solid
        for ray in (
            Ray((-5, 0, 0), (1, 0, 0)),
            Ray((0, 0, -10), (0, 0, 1)),
            Ray((-5, 0, -3), (1, 0, 1)),
            Ray((-5, 1, -3), (1, 0, 1)),
        ):
            res = ray_intervals(self.cyl, ray)
            for b in res.boundaries:
                t = Fraction(b.t)
                x = Fraction(ray.origin[0]) + t * Fraction(ray.direction[0])
                y = Fraction(ray.origin[1]) + t * Fraction(ray.direction[1])
                z = Fraction(ray.origin[2]) + t * Fraction(ray.direction[2])
                radial = (x - self.cyl.center[0]) ** 2 + (y - self.cyl.center[1]) ** 2
                on_side = radial == self.cyl.radius**2
                on_cap = z == self.cyl.z_lo or z == self.cyl.z_hi
                self.assertTrue(on_side or on_cap, f"boundary {b} off surface")
                self.assertTrue(self.cyl.z_lo <= z <= self.cyl.z_hi)
            for iv in res.intervals:
                mid = (iv.t0 + iv.t1) / 2.0
                self.assertTrue(
                    _cyl_membership(self.cyl, ray.origin, ray.direction, mid)
                )


class TestCylinderDegenerateRays(unittest.TestCase):
    def setUp(self):
        self.cyl = Cylinder((0, 0, 0), 1, -2, 3, "c")

    def test_tangent_to_side_is_touch(self):
        res = ray_intervals(self.cyl, Ray((-5, 1, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (5.0, 5.0))
        self.assertEqual([s.face for s in touch.sources], ["side"])
        self.assertEqual(touch.source.normal, (0.0, 1.0, 0.0))
        self.assertIsNone(res.nearest_boundary)

    def test_tangent_outside_z_range_misses(self):
        # tangent line at y=1, z=10: above the top cap
        res = ray_intervals(self.cyl, Ray((-5, 1, 10), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(res.touches, ())

    def test_ray_riding_side_surface(self):
        # ray on the side surface along the axis: zero-thickness contact run
        res = ray_intervals(self.cyl, Ray((1, 0, -10), (0, 0, 1)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (8.0, 13.0))
        self.assertEqual(sorted(s.face for s in touch.sources), ["side", "z_hi", "z_lo"])

    def test_ray_on_cap_plane(self):
        # ray travels on the bottom cap plane through the disk
        res = ray_intervals(self.cyl, Ray((-5, 0, -2), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (4.0, 6.0))
        self.assertEqual(sorted(s.face for s in touch.sources), ["side", "z_lo"])

    def test_cap_edge_only_touch(self):
        # ray on the cap plane, tangent to the rim circle: one isolated touch
        res = ray_intervals(self.cyl, Ray((-5, 1, -2), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 1)
        touch = res.touches[0]
        self.assertEqual((touch.t0, touch.t1), (5.0, 5.0))
        self.assertEqual(sorted(s.face for s in touch.sources), ["side", "z_lo"])

    def test_interior_origin(self):
        res = ray_intervals(self.cyl, Ray((0, 0, 0), (0, 0, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-2.0, 3.0)])
        self.assertEqual(res.nearest_boundary.t, 3.0)
        self.assertEqual(res.nearest_boundary.kind, "exit")
        self.assertEqual(res.nearest_boundary.source.face, "z_hi")


class TestCylinderBoolean(unittest.TestCase):
    def test_cylinder_minus_sphere(self):
        tree = Difference(
            Cylinder((0, 0, 0), 1, -2, 3, "c"),
            Sphere((0, 0, 0), 0.5, "s"),
        )
        res = ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals], [(-1.0, -0.5), (0.5, 1.0)]
        )
        # cavity walls: subtractor side faces with flipped normals
        cavity = [b for b in res.boundaries if b.source.leaf_id == "s"]
        self.assertEqual(len(cavity), 2)
        for b in cavity:
            self.assertTrue(b.source.flipped)
        self.assertEqual(cavity[0].source.normal, (1.0, 0.0, 0.0))
        self.assertEqual(cavity[1].source.normal, (-1.0, 0.0, 0.0))

    def test_sphere_minus_cylinder_flips_cylinder_normals(self):
        tree = Difference(
            Sphere((0, 0, 0), 3, "s"),
            Cylinder((0, 0, 0), 1, -2, 3, "c"),
        )
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals], [(2.0, 4.0), (6.0, 8.0)]
        )
        cyl_sources = [
            s for b in res.boundaries for s in b.sources if s.leaf_id == "c"
        ]
        self.assertEqual(len(cyl_sources), 2)
        for s in cyl_sources:
            self.assertEqual(s.face, "side")
            self.assertTrue(s.flipped)
        # flipped outward side normals point into the cylindrical bore
        self.assertEqual(cyl_sources[0].normal, (1.0, 0.0, 0.0))
        self.assertEqual(cyl_sources[1].normal, (-1.0, 0.0, 0.0))

    def test_box_intersection_cylinder(self):
        box = Box((-1, -1, -5), (1, 1, 1), "b")
        cyl = Cylinder((0, 0, 0), 1, -2, 3, "c")
        res = ray_intervals(
            Intersection([box, cyl]), Ray((0, -5, 0), (0, 1, 0))
        )
        # box clips y to [-1, 1]; cylinder z-range not limiting here
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(4.0, 6.0)])

    def test_zero_measure_contact_union(self):
        # ray tangent to the cylinder side while crossing a sphere: the
        # tangent is swallowed inside the solid, no phantom boundary
        tree = Union(
            [Sphere((0, 0, 0), 3, "s"), Cylinder((0, 0, 0), 1, -2, 3, "c")]
        )
        res = ray_intervals(tree, Ray((-5, 1, 0), (1, 0, 0)))
        root = 8 ** 0.5
        self.assertEqual(len(res.intervals), 1)
        iv = res.intervals[0]
        self.assertAlmostEqual(iv.t0, 5.0 - root, places=12)
        self.assertAlmostEqual(iv.t1, 5.0 + root, places=12)
        self.assertEqual(res.touches, ())


class TestCylinderValidation(unittest.TestCase):
    def test_radius_must_be_positive(self):
        with self.assertRaises(GeometryError) as ctx:
            Cylinder((0, 0, 0), 0, -1, 1, "c")
        self.assertIn("radius", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), -2, -1, 1, "c")

    def test_heights_must_differ(self):
        with self.assertRaises(GeometryError) as ctx:
            Cylinder((0, 0, 0), 1, 2, 2, "c")
        self.assertIn("z_hi", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, 3, -1, "c")

    def test_nan_inf_bool_rejected(self):
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, float("nan")), 1, -1, 1, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), float("inf"), -1, 1, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, True, 1, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, -1, float("nan"), "c")

    def test_leaf_id_rules(self):
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, -1, 1, "")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, -1, 1, 3)

    def test_error_path_is_locatable(self):
        with self.assertRaises(GeometryError) as ctx:
            Cylinder((0, 0, "x"), 1, -1, 1, "cyl9")
        self.assertIn("Cylinder(leaf_id='cyl9').center[2]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
