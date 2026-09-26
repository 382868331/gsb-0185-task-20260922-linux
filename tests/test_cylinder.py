"""Analytic and independent-reference tests for the finite cylinder leaf."""

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


def point(origin, direction, t):
    if not isinstance(t, Fraction):
        t = Fraction(t)
    return tuple(Fraction(origin[i]) + t * Fraction(direction[i]) for i in range(3))


def in_cylinder(cylinder, p, closed=False):
    radial = (
        (p[0] - cylinder.center[0]) ** 2
        + (p[1] - cylinder.center[1]) ** 2
    )
    z_min = cylinder.center[2] - cylinder.lower_height
    z_max = cylinder.center[2] + cylinder.upper_height
    radial_ok = radial <= cylinder.radius**2 if closed else radial < cylinder.radius**2
    z_ok = z_min <= p[2] <= z_max if closed else z_min < p[2] < z_max
    return radial_ok and z_ok


def ref_inside(node, p, closed=False):
    if isinstance(node, Cylinder):
        return in_cylinder(node, p, closed)
    if isinstance(node, Sphere):
        d2 = sum((p[i] - node.center[i]) ** 2 for i in range(3))
        return d2 <= node.radius**2 if closed else d2 < node.radius**2
    if isinstance(node, Box):
        if closed:
            return all(node.mins[i] <= p[i] <= node.maxs[i] for i in range(3))
        return all(node.mins[i] < p[i] < node.maxs[i] for i in range(3))
    if isinstance(node, Union):
        return any(ref_inside(child, p, closed) for child in node.children)
    if isinstance(node, Intersection):
        return all(ref_inside(child, p, closed) for child in node.children)
    if isinstance(node, Difference):
        return ref_inside(node.positive, p, closed) and not any(
            ref_inside(child, p, True) for child in node.subtractors
        )
    raise AssertionError("unknown CSG node")


def assert_midpoints_match(test_case, tree, ray, result):
    boundaries = [Fraction(0)]
    for interval in result.intervals:
        boundaries.extend((Fraction(interval.t0), Fraction(interval.t1)))
    boundaries.sort()
    if not result.intervals:
        samples = [Fraction(-1, 2), Fraction(0), Fraction(1, 2)]
    else:
        samples = []
        first = Fraction(result.intervals[0].t0)
        last = Fraction(result.intervals[-1].t1)
        samples.extend((first - 2, first - Fraction(1, 2)))
        for left, right in zip(result.intervals, result.intervals[1:]):
            samples.append(
                (Fraction(left.t1) + Fraction(right.t0)) / 2
            )
        samples.extend((last + Fraction(1, 2), last + 2))
        for interval in result.intervals:
            samples.append(
                (Fraction(interval.t0) + Fraction(interval.t1)) / 2
            )
    for sample_t in samples:
        sample_point = point(ray.origin, ray.direction, sample_t)
        inside_interval = any(
            Fraction(iv.t0) < sample_t < Fraction(iv.t1)
            for iv in result.intervals
        )
        test_case.assertIs(
            ref_inside(tree, sample_point),
            inside_interval,
            (sample_t, sample_point),
        )


class TestCylinderAnalytic(unittest.TestCase):
    def setUp(self):
        self.c = Cylinder((0, 0, 0), 2, 1, 3, "c")

    def test_axis_rays_use_caps_and_define_interior_origin(self):
        up = ray_intervals(self.c, Ray((0, 0, 0), (0, 0, 1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in up.intervals], [(-1.0, 3.0)])
        self.assertEqual(
            [s.face for b in up.boundaries for s in b.sources],
            ["z_lo", "z_hi"],
        )
        self.assertEqual(up.boundaries[0].source.normal, (0.0, 0.0, -1.0))
        self.assertEqual(up.boundaries[1].source.normal, (0.0, 0.0, 1.0))

        down = ray_intervals(self.c, Ray((0, 0, 0), (0, 0, -1)))
        self.assertEqual([(iv.t0, iv.t1) for iv in down.intervals], [(-3.0, 1.0)])
        self.assertEqual(
            [b.source.face for b in down.boundaries], ["z_hi", "z_lo"]
        )
        assert_midpoints_match(
            self, self.c, Ray((0, 0, 0), (0, 0, 1)), up
        )

    def test_side_events_and_exact_implicit_check(self):
        cylinder = Cylinder((0, 0, 0), 5, 4, 6, "c")
        ray = Ray((0, 3, 0), (4, 0, 3))
        result = ray_intervals(cylinder, ray)
        # Radial roots are t=-1 and t=1; z=-4 and z=6 occur at -4/3 and 2.
        self.assertEqual([(iv.t0, iv.t1) for iv in result.intervals], [(-1.0, 1.0)])
        start, end = result.boundaries
        self.assertEqual(start.source.face, "side")
        self.assertEqual(end.source.face, "side")
        for t in (Fraction(-1), Fraction(1)):
            p = point(ray.origin, ray.direction, t)
            self.assertEqual(p[0] ** 2 + p[1] ** 2, cylinder.radius**2)
            self.assertTrue(-cylinder.lower_height <= p[2] <= cylinder.upper_height)
        assert_midpoints_match(self, cylinder, ray, result)

    def test_side_riding_is_touch_segment_with_rim_caps(self):
        ray = Ray((2, 0, 0), (0, 0, 1))
        result = ray_intervals(self.c, ray)
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        touch = result.touches[0]
        self.assertEqual((touch.t0, touch.t1), (-1.0, 3.0))
        self.assertEqual(
            sorted(s.face for s in touch.sources),
            ["side", "z_hi", "z_lo"],
        )
        self.assertTrue(all(s.kind == "touch" for s in touch.sources))
        for t in (-1.0, 0.0, 3.0):
            self.assertFalse(
                ref_inside(self.c, point(ray.origin, ray.direction, t))
            )

    def test_cap_plane_contacts_distinguish_interior_and_rim(self):
        interior = ray_intervals(self.c, Ray((0, 0, 3), (1, 0, 0)))
        self.assertEqual(interior.intervals, ())
        self.assertEqual((interior.touches[0].t0, interior.touches[0].t1), (-2.0, 2.0))
        self.assertEqual(
            sorted(s.face for s in interior.touches[0].sources),
            ["side", "z_hi"],
        )

        rim = ray_intervals(self.c, Ray((2, 0, 3), (0, 1, 0)))
        self.assertEqual(rim.intervals, ())
        self.assertEqual(
            sorted(s.face for s in rim.touches[0].sources),
            ["side", "z_hi"],
        )
        self.assertEqual((rim.touches[0].t0, rim.touches[0].t1), (0.0, 0.0))
        self.assertEqual(
            rim.touches[0].source.normal, (1.0, 0.0, 0.0)
        )

    def test_cap_plane_contacts_need_not_be_at_origin(self):
        # This ray crosses the upper cap disk for 3 < t < 7.
        crossing = ray_intervals(self.c, Ray((-5, 0, 3), (1, 0, 0)))
        self.assertEqual(crossing.intervals, ())
        self.assertEqual(
            (crossing.touches[0].t0, crossing.touches[0].t1),
            (3.0, 7.0),
        )

        # This ray is tangent to the upper rim at t=5, away from the origin.
        tangent = ray_intervals(self.c, Ray((-3, 2, 3), (1, 0, 0)))
        self.assertEqual(tangent.intervals, ())
        self.assertEqual(
            (tangent.touches[0].t0, tangent.touches[0].t1),
            (3.0, 3.0),
        )
        self.assertEqual(
            sorted(s.face for s in tangent.touches[0].sources),
            ["side", "z_hi"],
        )

    def test_diagonal_rim_enter_or_exit_batches_side_and_cap(self):
        # At t=0 the ray starts on the lower rim and immediately enters.
        entering = ray_intervals(
            self.c, Ray((2, 0, -1), (-1, 0, 1))
        )
        self.assertEqual(entering.intervals[0].t0, 0.0)
        self.assertEqual(
            sorted(s.face for s in entering.boundaries[0].sources),
            ["side", "z_lo"],
        )

        # At t=0 the ray is on the upper rim and immediately leaves.
        leaving = ray_intervals(self.c, Ray((2, 0, 3), (1, 0, -1)))
        self.assertEqual(leaving.intervals, ())
        touch = leaving.touches[0]
        self.assertEqual(touch.t0, 0.0)
        self.assertEqual(
            sorted(s.face for s in touch.sources), ["side", "z_hi"]
        )

    def test_cylinder_minus_sphere_flips_sphere_normals(self):
        cylinder = Cylinder((0, 0, 0), 3, 4, 6, "tube")
        sphere = Sphere((0, 0, 0), 2, "ball")
        tree = Difference(cylinder, sphere)
        ray = Ray((-10, 0, 0), (1, 0, 0))
        result = ray_intervals(tree, ray)
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in result.intervals],
            [(7.0, 8.0), (12.0, 13.0)],
        )
        by_t = {b.t: b for b in result.boundaries}
        self.assertEqual(by_t[7.0].source.face, "side")
        self.assertFalse(by_t[7.0].source.flipped)
        self.assertEqual(by_t[8.0].source.leaf_id, "ball")
        self.assertTrue(by_t[8.0].source.flipped)
        self.assertEqual(by_t[8.0].source.normal, (1.0, 0.0, 0.0))
        assert_midpoints_match(self, tree, ray, result)

    def test_box_intersection_batches_coincident_sources(self):
        cylinder = Cylinder((0, 0, 0), 2, 1, 3, "tube")
        box = Box((-2, -2, -1), (2, 2, 3), "box")
        tree = Intersection([cylinder, box])
        ray = Ray((-5, 0, 0), (1, 0, 0))
        result = ray_intervals(tree, ray)
        self.assertEqual([(iv.t0, iv.t1) for iv in result.intervals], [(3.0, 7.0)])
        self.assertEqual(
            sorted(s.face for s in result.boundaries[0].sources),
            ["side", "x_lo"],
        )
        self.assertEqual(
            sorted(s.face for s in result.boundaries[1].sources),
            ["side", "x_hi"],
        )
        assert_midpoints_match(self, tree, ray, result)

    def test_zero_measure_intersection_contact(self):
        cylinder = Cylinder((0, 0, 0), 2, 1, 3, "tube")
        # The box shares only the upper-rim circle at x=2, z=3.
        box = Box((2, -2, 3), (4, 2, 5), "box")
        tree = Intersection([cylinder, box])
        ray = Ray((2, 0, -5), (0, 0, 1))
        result = ray_intervals(tree, ray)
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        self.assertEqual(result.touches[0].t0, 8.0)
        faces = {s.face for s in result.touches[0].sources}
        self.assertIn("side", faces)
        self.assertIn("z_hi", faces)
        assert_midpoints_match(self, tree, ray, result)

    def test_validation(self):
        with self.assertRaises(GeometryError) as ctx:
            Cylinder((0, 0, 0), 0, 1, 2, "c")
        self.assertIn("radius", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, 0, 2, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, 1, -2, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, 2, 2, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), True, 1, 2, "c")
        with self.assertRaises(GeometryError):
            Cylinder((0, 0, 0), 1, 1, 2, "")


if __name__ == "__main__":
    unittest.main()
