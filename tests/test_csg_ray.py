"""Tests for csg_ray.

Expectations are produced by an *independent* strict point-membership
reference implemented below; it never imports or calls ray_intervals.
Round-trip style checks are only supplementary.  Analytic concentric hollow
spheres provide exact endpoint checks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csg_ray import (  # noqa: E402
    MAX_NODES,
    Box,
    Boundary,
    BudgetExceededError,
    Difference,
    Intersection,
    InvalidGeometryError,
    InvalidRayError,
    Ray,
    Sphere,
    Touch,
    Union,
    ray_intervals,
    solid_at,
)


# --------------------------------------------------------------------------- #
# Independent reference: strict 3D point membership (never uses the core).
# --------------------------------------------------------------------------- #


def ref_inside(node, p):
    """Strict interior membership of point p in the closed-set CSG tree.

    Boundary points (sphere surface, box faces) are NOT interior.
    """
    x, y, z = p
    if isinstance(node, Sphere):
        cx, cy, cz = node.center
        dx, dy, dz = x - cx, y - cy, z - cz
        return dx * dx + dy * dy + dz * dz < node.radius * node.radius
    if isinstance(node, Box):
        return (node.min_corner[0] < x < node.max_corner[0]
                and node.min_corner[1] < y < node.max_corner[1]
                and node.min_corner[2] < z < node.max_corner[2])
    l = ref_inside(node.left, p)
    r = ref_inside(node.right, p)
    if isinstance(node, Union):
        return l or r
    if isinstance(node, Intersection):
        return l and r
    return l and not r


def core_open_inside(result, t):
    """t strictly inside a reported interval."""
    return any(iv.t_enter < t < iv.t_exit for iv in result.intervals)


def sample_check(tree, ray, ts, case):
    """Compare the core against the independent reference on a t grid."""
    result = ray_intervals(tree, ray)
    for t in ts:
        p = tuple(ray.origin[k] + ray.direction[k] * t for k in range(3))
        expected = ref_inside(tree, p)
        got = core_open_inside(result, t)
        case.assertEqual(
            got, expected,
            msg=f"t={t!r} p={p}: core open-interval={got}, independent "
                f"strict interior={expected}")
    return result


def _grid():
    # Irregular deterministic grid avoiding likely exact endpoints.
    ts = []
    for k in range(-40, 41):
        ts.append(k * 0.37)
        ts.append(k * 0.37 + 0.11)
    return ts


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

SPHERE_OUTER = Sphere("outer", (0.0, 0.0, 0.0), 5.0)
SPHERE_INNER = Sphere("inner", (0.0, 0.0, 0.0), 2.0)
BOX_A = Box("a", (0.0, 0.0, -1.0), (2.0, 1.0, 1.0))
BOX_B = Box("b", (2.0, 0.0, -1.0), (4.0, 1.0, 1.0))
BOX_C = Box("c", (1.0, -2.0, -2.0), (3.0, 2.0, 2.0))


class TestPrimitivesAndBooleansBySampling(unittest.TestCase):
    """Small-sampling acceptance: interval *interiors* vs reference."""

    GRID = _grid()

    def test_sphere(self):
        ray = Ray((-7.0, 0.3, -0.2), (1.0, 0.0, 0.0))
        result = sample_check(SPHERE_OUTER, ray, self.GRID, self)
        self.assertEqual(len(result.intervals), 1)

    def test_box(self):
        ray = Ray((-3.0, 0.4, 0.6), (1.0, 0.0, 0.0))
        sample_check(BOX_A, ray, self.GRID, self)

    def test_union(self):
        tree = Union(SPHERE_OUTER, BOX_C)
        ray = Ray((-8.0, 0.5, -0.4), (1.0, 0.13, 0.07))
        result = sample_check(tree, ray, self.GRID, self)
        for iv in result.intervals:
            self.assertLess(iv.t_enter, iv.t_exit)
            self.assertTrue(iv.enter_sources and iv.exit_sources)

    def test_intersection(self):
        tree = Intersection(SPHERE_OUTER, BOX_C)
        ray = Ray((-6.0, 0.2, 0.1), (1.0, -0.09, 0.17))
        sample_check(tree, ray, self.GRID, self)

    def test_difference(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        ray = Ray((-7.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        result = sample_check(tree, ray, self.GRID, self)
        self.assertEqual(len(result.intervals), 2)

    def test_nested_with_cavity(self):
        cavity = Union(SPHERE_INNER,
                       Box("slit", (-1.0, 2.0, -0.5), (1.0, 6.0, 0.5)))
        tree = Difference(SPHERE_OUTER, cavity)
        ray = Ray((-7.0, 0.25, 0.1), (1.0, 0.0, 0.0))
        sample_check(tree, ray, self.GRID, self)

    def test_negative_direction(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        ray = Ray((7.0, 0.4, -0.3), (-1.0, 0.0, 0.0))
        sample_check(tree, ray, self.GRID, self)


class TestExactEndpointsConcentricHollowSphere(unittest.TestCase):
    """Analytic concentric hollow sphere: exact endpoint verification."""

    def test_ray_through_center_from_inside(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        result = ray_intervals(tree, Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        ends = [(iv.t_enter, iv.t_exit) for iv in result.intervals]
        # Exact equality: endpoints are rational and produced via Fraction.
        self.assertEqual(ends, [(-5.0, -2.0), (2.0, 5.0)])

    def test_ray_starting_outside(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        result = ray_intervals(tree, Ray((-7.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        ends = [(iv.t_enter, iv.t_exit) for iv in result.intervals]
        self.assertEqual(ends, [(2.0, 5.0), (9.0, 12.0)])

    def test_cavity_face_normals_flipped(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        result = ray_intervals(tree, Ray((-7.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        first, second = result.intervals

        # Cavity wall at t=5 (exiting solid into the void): point x=-2,
        # inner sphere's own outward normal is -x; the flipped effective
        # outward normal of the result points into the cavity, i.e. +x.
        self.assertEqual(first.t_exit, 5.0)
        rep = first.representative_exit
        self.assertEqual(rep.leaf_id, "inner")
        self.assertEqual(rep.face, "surface")
        self.assertTrue(rep.flipped)
        self.assertEqual(rep.normal, (1.0, 0.0, 0.0))

        # Outer wall normals are unflipped.
        self.assertEqual(first.representative_enter.leaf_id, "outer")
        self.assertFalse(first.representative_enter.flipped)
        self.assertEqual(first.representative_enter.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(second.representative_exit.normal, (1.0, 0.0, 0.0))

        # Re-entering solid at t=9 is the far side of the inner sphere:
        # inner normal +x at x=2, flipped to -x.
        self.assertEqual(second.t_enter, 9.0)
        self.assertEqual(second.representative_enter.leaf_id, "inner")
        self.assertTrue(second.representative_enter.flipped)
        self.assertEqual(second.representative_enter.normal, (-1.0, 0.0, 0.0))

    def test_nearest_forward_boundary(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        result = ray_intervals(tree, Ray((-7.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertIsInstance(result.nearest_boundary, Boundary)
        self.assertEqual(result.nearest_boundary.t, 2.0)
        self.assertEqual(result.nearest_boundary.kind, "enter")


class TestCoincidentBoxFaces(unittest.TestCase):
    def test_union_shared_face_merges_no_spurious_boundary(self):
        tree = Union(BOX_A, BOX_B)
        result = ray_intervals(tree, Ray((-1.0, 0.5, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit) for iv in result.intervals],
                         [(1.0, 5.0)])
        # No boundary at the shared face t = 2.
        bs = {iv.t_enter for iv in result.intervals}
        bs |= {iv.t_exit for iv in result.intervals}
        self.assertNotIn(2.0, bs)
        self.assertEqual(result.nearest_boundary.t, 1.0)

    def test_difference_shared_face_no_phantom_boundary(self):
        # A \\ B with B flush against A's far face leaves [0, 2] on x.
        tree = Difference(BOX_A, BOX_B)
        result = ray_intervals(tree, Ray((-1.0, 0.5, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit) for iv in result.intervals],
                         [(1.0, 3.0)])
        self.assertEqual(result.touches, ())

    def test_intersection_touching_boxes_is_touch_with_both_faces(self):
        # Closed boxes meet on the plane x = 2: the ray meets it at one point.
        tree = Intersection(BOX_A, BOX_B)
        result = ray_intervals(tree, Ray((-1.0, 0.5, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        touch = result.touches[0]
        self.assertIsInstance(touch, Touch)
        self.assertEqual(touch.t, 3.0)
        ids = [s.leaf_id for s in touch.sources]
        self.assertEqual(sorted(ids), ["a", "b"])
        faces = {s.leaf_id: s.face for s in touch.sources}
        self.assertEqual(faces["a"], "x+")
        self.assertEqual(faces["b"], "x-")

    def test_identical_boxes_difference_is_empty(self):
        dup = Box("a2", (0.0, 0.0, -1.0), (2.0, 1.0, 1.0))
        tree = Difference(BOX_A, dup)
        result = ray_intervals(tree, Ray((-3.0, 0.5, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(result.touches, ())
        self.assertIsNone(result.nearest_boundary)

    def test_coincident_face_source_representative_lexicographic(self):
        late = Box("z-id", (0.0, 0.0, -1.0), (2.0, 1.0, 1.0))
        early = Box("a-id", (0.0, 0.0, -1.0), (2.0, 1.0, 1.0))
        tree = Union(late, early)
        result = ray_intervals(tree, Ray((-3.0, 0.5, 0.0),
                                         (1.0, 0.0, 0.0)))
        iv = result.intervals[0]
        self.assertEqual([s.leaf_id for s in iv.enter_sources],
                         ["a-id", "z-id"])
        self.assertEqual(iv.representative_enter.leaf_id, "a-id")
        self.assertEqual([s.leaf_id for s in iv.exit_sources],
                         ["a-id", "z-id"])


class TestTangency(unittest.TestCase):
    def test_sphere_tangent_is_touch_not_interval(self):
        sphere = Sphere("s", (0.0, 1.0, 0.0), 1.0)
        result = ray_intervals(sphere, Ray((-3.0, 0.0, 0.0),
                                           (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        self.assertEqual(result.touches[0].t, 3.0)
        self.assertEqual(result.touches[0].representative.leaf_id, "s")
        # Tangency is not solid interior and not a positive solid boundary.
        self.assertFalse(solid_at(result, 3.0))
        self.assertIsNone(result.nearest_boundary)

    def test_tangent_at_origin_has_no_forward_boundary(self):
        sphere = Sphere("s", (0.0, 1.0, 0.0), 1.0)
        result = ray_intervals(sphere, Ray((0.0, 0.0, 0.0),
                                           (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(result.touches[0].t, 0.0)
        # "Nearest positive" requires t > 0 strictly.
        self.assertIsNone(result.nearest_boundary)

    def test_near_miss_exact_rational_decision(self):
        # Sphere centered at y=1, radius 1: tangent line is y=2.  Offsets of
        # 1e-13 put the float discriminant inside the promotion band, but the
        # exact Fraction discriminant decides miss vs tiny interval.
        sphere = Sphere("s", (0.0, 1.0, 0.0), 1.0)
        ray_miss = Ray((-3.0, 2.0 + 1e-13, 0.0), (1.0, 0.0, 0.0))
        result = ray_intervals(sphere, ray_miss)
        self.assertEqual(result.intervals, ())
        self.assertEqual(result.touches, ())

        ray_graze = Ray((-3.0, 2.0 - 1e-13, 0.0), (1.0, 0.0, 0.0))
        result2 = ray_intervals(sphere, ray_graze)
        self.assertEqual(len(result2.intervals), 1)
        iv = result2.intervals[0]
        self.assertLess(iv.t_enter, 3.0)
        self.assertGreater(iv.t_exit, 3.0)
        self.assertLess(iv.t_exit - iv.t_enter, 1e-5)

    def test_box_edge_touch_two_faces(self):
        box = Box("b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        # Same-sign x/y components graze the edge (x=0, y=1) at one point:
        # before and after t=1 the line is outside the box.
        result = ray_intervals(box,
                               Ray((-1.0, 0.0, 0.5), (1.0, 1.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        self.assertEqual(result.touches[0].t, 1.0)
        self.assertEqual(sorted(s.face for s in result.touches[0].sources),
                         ["x-", "y+"])

    def test_box_corner_touch_three_faces(self):
        box = Box("b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        # Grazes the corner (0,1,1): x-, y+ and z+ bind simultaneously.
        result = ray_intervals(box,
                               Ray((-1.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        self.assertEqual(result.touches[0].t, 1.0)
        self.assertEqual(sorted(s.face for s in result.touches[0].sources),
                         ["x-", "y+", "z+"])

    def test_tangent_buried_inside_other_solid_is_suppressed(self):
        # The sphere grazes at t=3; that point is strictly inside the box,
        # so the union is solid on both sides and the touch must not appear.
        sphere = Sphere("s", (0.0, 1.0, 0.0), 1.0)
        box = Box("b", (-1.0, -1.0, -1.0), (4.0, 2.0, 1.0))
        result = ray_intervals(Union(sphere, box),
                               Ray((-3.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit)
                          for iv in result.intervals], [(2.0, 7.0)])
        self.assertEqual(result.touches, ())

        # Same geometry under Difference: the tangent point is removed with
        # the enclosing box volume, leaving neither interval nor touch.
        result2 = ray_intervals(Difference(sphere, box),
                                Ray((-3.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual(result2.intervals, ())
        self.assertEqual(result2.touches, ())

    def test_sphere_tangent_on_box_face_intersection_touch_two_sources(self):
        sphere = Sphere("s", (0.0, 1.0, 0.0), 1.0)
        box = Box("b", (0.0, -1.0, -1.0), (3.0, 3.0, 1.0))
        result = ray_intervals(Intersection(sphere, box),
                               Ray((-3.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(len(result.touches), 1)
        touch = result.touches[0]
        self.assertEqual(touch.t, 3.0)
        self.assertEqual(sorted(s.leaf_id for s in touch.sources),
                         ["b", "s"])


class TestRayStartingInside(unittest.TestCase):
    def test_start_inside_sphere(self):
        result = ray_intervals(SPHERE_OUTER,
                               Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit) for iv in result.intervals],
                         [(-5.0, 5.0)])
        # Forward boundary is the exit; the entry lies behind the origin.
        self.assertEqual(result.nearest_boundary.kind, "exit")
        self.assertEqual(result.nearest_boundary.t, 5.0)

    def test_start_on_surface_pointing_outward(self):
        result = ray_intervals(SPHERE_OUTER,
                               Ray((5.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit) for iv in result.intervals],
                         [(-10.0, 0.0)])
        self.assertIsNone(result.nearest_boundary)

    def test_start_inside_cavity(self):
        tree = Difference(SPHERE_OUTER, SPHERE_INNER)
        result = ray_intervals(tree, Ray((0.0, 0.0, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertFalse(solid_at(result, 0.0))
        self.assertEqual(result.nearest_boundary.t, 2.0)
        self.assertEqual(result.nearest_boundary.kind, "enter")


class TestIntervalShapeInvariants(unittest.TestCase):
    def test_sorted_disjoint_nondegenerate(self):
        cavity = Union(SPHERE_INNER,
                       Box("slit", (-1.0, 2.0, -0.5), (1.0, 6.0, 0.5)))
        tree = Difference(SPHERE_OUTER, cavity)
        result = ray_intervals(tree, Ray((-7.0, 0.2, 0.1),
                                         (1.0, 0.0, 0.0)))
        ivs = result.intervals
        for iv in ivs:
            self.assertLess(iv.t_enter, iv.t_exit)
            self.assertTrue(iv.enter_sources)
            self.assertTrue(iv.exit_sources)
        for a, b in zip(ivs, ivs[1:]):
            self.assertLess(a.t_exit, b.t_enter)

    def test_miss_is_empty_result_not_error(self):
        result = ray_intervals(SPHERE_OUTER,
                               Ray((100.0, 100.0, 100.0),
                                   (1.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, ())
        self.assertEqual(result.touches, ())
        self.assertIsNone(result.nearest_boundary)


class TestInputRejection(unittest.TestCase):
    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidGeometryError):
                ray_intervals(Sphere("s", (0.0, 0.0, 0.0), bad),
                              Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
            with self.assertRaises(InvalidGeometryError):
                ray_intervals(Sphere("s", (bad, 0.0, 0.0), 1.0),
                              Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
            with self.assertRaises(InvalidRayError):
                ray_intervals(SPHERE_OUTER,
                              Ray((bad, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_bool_rejected_as_number(self):
        # bool reaches validation through the ray's numeric components.
        with self.assertRaises(TypeError):
            ray_intervals(Sphere("s", (0.0, 0.0, 0.0), 1),
                          Ray((0.0, 0.0, 0.0), (True, 0.0, 0.0)))
        with self.assertRaises(TypeError):
            ray_intervals(Sphere("s", (False, 0.0, 0.0), 1.0),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_zero_direction(self):
        with self.assertRaises(InvalidRayError):
            ray_intervals(SPHERE_OUTER,
                          Ray((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def test_bad_radius_box_and_types(self):
        with self.assertRaises(InvalidGeometryError):
            ray_intervals(Sphere("s", (0.0, 0.0, 0.0), 0.0),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        with self.assertRaises(InvalidGeometryError):
            ray_intervals(Box("b", (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        with self.assertRaises(InvalidGeometryError):
            ray_intervals("not a tree",
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        with self.assertRaises(InvalidRayError):
            ray_intervals(SPHERE_OUTER, (0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            ray_intervals(Sphere(1, (0.0, 0.0, 0.0), 1.0),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_duplicate_leaf_id(self):
        with self.assertRaises(InvalidGeometryError):
            ray_intervals(Union(Sphere("s", (0.0, 0.0, 0.0), 1.0),
                                Sphere("s", (2.0, 0.0, 0.0), 1.0)),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_shared_node_is_dag_error(self):
        leaf = Sphere("s", (0.0, 0.0, 0.0), 1.0)
        with self.assertRaises(InvalidGeometryError):
            ray_intervals(Union(leaf, leaf),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_error_messages_are_located(self):
        try:
            ray_intervals(Box("b", (0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0)),
                          Ray((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        except InvalidRayError:
            pass  # zero direction check may come first depending on order
        try:
            ray_intervals(Sphere("s", (0.0, 0.0, 0.0), -3.0),
                          Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        except InvalidGeometryError as exc:
            self.assertIn("radius", str(exc))
            self.assertIn("s", str(exc))
        else:  # pragma: no cover
            self.fail("expected InvalidGeometryError")


class TestBudget(unittest.TestCase):
    def test_over_100_nodes_is_budget_error_not_miss(self):
        # A full binary tree has 2*L - 1 nodes. 51 leaves + 50 ops = 101.
        node = Sphere("s0", (0.0, 0.0, 0.0), 1.0)
        for k in range(1, 51):
            node = Union(node,
                         Sphere(f"s{k}", (float(k), 0.0, 0.0), 1.0))
        with self.assertRaises(BudgetExceededError):
            ray_intervals(node, Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    def test_max_sized_tree_99_nodes_accepted(self):
        # 50 leaves + 49 operators = 99 (largest full binary tree <= 100).
        node = Sphere("s0", (0.0, 0.0, 0.0), 1.0)
        for k in range(1, 50):
            node = Union(node,
                         Sphere(f"s{k}", (float(k), 0.0, 0.0), 1.0))
        result = ray_intervals(node, Ray((0.0, 0.0, 0.0),
                                         (1.0, 0.0, 0.0)))
        self.assertIsNotNone(result)


class TestFaceSources(unittest.TestCase):
    def test_box_face_names_and_normals(self):
        box = Box("b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        result = ray_intervals(box, Ray((-2.0, 0.5, 0.5),
                                        (1.0, 0.0, 0.0)))
        iv = result.intervals[0]
        self.assertEqual(iv.t_enter, 2.0)
        self.assertEqual(iv.t_exit, 3.0)
        self.assertEqual(iv.representative_enter.face, "x-")
        self.assertEqual(iv.representative_enter.normal, (-1.0, 0.0, 0.0))
        self.assertEqual(iv.representative_exit.face, "x+")
        self.assertEqual(iv.representative_exit.normal, (1.0, 0.0, 0.0))

    def test_difference_box_face_normal_flip(self):
        outer = Box("outer", (0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
        inner = Box("inner", (1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
        result = ray_intervals(Difference(outer, inner),
                               Ray((-1.0, 2.0, 2.0), (1.0, 0.0, 0.0)))
        self.assertEqual([(iv.t_enter, iv.t_exit) for iv in result.intervals],
                         [(1.0, 2.0), (4.0, 5.0)])
        enter_cavity = result.intervals[0].representative_exit
        self.assertEqual(enter_cavity.leaf_id, "inner")
        self.assertEqual(enter_cavity.face, "x-")
        self.assertTrue(enter_cavity.flipped)
        # Inner x- face normally points -x; flipped effective normal is +x.
        self.assertEqual(enter_cavity.normal, (1.0, 0.0, 0.0))

    def test_nested_difference_flip_parity(self):
        # Difference within a Difference right subtree cancels the flip.
        a = Box("a", (0.0, 0.0, 0.0), (6.0, 6.0, 6.0))
        b = Box("b", (1.0, 1.0, 1.0), (5.0, 5.0, 5.0))
        c = Box("c", (2.0, 2.0, 2.0), (4.0, 4.0, 4.0))
        # a \\ (b \\ c) = (a \\ b) union c on this line.
        tree = Difference(a, Difference(b, c))
        result = ray_intervals(tree, Ray((-1.0, 3.0, 3.0),
                                         (1.0, 0.0, 0.0)))
        ends = [(iv.t_enter, iv.t_exit) for iv in result.intervals]
        self.assertEqual(ends, [(1.0, 2.0), (3.0, 5.0), (6.0, 7.0)])
        srcs = {s.leaf_id: s for iv in result.intervals
                for s in iv.enter_sources + iv.exit_sources}
        self.assertFalse(srcs["c"].flipped)
        self.assertTrue(srcs["b"].flipped)


if __name__ == "__main__":
    unittest.main()
