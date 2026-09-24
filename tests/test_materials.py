"""Material id, face orientation, nested differences and tangent/coincident
boundary tests.

Every expected value below is derived by hand on small-integer geometry, not
by calling the implementation under test.  Scenes:

* concentric / axis-aligned primitives whose event t values are integers,
* nested differences ``A - (B - C)`` and ``A - (B - (C - D))`` whose solid
  runs follow directly from nested set algebra,
* externally / internally tangent solids sharing exactly one endpoint.
"""

import unittest

from csg_rays import (
    ORIENT_BACK,
    ORIENT_FRONT,
    ORIENT_GRAZING,
    Box,
    Difference,
    GeometryError,
    Intersection,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)


def _boundary_map(result):
    return {b.t: b for b in result.boundaries}


class TestMaterialId(unittest.TestCase):
    def test_material_carried_onto_every_source(self):
        hollow = Difference(
            Sphere((0, 0, 0), 5, "outer", "steel"),
            Sphere((0, 0, 0), 2, "inner", "air"),
        )
        res = ray_intervals(hollow, Ray((-10, 0, 0), (1, 0, 0)))
        materials = {b.t: b.source.material for b in res.boundaries}
        self.assertEqual(materials, {5.0: "steel", 8.0: "air", 12.0: "air", 15.0: "steel"})
        # every source on every boundary carries its own leaf material
        for b in res.boundaries:
            for s in b.sources:
                self.assertEqual(s.material, "steel" if s.leaf_id == "outer" else "air")

    def test_material_defaults_to_none(self):
        res = ray_intervals(
            Sphere((0, 0, 0), 2, "s"), Ray((-5, 0, 0), (1, 0, 0))
        )
        for b in res.boundaries:
            self.assertIsNone(b.source.material)

    def test_box_material_carried(self):
        res = ray_intervals(
            Box((0, 0, 0), (2, 2, 2), "b", "alu"), Ray((-1, 1, 1), (1, 0, 0))
        )
        self.assertEqual([b.source.material for b in res.boundaries], ["alu", "alu"])

    def test_invalid_material_rejected(self):
        with self.assertRaises(GeometryError) as ctx:
            Sphere((0, 0, 0), 1, "s", "")
        self.assertIn("material", str(ctx.exception))
        with self.assertRaises(GeometryError) as ctx:
            Box((0, 0, 0), (1, 1, 1), "b", 7)
        self.assertIn("material", str(ctx.exception))
        for bad in (True, b"x", 3.5):
            with self.assertRaises(GeometryError):
                Sphere((0, 0, 0), 1, "s", bad)

    def test_invalid_material_leaves_no_partial_construction(self):
        # A rejected leaf is never constructed; a subsequent valid leaf is
        # unaffected (ray_intervals is a pure function with no partial state).
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, "s", 7)  # int is not a material id
        good = Sphere((0, 0, 0), 1, "s", "ok")
        res = ray_intervals(good, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual(len(res.intervals), 1)
        self.assertEqual(res.intervals[0].start.source.material, "ok")


class TestOrientation(unittest.TestCase):
    def test_sphere_front_and_back(self):
        # Ray (-10,0,0)+t(1,0,0) through sphere r=5: entry face normal -x
        # (against the ray) -> front, exit face normal +x (along the ray)
        # -> back.
        res = ray_intervals(
            Sphere((0, 0, 0), 5, "s", "m"), Ray((-10, 0, 0), (1, 0, 0))
        )
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.orientation, ORIENT_FRONT)
        self.assertEqual(exit_.source.orientation, ORIENT_BACK)

    def test_orientation_independent_of_flipped(self):
        # Hollow sphere: inner (subtractor) faces have flipped normals but the
        # orientation is still decided from the leaf's own outward normal.
        hollow = Difference(
            Sphere((0, 0, 0), 5, "outer"), Sphere((0, 0, 0), 2, "inner")
        )
        res = ray_intervals(hollow, Ray((-10, 0, 0), (1, 0, 0)))
        b = _boundary_map(res)
        # t=8: the ray enters the inner ball at x=-2, whose geometric outward
        # normal is -x (against the +x ray) -> front; the reported normal is
        # flipped to +x.  t=12: geometric outward normal +x (along the ray)
        # -> back, reported flipped to -x.
        self.assertTrue(b[8.0].source.flipped)
        self.assertEqual(b[8.0].source.orientation, ORIENT_FRONT)
        self.assertEqual(b[8.0].source.normal, (1.0, 0.0, 0.0))
        self.assertTrue(b[12.0].source.flipped)
        self.assertEqual(b[12.0].source.orientation, ORIENT_BACK)
        self.assertEqual(b[12.0].source.normal, (-1.0, 0.0, 0.0))
        # outer faces are not flipped
        self.assertFalse(b[5.0].source.flipped)
        self.assertFalse(b[15.0].source.flipped)

    def test_box_corner_faces_oriented(self):
        # Ray (-1,-1,-1)+t(1,1,1) through unit-ish box (0,0,0)..(2,2,2):
        # every low-face normal points against (+1,+1,+1) -> front;
        # every high-face normal points along it -> back.
        res = ray_intervals(
            Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, -1, -1), (1, 1, 1))
        )
        enter, exit_ = res.boundaries
        self.assertTrue(all(s.orientation == ORIENT_FRONT for s in enter.sources))
        self.assertTrue(all(s.orientation == ORIENT_BACK for s in exit_.sources))

    def test_tangent_sphere_is_grazing(self):
        res = ray_intervals(
            Sphere((0, 0, 0), 2, "s"), Ray((-5, 2, 0), (1, 0, 0))
        )
        (touch,) = res.touches
        self.assertEqual(touch.source.orientation, ORIENT_GRAZING)
        self.assertEqual(touch.source.kind, "touch")
        self.assertIsNone(touch.source.material)

    def test_irrational_sphere_roots_oriented_by_root_order(self):
        # Sphere r=5, ray (-6,0,0)+t(1,1,0): (t-6)^2+t^2=25
        # -> 2t^2-12t+11=0, discriminant 56 (irrational), roots
        # t = 3 +/- sqrt(14)/2.  Smaller root = entry (front), larger = exit
        # (back); check the root-sum/product identities by hand instead of
        # trusting an implementation-computed expected t.
        res = ray_intervals(
            Sphere((0, 0, 0), 5, "s", "glass"), Ray((-6, 0, 0), (1, 1, 0))
        )
        self.assertEqual(len(res.intervals), 1)
        iv = res.intervals[0]
        # Vieta: t0+t1 = 6, t0*t1 = 11/2
        self.assertAlmostEqual(iv.t0 + iv.t1, 6.0, places=12)
        self.assertAlmostEqual(iv.t0 * iv.t1, 5.5, places=12)
        enter, exit_ = res.boundaries
        self.assertEqual(enter.kind, "enter")
        self.assertEqual(enter.source.orientation, ORIENT_FRONT)
        self.assertEqual(exit_.source.orientation, ORIENT_BACK)
        self.assertEqual(enter.source.material, "glass")
        # normals are genuine unit vectors (Decimal path for irrational t)
        for b in res.boundaries:
            n = b.source.normal
            self.assertAlmostEqual(n[0] ** 2 + n[1] ** 2 + n[2] ** 2, 1.0, places=12)

    def test_touch_carries_material(self):
        res = ray_intervals(
            Sphere((0, 0, 0), 2, "s", "coat"), Ray((-5, 2, 0), (1, 0, 0))
        )
        (touch,) = res.touches
        self.assertTrue(all(s.material == "coat" for s in touch.sources))

    def test_face_riding_rays_grazing_face(self):
        # Ray rides on the box face y=0; the ridden face is grazing while the
        # x end-caps remain front/back.
        res = ray_intervals(
            Box((0, 0, 0), (2, 2, 2), "b"), Ray((-1, 0, 1), (1, 0, 0))
        )
        (touch,) = res.touches
        by_face = {s.face: s.orientation for s in touch.sources}
        self.assertEqual(by_face["y_lo"], ORIENT_GRAZING)
        self.assertEqual(by_face["x_lo"], ORIENT_FRONT)
        self.assertEqual(by_face["x_hi"], ORIENT_BACK)

    def test_enter_always_front_exit_always_back(self):
        # For a transversal leaf crossing, the geometric outward normal at an
        # entry points against the ray (front) and at an exit along it (back),
        # whichever direction the ray travels.  Holds on both orientations.
        sphere = Sphere((0, 0, 0), 5, "s")
        rays = (
            ((-10, 0, 0), (1, 0, 0)),
            ((10, 0, 0), (-1, 0, 0)),
        )
        for origin, direction in rays:
            res = ray_intervals(sphere, Ray(origin, direction))
            for boundary in res.boundaries:
                expected = ORIENT_FRONT if boundary.kind == "enter" else ORIENT_BACK
                self.assertEqual(boundary.source.orientation, expected)


class TestNestedDifferences(unittest.TestCase):
    def test_A_minus_B_minus_C_solid_runs_and_parity(self):
        # Balls centered at the origin: A r=10, B r=6, C r=2.
        # A - (B - C) = (A - B) union (A intersect B intersect C), so along
        # the x-axis from -20: solid (10,14), (18,22), (26,30); empty gaps
        # (14,18) and (22,26).
        tree = Difference(
            Sphere((0, 0, 0), 10, "A", "mA"),
            Difference(Sphere((0, 0, 0), 6, "B", "mB"), Sphere((0, 0, 0), 2, "C", "mC")),
        )
        res = ray_intervals(tree, Ray((-20, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(10.0, 14.0), (18.0, 22.0), (26.0, 30.0)],
        )
        b = _boundary_map(res)
        # B is one Difference level deep -> flipped; C is two deep -> even,
        # not flipped.
        self.assertFalse(b[10.0].source.flipped)
        self.assertEqual(b[10.0].source.leaf_id, "A")
        self.assertTrue(b[14.0].source.flipped)
        self.assertEqual(b[14.0].source.leaf_id, "B")
        self.assertFalse(b[18.0].source.flipped)
        self.assertEqual(b[18.0].source.leaf_id, "C")
        self.assertFalse(b[22.0].source.flipped)
        self.assertEqual(b[22.0].source.leaf_id, "C")
        self.assertTrue(b[26.0].source.flipped)
        self.assertEqual(b[26.0].source.leaf_id, "B")
        self.assertFalse(b[30.0].source.flipped)
        self.assertEqual(b[30.0].source.leaf_id, "A")
        # boundary kind sequence on a left-to-right sweep
        self.assertEqual(
            [x.kind for x in res.boundaries],
            ["enter", "exit", "enter", "exit", "enter", "exit"],
        )
        # materials survive nesting
        self.assertEqual(b[18.0].source.material, "mC")
        self.assertEqual(b[14.0].source.material, "mB")

    def test_depth_three_parity(self):
        # A r=20, B r=15, C r=10, D r=5 all concentric.
        # A - (B - (C - D)): subtractor depths B=1 (flip), C=2 (no flip),
        # D=3 (flip).  Solid shell run boundaries alternate with these leaves.
        tree = Difference(
            Sphere((0, 0, 0), 20, "A"),
            Difference(
                Sphere((0, 0, 0), 15, "B"),
                Difference(Sphere((0, 0, 0), 10, "C"), Sphere((0, 0, 0), 5, "D")),
            ),
        )
        res = ray_intervals(tree, Ray((-40, 0, 0), (1, 0, 0)))
        # crossing t's: A 20/60, B 25/55, C 30/50, D 35/45
        # result alternates solid/empty at every boundary (generic nesting)
        expect = {
            20.0: ("A", False), 25.0: ("B", True), 30.0: ("C", False),
            35.0: ("D", True), 45.0: ("D", True), 50.0: ("C", False),
            55.0: ("B", True), 60.0: ("A", False),
        }
        b = _boundary_map(res)
        self.assertEqual(set(b), set(expect))
        for t, (leaf, flipped) in expect.items():
            self.assertEqual(b[t].source.leaf_id, leaf, t)
            self.assertEqual(b[t].source.flipped, flipped, t)

    def test_chained_subtractors_each_flipped(self):
        # A - B - C with B, C disjoint siblings along x; origin (-1,1,1),
        # ray (1,0,0), so t = x + 1.
        # A: x in (0,30) -> t (1,31); B: x in (8,10) -> t (9,11);
        # C: x in (20,22) -> t (21,23).  Both subtractors are one Difference
        # level deep, so both are flipped.
        tree = Difference(
            Box((0, 0, 0), (30, 2, 2), "A"),
            Box((8, -1, -1), (10, 3, 3), "B"),
            Box((20, -1, -1), (22, 3, 3), "C"),
        )
        res = ray_intervals(tree, Ray((-1, 1, 1), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(1.0, 9.0), (11.0, 21.0), (23.0, 31.0)],
        )
        flipped_leaves = {
            s.leaf_id for b in res.boundaries for s in b.sources if s.flipped
        }
        self.assertEqual(flipped_leaves, {"B", "C"})


class TestTangentAndCoincidentBoundaries(unittest.TestCase):
    def test_externally_tangent_spheres_union_is_continuous(self):
        # A r=2 centered 0, B r=2 centered 4: tangent only at x=2 (t=7 from
        # origin -5).  Both closed balls cover x=2, so along the x-axis the
        # union is one continuous solid run x in (-2, 6), i.e. t (3, 11);
        # the coincident A-exit / B-enter at t=7 create no phantom boundary.
        tree = Union([Sphere((0, 0, 0), 2, "a"), Sphere((4, 0, 0), 2, "b")])
        res = ray_intervals(tree, Ray((-5, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(3.0, 11.0)])
        self.assertEqual([b.t for b in res.boundaries], [3.0, 11.0])
        self.assertEqual(res.touches, ())

    def test_internally_tangent_subtractor_no_phantom_at_tangent(self):
        # A r=4 centered 0 (solid x in (-4,4)); B r=2 centered 2 covers
        # x in (0,4) and is internally tangent to A at x=4.  A - B along the
        # x-axis from -10 is solid (6,10) only; t=14 is A-exit coincident
        # with B-exit into already-empty space, producing no boundary there.
        tree = Difference(
            Sphere((0, 0, 0), 4, "A"), Sphere((2, 0, 0), 2, "B")
        )
        res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(6.0, 10.0)])
        self.assertEqual([b.t for b in res.boundaries], [6.0, 10.0])
        b = _boundary_map(res)
        self.assertEqual(b[6.0].source.leaf_id, "A")       # enter A
        self.assertEqual(b[10.0].source.leaf_id, "B")      # enter subtractor B
        self.assertTrue(b[10.0].source.flipped)

    def test_external_tangent_touch_reported_when_empty_on_both_sides(self):
        # Two disjoint balls tangent only at x=2.  On the line y=2 the ray
        # (-5,2,0)+t(1,0,0) grazes ball a at its top (t=5, point (0,2,0)) and
        # ball b at its top (t=9, point (4,2,0)): two distinct isolated
        # touches, each empty on both sides, not solid interior.
        tree = Union([Sphere((0, 0, 0), 2, "a"), Sphere((4, 0, 0), 2, "b")])
        res = ray_intervals(tree, Ray((-5, 2, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        self.assertEqual(len(res.touches), 2)
        self.assertEqual([(t.t0, t.t1) for t in res.touches], [(5.0, 5.0), (9.0, 9.0)])
        leaf_by_t = {t.t0: t.source.leaf_id for t in res.touches}
        self.assertEqual(leaf_by_t, {5.0: "a", 9.0: "b"})
        for touch in res.touches:
            self.assertTrue(
                all(s.orientation == ORIENT_GRAZING for s in touch.sources)
            )

    def test_coincident_shared_face_fixed_representative_priority(self):
        # Boxes a (0..2) and b (2..4) share the plane x=2; Difference(a,b)
        # ends at t=3 where A-exit and flipped B-enter coincide.  Both faces
        # are valid sources; the representative is fixed by leaf-id order.
        a = Box((0, 0, 0), (2, 2, 2), "a")
        b = Box((2, 0, 0), (4, 2, 2), "b")
        res = ray_intervals(Difference(a, b), Ray((-1, 1, 1), (1, 0, 0)))
        (exit_boundary,) = [x for x in res.boundaries if x.t == 3.0]
        self.assertEqual(
            [s.leaf_id for s in exit_boundary.sources], ["a", "b"]
        )
        self.assertIs(exit_boundary.source, exit_boundary.sources[0])
        self.assertEqual(exit_boundary.source.leaf_id, "a")

    def test_three_way_coincident_endpoint_representative(self):
        # Three identical boxes with leaf ids zzz/mmm/aaa share the exact same
        # low x face at t=1 (ray origin (-1,1,1)).  All three are valid
        # sources; the representative is deterministically the lexicographic
        # minimum ("aaa"), independent of construction order.
        ray = Ray((-1, 1, 1), (1, 0, 0))
        boxes = [
            Box((0, 0, 0), (2, 2, 2), "zzz"),
            Box((0, 0, 0), (2, 2, 2), "mmm"),
            Box((0, 0, 0), (2, 2, 2), "aaa"),
        ]
        res = ray_intervals(Union(boxes), ray)
        enter = res.boundaries[0]
        self.assertEqual(enter.t, 1.0)
        self.assertEqual(
            [s.leaf_id for s in enter.sources], ["aaa", "mmm", "zzz"]
        )
        self.assertEqual(enter.source.leaf_id, "aaa")


if __name__ == "__main__":
    unittest.main()
