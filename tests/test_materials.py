"""Tests for the material-id / surface-orientation iteration.

All expected values below are derived by hand from small-integer geometry
(the event t values are exact integers); the implementation under test is
never used to produce them.
"""

import random
import unittest
from types import MappingProxyType

from csg_rays import (
    BOX_FACES,
    DEFAULT_MATERIAL,
    Box,
    Difference,
    GeometryError,
    Intersection,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)


class TestPrimitiveMaterials(unittest.TestCase):
    def test_default_material_on_sphere_and_box(self):
        s = Sphere((0, 0, 0), 2, "s")
        b = Box((0, 0, 0), (2, 2, 2), "b")
        self.assertEqual(s.material, DEFAULT_MATERIAL)
        self.assertEqual(s.materials["sphere"], DEFAULT_MATERIAL)
        self.assertEqual(set(b.materials), set(BOX_FACES))
        self.assertTrue(all(m == DEFAULT_MATERIAL for m in b.materials.values()))

        res = ray_intervals(s, Ray((-5, 0, 0), (1, 0, 0)))
        for boundary in res.boundaries:
            self.assertEqual(boundary.source.material, DEFAULT_MATERIAL)
            self.assertEqual(boundary.source.orientation, "outward")
            self.assertFalse(boundary.source.flipped)

    def test_sphere_material_propagates_to_every_source(self):
        res = ray_intervals(
            Sphere((0, 0, 0), 2, "s", material="steel"),
            Ray((-5, 0, 0), (1, 0, 0)),
        )
        enter, exit_ = res.boundaries
        self.assertEqual(
            [s.material for s in enter.sources + exit_.sources],
            ["steel", "steel"],
        )
        self.assertEqual(enter.source.orientation, "outward")

    def test_box_per_face_materials(self):
        box = Box(
            (0, 0, 0), (2, 2, 2), "b",
            material="plastic", face_materials={"x_lo": "red"},
        )
        # the resolved map keeps both the override and the primitive default
        self.assertEqual(box.materials["x_lo"], "red")
        self.assertEqual(box.materials["x_hi"], "plastic")
        self.assertEqual(box.materials["z_lo"], "plastic")

        res = ray_intervals(box, Ray((-1, 1, 1), (1, 0, 0)))
        enter, exit_ = res.boundaries
        self.assertEqual(enter.source.face, "x_lo")
        self.assertEqual(enter.source.material, "red")
        self.assertEqual(exit_.source.face, "x_hi")
        self.assertEqual(exit_.source.material, "plastic")

    def test_corner_hit_carries_each_face_material(self):
        # face priority, not material-name priority: x_lo keeps "zzz" and is
        # still the representative because faces sort x_lo < y_lo < z_lo.
        box = Box(
            (0, 0, 0), (2, 2, 2), "b",
            material="base",
            face_materials={"x_lo": "zzz", "y_lo": "mmm", "z_lo": "aaa"},
        )
        res = ray_intervals(box, Ray((-1, -1, -1), (1, 1, 1)))
        enter, exit_ = res.boundaries
        self.assertEqual(
            [(s.face, s.material) for s in enter.sources],
            [("x_lo", "zzz"), ("y_lo", "mmm"), ("z_lo", "aaa")],
        )
        self.assertEqual(enter.source.face, "x_lo")
        self.assertEqual(enter.source.material, "zzz")
        # the exit corner keeps the primitive default on every face
        self.assertEqual([s.material for s in exit_.sources], ["base"] * 3)


class TestMaterialPropagationThroughBooleans(unittest.TestCase):
    def test_union_outer_faces_keep_their_materials(self):
        a = Box((0, 0, 0), (2, 2, 2), "aa", material="m_aa")
        b = Box((2, 0, 0), (4, 2, 2), "bb", material="m_bb")
        res = ray_intervals(Union([a, b]), Ray((-1, 1, 1), (1, 0, 0)))
        # one merged interval (1,5); the shared face at t=3 creates no boundary
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(1.0, 5.0)])
        left, right = res.boundaries
        self.assertEqual((left.source.leaf_id, left.source.material), ("aa", "m_aa"))
        self.assertEqual((right.source.leaf_id, right.source.material), ("bb", "m_bb"))

    def test_intersection_endpoints_keep_operand_materials(self):
        # A centered at -1 (r=3): solid t in (-4, 2); B centered at +1: (-2, 4)
        a = Sphere((-1, 0, 0), 3, "a", material="ma")
        b = Sphere((1, 0, 0), 3, "b", material="mb")
        res = ray_intervals(Intersection([a, b]), Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-2.0, 2.0)])
        enter, exit_ = res.boundaries
        # entering the intersection at t=-2 crosses B's surface; exiting at
        # t=2 crosses A's surface
        self.assertEqual((enter.source.leaf_id, enter.source.material), ("b", "mb"))
        self.assertEqual((exit_.source.leaf_id, exit_.source.material), ("a", "ma"))

    def test_coincident_endpoint_representative_priority_is_leaf_then_face(self):
        # material names deliberately oppose the leaf-id order: the priority
        # is (leaf_id, face), never the material id.
        a = Box((0, 0, 0), (2, 2, 2), "aa", material="zzz-mat")
        b = Box((2, 0, 0), (4, 2, 2), "bb", material="aaa-mat")
        res = ray_intervals(Difference(a, b), Ray((-1, 1, 1), (1, 0, 0)))
        # solid (1,3); at t=3 A's x_hi exits and B's x_lo enters simultaneously
        shared = res.boundaries[-1]
        self.assertEqual(shared.t, 3.0)
        self.assertEqual(
            [(s.leaf_id, s.face, s.material) for s in shared.sources],
            [("aa", "x_hi", "zzz-mat"), ("bb", "x_lo", "aaa-mat")],
        )
        self.assertEqual(shared.source.leaf_id, "aa")
        self.assertEqual(shared.source.material, "zzz-mat")
        self.assertFalse(shared.sources[0].flipped)
        self.assertTrue(shared.sources[1].flipped)
        self.assertEqual(shared.sources[1].orientation, "inward")


class TestDifferenceOrientation(unittest.TestCase):
    def test_hollow_sphere_materials_and_facing(self):
        hollow = Difference(
            Sphere((0, 0, 0), 5, "outer", material="shell"),
            Sphere((0, 0, 0), 2, "inner", material="liner"),
        )
        res = ray_intervals(hollow, Ray((-10, 0, 0), (1, 0, 0)))
        expected = [
            # t, kind, leaf, material, orientation, flipped, normal
            (5.0, "enter", "outer", "shell", "outward", False, (-1.0, 0.0, 0.0)),
            (8.0, "exit", "inner", "liner", "inward", True, (1.0, 0.0, 0.0)),
            (12.0, "enter", "inner", "liner", "inward", True, (-1.0, 0.0, 0.0)),
            (15.0, "exit", "outer", "shell", "outward", False, (1.0, 0.0, 0.0)),
        ]
        for boundary, (t, kind, leaf, mat, facing, flipped, normal) in zip(
            res.boundaries, expected
        ):
            self.assertEqual((boundary.t, boundary.kind), (t, kind))
            s = boundary.source
            self.assertEqual((s.leaf_id, s.material), (leaf, mat))
            self.assertEqual(s.orientation, facing)
            self.assertEqual(s.flipped, flipped)
            self.assertEqual(s.normal, normal)
        iv0, iv1 = res.intervals
        self.assertEqual(iv0.start.source.material, "shell")
        self.assertEqual(iv0.end.source.material, "liner")
        self.assertEqual(iv1.start.source.material, "liner")
        self.assertEqual(iv1.end.source.material, "shell")

    def test_nested_difference_toggles_facing_by_parity(self):
        # tree = A - (B - C), radii A=20 B=10 C=5, all concentric.
        # Polarity: A +1, B -1 (nested subtractor), C +1 (positive inside it).
        a = Sphere((0, 0, 0), 20, "A", material="steel")
        b = Sphere((0, 0, 0), 10, "B", material="brass")
        c = Sphere((0, 0, 0), 5, "C", material="copper")
        tree = Difference(a, Difference(b, c))
        res = ray_intervals(tree, Ray((-25, 0, 0), (1, 0, 0)))
        # solid inside A and (outside B or inside C)
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(5.0, 15.0), (20.0, 30.0), (35.0, 45.0)],
        )
        expected = [
            (5.0, "enter", "A", "steel", "outward", False, (-1.0, 0.0, 0.0)),
            (15.0, "exit", "B", "brass", "inward", True, (1.0, 0.0, 0.0)),
            (20.0, "enter", "C", "copper", "outward", False, (-1.0, 0.0, 0.0)),
            (30.0, "exit", "C", "copper", "outward", False, (1.0, 0.0, 0.0)),
            (35.0, "enter", "B", "brass", "inward", True, (-1.0, 0.0, 0.0)),
            (45.0, "exit", "A", "steel", "outward", False, (1.0, 0.0, 0.0)),
        ]
        self.assertEqual(len(res.boundaries), len(expected))
        for boundary, (t, kind, leaf, mat, facing, flipped, normal) in zip(
            res.boundaries, expected
        ):
            self.assertEqual((boundary.t, boundary.kind), (t, kind))
            s = boundary.source
            self.assertEqual((s.leaf_id, s.material, s.orientation, s.flipped),
                             (leaf, mat, facing, flipped))
            self.assertEqual(s.normal, normal)
        self.assertEqual(res.nearest_boundary.t, 5.0)
        self.assertEqual(res.nearest_boundary.source.material, "steel")

    def test_difference_of_differences_keeps_inner_subtractor_inward(self):
        # tree = (A - B) - C, concentric radii A=20, B=10, C=5: both B and C
        # surfaces are authored as subtractors (inward).  C lies wholly inside
        # the region B already removed, so C's events at t=20/30 are buried and
        # produce no phantom boundaries; the surviving cavity wall is B's.
        a = Sphere((0, 0, 0), 20, "A", material="steel")
        b = Sphere((0, 0, 0), 10, "B", material="brass")
        c = Sphere((0, 0, 0), 5, "C", material="copper")
        tree = Difference(Difference(a, b), c)
        res = ray_intervals(tree, Ray((-25, 0, 0), (1, 0, 0)))
        self.assertEqual(
            [(iv.t0, iv.t1) for iv in res.intervals],
            [(5.0, 15.0), (35.0, 45.0)],
        )
        self.assertEqual(
            [(bd.t, bd.kind, bd.source.leaf_id, bd.source.orientation,
              bd.source.flipped, bd.source.material) for bd in res.boundaries],
            [
                (5.0, "enter", "A", "outward", False, "steel"),
                (15.0, "exit", "B", "inward", True, "brass"),
                (35.0, "enter", "B", "inward", True, "brass"),
                (45.0, "exit", "A", "outward", False, "steel"),
            ],
        )
        self.assertEqual(
            [bd.t for bd in res.boundaries if bd.source.leaf_id == "C"], []
        )


class TestTangentBoundaries(unittest.TestCase):
    def test_tangent_touch_keeps_material_and_outward_facing(self):
        res = ray_intervals(
            Sphere((0, 0, 0), 2, "s", material="paint"),
            Ray((-5, 2, 0), (1, 0, 0)),
        )
        self.assertEqual(res.intervals, ())
        (touch,) = res.touches
        s = touch.source
        self.assertEqual((touch.t0, touch.t1), (5.0, 5.0))
        self.assertEqual((s.leaf_id, s.material, s.orientation, s.flipped),
                         ("s", "paint", "outward", False))
        self.assertEqual(s.normal, (0.0, 1.0, 0.0))
        self.assertEqual(s.kind, "touch")

    def test_subtractor_tangent_on_face_riding_contact_is_inward_not_flipped(self):
        # Positive A = box [0,4]x[0,2]x[0,2]; the ray (-1,2,1)+t*(1,0,0) rides
        # exactly along A's y_hi face, a zero-thickness contact run for t in
        # [1,5].  Subtractor B = sphere center (2,3,1) r=1 is tangent to that
        # ray at t=3 (hit point (2,2,1), authored outward normal (0,-1,0)).
        # Difference keeps the whole T-run as one touch (1,5) and the
        # subtractor's tangent point joins its sources without splitting it.
        a = Box((0, 0, 0), (4, 2, 2), "A", material="steel")
        b = Sphere((2, 3, 1), 1, "B", material="brass")
        res = ray_intervals(Difference(a, b), Ray((-1, 2, 1), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        (touch,) = res.touches
        self.assertEqual((touch.t0, touch.t1), (1.0, 5.0))

        # the positive riding faces carry their authored outward facing
        a_faces = sorted((s.face, s.orientation, s.flipped)
                         for s in touch.sources if s.leaf_id == "A")
        self.assertEqual(a_faces, [
            ("x_hi", "outward", False),
            ("x_lo", "outward", False),
            ("y_hi", "outward", False),
        ])

        # B authored the contact as a subtractor surface -> inward ...
        b_sources = [s for s in touch.sources if s.leaf_id == "B"]
        self.assertEqual(len(b_sources), 1)
        s = b_sources[0]
        self.assertEqual(s.kind, "touch")
        self.assertEqual(s.material, "brass")
        self.assertEqual(s.orientation, "inward")
        # ... but a zero-measure contact is not a cavity wall, so the geometric
        # normal is B's real outward normal, never negated.
        self.assertFalse(s.flipped)
        self.assertEqual(s.normal, (0.0, -1.0, 0.0))
        self.assertIsNone(res.nearest_boundary)

    def test_externally_tangent_union_touch_is_swallowed(self):
        # two r=2 balls centered (-2,0,0) and (2,0,0), tangent at the origin
        a = Sphere((-2, 0, 0), 2, "a", material="ma")
        b = Sphere((2, 0, 0), 2, "b", material="mb")
        res = ray_intervals(Union([a, b]), Ray((0, 0, 0), (1, 0, 0)))
        # a spans [-4,0], b spans [0,4]: the tangent point at t=0 is swallowed
        self.assertEqual([(iv.t0, iv.t1) for iv in res.intervals], [(-4.0, 4.0)])
        self.assertEqual(res.touches, ())
        # outer faces: a at t=-4, b at t=4
        self.assertEqual([bd.source.material for bd in res.boundaries], ["ma", "mb"])

    def test_two_tangent_leaves_intersect_in_one_multimaterial_touch(self):
        # Two r=2 balls tangent to the x-axis at the origin from opposite
        # sides: both give a zero-measure contact at t=0 and nothing else, so
        # their intersection is exactly that point -- one touch carrying both
        # surfaces, each with its own material and authored normal.
        a = Sphere((0, 2, 0), 2, "a", material="ma")
        b = Sphere((0, -2, 0), 2, "b", material="mb")
        res = ray_intervals(Intersection([a, b]), Ray((0, 0, 0), (1, 0, 0)))
        self.assertEqual(res.intervals, ())
        (touch,) = res.touches
        self.assertEqual((touch.t0, touch.t1), (0.0, 0.0))
        self.assertEqual(
            [(s.leaf_id, s.material, s.orientation, s.normal) for s in touch.sources],
            [
                ("a", "ma", "outward", (0.0, -1.0, 0.0)),
                ("b", "mb", "outward", (0.0, 1.0, 0.0)),
            ],
        )
        self.assertEqual(touch.source.leaf_id, "a")  # leaf-id priority
        self.assertFalse(touch.source.flipped)


class TestMaterialValidation(unittest.TestCase):
    def test_bad_material_ids_rejected_at_construction(self):
        with self.assertRaises(GeometryError) as ctx:
            Sphere((0, 0, 0), 1, "s", material=3)
        self.assertIn(".material", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, "s", material="")
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, "s", material=True)
        with self.assertRaises(GeometryError):
            Box((0, 0, 0), (1, 1, 1), "b", material=None)

    def test_unknown_face_key_rejected(self):
        with self.assertRaises(GeometryError) as ctx:
            Box((0, 0, 0), (1, 1, 1), "b",
                face_materials={"w_lo": "x"})
        self.assertIn("w_lo", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Sphere((0, 0, 0), 1, "s", face_materials={"x_lo": "x"})
        # the sphere override slot accepts exactly its one surface
        s = Sphere((0, 0, 0), 1, "s", face_materials={"sphere": "gloss"})
        self.assertEqual(s.materials["sphere"], "gloss")

    def test_face_materials_must_be_a_mapping(self):
        with self.assertRaises(GeometryError) as ctx:
            Box((0, 0, 0), (1, 1, 1), "b",
                face_materials=[("x_lo", "m")])
        self.assertIn("face_materials", str(ctx.exception))
        with self.assertRaises(GeometryError):
            Box((0, 0, 0), (1, 1, 1), "b", face_materials="x_lo=m")
        with self.assertRaises(GeometryError):
            Box((0, 0, 0), (1, 1, 1), "b",
                face_materials={"x_lo": ""})

    def test_resolved_material_map_is_immutable_and_node_is_value_based(self):
        b1 = Box((0, 0, 0), (1, 1, 1), "b",
                 material="m", face_materials={"x_lo": "r"})
        # resolved per-face map is an immutable mapping ...
        self.assertIsInstance(b1.materials, MappingProxyType)
        with self.assertRaises(TypeError):
            b1.materials["x_lo"] = "other"
        # ... and the override field is normalized to a value-based tuple so
        # frozen nodes stay comparable/hashable even with face overrides
        self.assertEqual(b1.face_materials, (("x_lo", "r"),))
        self.assertEqual(hash(b1), hash(b1))
        b2 = Box((0, 0, 0), (1, 1, 1), "b",
                 material="m", face_materials={"x_lo": "r"})
        self.assertEqual(b1, b2)
        self.assertEqual(hash(b1), hash(b2))
        b3 = Box((0, 0, 0), (1, 1, 1), "b",
                 material="m", face_materials={"x_lo": "different"})
        self.assertNotEqual(b1, b3)
        self.assertIsNone(
            Box((0, 0, 0), (1, 1, 1), "b").face_materials
        )

    def test_failed_validation_leaves_no_partial_state_in_call(self):
        # an invalid tree built from valid-but-duplicate-material leaves still
        # raises from the pure ray_intervals call; no result object exists
        tree = Union([
            Sphere((0, 0, 0), 1, "dup", material="m1"),
            Sphere((5, 0, 0), 1, "dup", material="m2"),
        ])
        with self.assertRaises(GeometryError):
            ray_intervals(tree, Ray((0, 0, 0), (1, 0, 0)))


# ---------------------------------------------------------------------------
# randomized material / orientation consistency
#
# Expectations come straight from the *input* nodes: the material of a source
# must equal that leaf's resolved per-face map, and the orientation must equal
# the subtractor-nesting parity computed by an independent tree walk.  Neither
# is derived from the result under test.
# ---------------------------------------------------------------------------

_MATERIAL_POOL = ("m0", "m1", "m2", "steel", "glass")


def _independent_polarities(tree):
    """leaf_id -> +1 (outward) / -1 (inward), parity of subtractor nesting."""
    polarities = {}

    def walk(node, polarity):
        if isinstance(node, (Sphere, Box)):
            polarities[node.leaf_id] = polarity
        elif isinstance(node, (Union, Intersection)):
            for child in node.children:
                walk(child, polarity)
        else:  # Difference
            walk(node.positive, polarity)
            for sub in node.subtractors:
                walk(sub, -polarity)

    walk(tree, 1)
    return polarities


def _mat_tree(rng, depth, id_counter):
    if depth == 0 or rng.random() < 0.45:
        leaf_id = f"L{next(id_counter)}"
        material = rng.choice(_MATERIAL_POOL)
        if rng.random() < 0.5:
            center = tuple(rng.randint(-3, 3) for _ in range(3))
            return Sphere(center, rng.randint(1, 3), leaf_id, material=material)
        mins = tuple(rng.randint(-3, 2) for _ in range(3))
        maxs = tuple(m + rng.randint(1, 3) for m in mins)
        face_materials = None
        if rng.random() < 0.5:
            face_materials = {
                face: rng.choice(_MATERIAL_POOL)
                for face in rng.sample(BOX_FACES, rng.randint(1, 3))
            }
        return Box(mins, maxs, leaf_id, material=material,
                   face_materials=face_materials)
    op = rng.choice("uid")
    subs = [_mat_tree(rng, depth - 1, id_counter) for _ in range(rng.randint(2, 3))]
    if op == "u":
        return Union(subs)
    if op == "i":
        return Intersection(subs)
    return Difference(subs[0], *subs[1:])


class TestRandomizedMaterialConsistency(unittest.TestCase):
    def test_material_and_orientation_against_input_nodes(self):
        rng = random.Random(20260924)
        cases = 0
        for _ in range(250):
            id_counter = iter(range(1000))
            tree = _mat_tree(rng, depth=3, id_counter=id_counter)
            leaves = {}

            def collect(node):
                if isinstance(node, (Sphere, Box)):
                    leaves[node.leaf_id] = node
                elif isinstance(node, (Union, Intersection)):
                    for child in node.children:
                        collect(child)
                else:
                    collect(node.positive)
                    for sub in node.subtractors:
                        collect(sub)

            collect(tree)
            polarities = _independent_polarities(tree)
            origin = tuple(rng.randint(-5, 5) for _ in range(3))
            direction = tuple(rng.randint(-3, 3) for _ in range(3))
            if not any(direction):
                continue
            res = ray_intervals(tree, Ray(origin, direction))
            cases += 1

            for boundary in res.boundaries:
                for s in boundary.sources:
                    leaf = leaves[s.leaf_id]
                    self.assertEqual(s.material, leaf.materials[s.face])
                    expected_facing = (
                        "inward" if polarities[s.leaf_id] < 0 else "outward"
                    )
                    self.assertEqual(s.orientation, expected_facing)
                    # a solid boundary on an inward-authored face is a cavity
                    # wall and must carry the flipped geometric normal
                    self.assertEqual(
                        s.flipped, polarities[s.leaf_id] < 0
                    )
                # representative follows the fixed (leaf_id, face) priority
                keys = [(s.leaf_id, s.face) for s in boundary.sources]
                self.assertEqual(keys, sorted(keys))
                self.assertIs(boundary.source, boundary.sources[0])

            for touch in res.touches:
                for s in touch.sources:
                    leaf = leaves[s.leaf_id]
                    self.assertEqual(s.material, leaf.materials[s.face])
                    expected_facing = (
                        "inward" if polarities[s.leaf_id] < 0 else "outward"
                    )
                    self.assertEqual(s.orientation, expected_facing)
                    # contacts are never geometrically flipped cavity walls
                    self.assertFalse(s.flipped)
                keys = [(s.leaf_id, s.face, s.kind) for s in touch.sources]
                self.assertEqual(keys, sorted(keys))

        self.assertGreater(cases, 200)


if __name__ == "__main__":
    unittest.main()
