"""csg_ray -- CSG ray interval intersection with cavities (offline, stdlib only).

Public API (see README.md for the full contract):

    ray_intervals(tree, ray) -> RayIntervalResult

Tree nodes (frozen dataclasses):
    Sphere(leaf_id, center, radius)
    Box(leaf_id, min_corner, max_corner)
    Union(left, right)
    Intersection(left, right)
    Difference(left, right)
    Ray(origin, direction)

Result:
    RayIntervalResult.intervals        sorted disjoint closed solid intervals
    RayIntervalResult.touches          zero-measure tangent contact points
    RayIntervalResult.nearest_boundary nearest solid boundary with t > 0

Errors:
    InvalidGeometryError, InvalidRayError  -- located contract violations
    BudgetExceededError                    -- tree larger than MAX_NODES
    A geometric miss is *not* an error: intervals/touches come back empty and
    nearest_boundary is None.

Algorithm outline
-----------------
Every leaf is converted into enter/exit events on the ray parameter t (plus
zero-measure "touch" events for tangencies).  Coincident events at one t are
processed as one batch: each CSG node is classified (inside-before,
inside-after, contains-point) bottom-up, so coincident events that do not
change the result set never emit a boundary.  Face normals of leaves inside a
Difference right subtree are emitted flipped (the cavity walls).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Optional

__all__ = [
    "MAX_NODES",
    "Sphere",
    "Box",
    "Union",
    "Intersection",
    "Difference",
    "Ray",
    "FaceSource",
    "Interval",
    "Touch",
    "Boundary",
    "RayIntervalResult",
    "CSGError",
    "InvalidGeometryError",
    "InvalidRayError",
    "BudgetExceededError",
    "ray_intervals",
    "solid_at",
]

MAX_NODES = 100

# Float discriminants inside this *relative* band of zero are re-decided with
# exact Fraction arithmetic and 60-digit Decimal (see _sphere_events).
_DISC_EPS = 1e-12
_DECIMAL_PREC = 60

_AXIS_NAME = ("x", "y", "z")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class CSGError(ValueError):
    """Base class for contract violations located in user input."""


class InvalidGeometryError(CSGError):
    """The CSG tree violates the geometry/structure contract."""


class InvalidRayError(CSGError):
    """The ray violates the contract (non-finite value or zero direction)."""


class BudgetExceededError(RuntimeError):
    """Resource budget exhausted (tree larger than MAX_NODES).

    Kept distinct from a proved geometric miss, which returns an empty result.
    """


# --------------------------------------------------------------------------- #
# Public data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Sphere:
    leaf_id: str
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True, slots=True)
class Box:
    leaf_id: str
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Union:
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Intersection:
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Difference:
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Ray:
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FaceSource:
    """One face responsible for a result boundary.

    normal is the *effective* outward normal of the result surface: faces of
    leaves occurring in the right operand of a Difference are emitted with
    flipped=True and normal negated (the cavity walls).
    """

    leaf_id: str
    face: str  # sphere: "surface"; box: "x-","x+","y-","y+","z-","z+"
    normal: tuple[float, float, float]
    flipped: bool


@dataclass(frozen=True, slots=True)
class Interval:
    """Closed solid parameter interval [t_enter, t_exit] on the ray line."""

    t_enter: float
    t_exit: float
    enter_sources: tuple[FaceSource, ...]
    exit_sources: tuple[FaceSource, ...]

    @property
    def representative_enter(self) -> FaceSource:
        return self.enter_sources[0]

    @property
    def representative_exit(self) -> FaceSource:
        return self.exit_sources[0]


@dataclass(frozen=True, slots=True)
class Touch:
    """Zero-measure tangent contact at parameter t (never solid interior)."""

    t: float
    sources: tuple[FaceSource, ...]

    @property
    def representative(self) -> FaceSource:
        return self.sources[0]


@dataclass(frozen=True, slots=True)
class Boundary:
    t: float
    kind: str  # "enter" or "exit"
    sources: tuple[FaceSource, ...]

    @property
    def representative(self) -> FaceSource:
        return self.sources[0]


@dataclass(frozen=True, slots=True)
class RayIntervalResult:
    intervals: tuple[Interval, ...]
    touches: tuple[Touch, ...]
    nearest_boundary: Optional[Boundary]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _is_real(v: Any) -> bool:
    # bool is a subclass of int and must be rejected for every numeric slot.
    return not isinstance(v, bool) and isinstance(v, (int, float, Decimal))


def _check_finite(v: Any, path: str,
                  err: type[CSGError] = InvalidRayError) -> None:
    if not _is_real(v):
        raise TypeError(f"{path}: expected a real number (int/float/Decimal), "
                        f"got {type(v).__name__}")
    if isinstance(v, float) and not math.isfinite(v):
        raise err(f"{path}: value must be finite, got {v!r}")
    if isinstance(v, Decimal) and (v.is_nan() or v.is_infinite()):
        raise err(f"{path}: value must be finite, got {v!r}")


def _check_vec3(v: Any, path: str,
                err: type[CSGError] = InvalidRayError) -> None:
    if not isinstance(v, (tuple, list)) or isinstance(v, (str, bytes)):
        raise TypeError(f"{path}: expected a 3-sequence of numbers, "
                        f"got {type(v).__name__}")
    if len(v) != 3:
        raise err(f"{path}: expected exactly 3 components, got {len(v)}")
    for i, c in enumerate(v):
        _check_finite(c, f"{path}[{i}]", err)


def _check_id(v: Any, path: str) -> None:
    if not isinstance(v, str):
        raise TypeError(f"{path}: leaf id must be str, got {type(v).__name__}")
    if not v:
        raise InvalidGeometryError(f"{path}: leaf id must be non-empty")


def _to_fraction(v: Any) -> Fraction:
    if isinstance(v, Fraction):
        return v
    return Fraction(v)  # exact for int and binary float


_BINARY_OPS = (Union, Intersection, Difference)
_LEAF_TYPES = (Sphere, Box)
_NODE_TYPES = _LEAF_TYPES + _BINARY_OPS


def _validate_and_index(tree: Any):
    """Enforce the whole structural/value contract in one traversal.

    Returns (leaves, flip, node_count): leaves maps leaf_id to node, flip maps
    leaf_id to normal-flip parity (odd count of Difference-right ancestries).
    The public API is pure, so a raised error can never leave partial state.
    """
    if not isinstance(tree, _NODE_TYPES):
        raise InvalidGeometryError(
            "tree root: must be one of Sphere, Box, Union, Intersection, "
            f"Difference, got {type(tree).__name__}")

    leaves: dict[str, Any] = {}
    flip: dict[str, bool] = {}
    seen: set[int] = set()
    count = 0
    stack: list[tuple[Any, str, bool]] = [(tree, "tree", False)]

    while stack:
        node, path, parity = stack.pop()
        if id(node) in seen:
            raise InvalidGeometryError(
                f"{path}: node is shared by two parents (input must be a "
                "tree, not a DAG; construct separate node objects)")
        seen.add(id(node))
        count += 1
        if count > MAX_NODES:
            raise BudgetExceededError(
                f"{path}: tree exceeds MAX_NODES={MAX_NODES} (budget "
                "exhaustion; distinct from a proved geometric miss)")

        if isinstance(node, Sphere):
            where = f"{path}/Sphere({node.leaf_id!r})"
            _check_id(node.leaf_id, f"{path}.leaf_id")
            _check_vec3(node.center, f"{where}.center", InvalidGeometryError)
            _check_finite(node.radius, f"{where}.radius",
                          InvalidGeometryError)
            if node.radius <= 0:
                raise InvalidGeometryError(
                    f"{where}.radius: must be > 0, got {node.radius!r}")
            if node.leaf_id in leaves:
                raise InvalidGeometryError(
                    f"{path}: duplicate leaf id {node.leaf_id!r}")
            leaves[node.leaf_id] = node
            flip[node.leaf_id] = parity
        elif isinstance(node, Box):
            where = f"{path}/Box({node.leaf_id!r})"
            _check_id(node.leaf_id, f"{path}.leaf_id")
            _check_vec3(node.min_corner, f"{where}.min_corner",
                        InvalidGeometryError)
            _check_vec3(node.max_corner, f"{where}.max_corner",
                        InvalidGeometryError)
            for k in range(3):
                if not node.min_corner[k] < node.max_corner[k]:
                    raise InvalidGeometryError(
                        f"{where}: min_corner[{k}] must be strictly < "
                        f"max_corner[{k}], got {node.min_corner[k]!r} >= "
                        f"{node.max_corner[k]!r}")
            if node.leaf_id in leaves:
                raise InvalidGeometryError(
                    f"{path}: duplicate leaf id {node.leaf_id!r}")
            leaves[node.leaf_id] = node
            flip[node.leaf_id] = parity
        else:
            if not isinstance(node.left, _NODE_TYPES):
                raise InvalidGeometryError(
                    f"{path}.{type(node).__name__}.left: must be a CSG node, "
                    f"got {type(node.left).__name__}")
            if not isinstance(node.right, _NODE_TYPES):
                raise InvalidGeometryError(
                    f"{path}.{type(node).__name__}.right: must be a CSG "
                    f"node, got {type(node.right).__name__}")
            stack.append((node.left,
                          f"{path}.{type(node).__name__}.left", parity))
            stack.append((node.right,
                          f"{path}.{type(node).__name__}.right",
                          parity ^ isinstance(node, Difference)))

    return leaves, flip, count


# --------------------------------------------------------------------------- #
# Leaf primitives -> events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Event:
    t: Fraction
    leaf_id: str
    kind: str  # "enter" / "exit" / "touch"
    # faces responsible at this t: (face name, outward normal, exact rationals)
    faces: tuple[tuple[str, tuple[Fraction, Fraction, Fraction]], ...]


def _axis_normal(k: int, positive: bool):
    sign = 1 if positive else -1
    return tuple(Fraction(sign if i == k else 0) for i in range(3))


def _sphere_events(sphere: Sphere, origin, direction) -> list[_Event]:
    c = tuple(_to_fraction(v) for v in sphere.center)
    r = _to_fraction(sphere.radius)
    f = tuple(origin[k] - c[k] for k in range(3))

    # |o + d t - c|^2 = r^2  ->  a t^2 + b t + q = 0
    a = sum(direction[k] * direction[k] for k in range(3))
    b = 2 * sum(f[k] * direction[k] for k in range(3))
    q = sum(f[k] * f[k] for k in range(3)) - r * r

    lid = sphere.leaf_id

    def normal_at(t: Fraction):
        return (("surface",
                 tuple((f[k] + direction[k] * t) / r for k in range(3))),)

    # Exact rational discriminant first; a float estimate only decides whether
    # the expensive Decimal promotion is needed.
    disc_exact = b * b - 4 * a * q
    if disc_exact == 0:
        t = -b / (2 * a)
        return [_Event(t, lid, "touch", normal_at(t))]

    bf = float(b)
    qf = float(q)
    af = float(a)
    disc_f = bf * bf - 4.0 * af * qf
    scale = max(1.0, abs(bf * bf), abs(4.0 * af * qf))
    near_degenerate = abs(disc_f) <= _DISC_EPS * scale

    if disc_exact > 0:
        if near_degenerate:
            # Promote root extraction too, so the close double root is
            # evaluated at 60-digit precision rather than by float sqrt.
            with localcontext() as ctx:
                ctx.prec = _DECIMAL_PREC
                ad = Decimal(a.numerator) / Decimal(a.denominator)
                bd = Decimal(b.numerator) / Decimal(b.denominator)
                sd = (bd * bd - Decimal(4) * ad
                      * (Decimal(q.numerator)
                         / Decimal(q.denominator))).sqrt()
                r1 = Fraction((-bd - sd) / (2 * ad))
                r2 = Fraction((-bd + sd) / (2 * ad))
            t1, t2 = sorted((r1, r2))
        else:
            s = math.sqrt(disc_f)
            t1, t2 = sorted((Fraction((-bf - s) / (2.0 * af)),
                             Fraction((-bf + s) / (2.0 * af))))
        return [
            _Event(t1, lid, "enter", normal_at(t1)),
            _Event(t2, lid, "exit", normal_at(t2)),
        ]

    # disc_exact < 0: genuine miss; only float roundoff could make it look
    # tangent, and the exact Fraction comparison is authoritative.
    return []


def _box_events(box: Box, origin, direction) -> list[_Event]:
    lo = tuple(_to_fraction(v) for v in box.min_corner)
    hi = tuple(_to_fraction(v) for v in box.max_corner)
    lid = box.leaf_id

    # Slab intersection of three axis intervals.
    nears: list[tuple[Fraction, str, tuple]] = []
    fars: list[tuple[Fraction, str, tuple]] = []
    for k in range(3):
        ax = _AXIS_NAME[k]
        d = direction[k]
        o = origin[k]
        if d == 0:
            if o < lo[k] or o > hi[k]:
                return []  # parallel to slab and outside it
            if o == lo[k] or o == hi[k]:
                # Ray lies in a face plane.  Such a 2D-face run is declared
                # out of scope and treated as not-solid-interior (see README:
                # known limitations).  Other axes still constrain nothing.
                return []
            continue
        t_lo = (lo[k] - o) / d
        t_hi = (hi[k] - o) / d
        if t_lo < t_hi:
            n_name, n_norm = f"{ax}-", _axis_normal(k, False)
            x_name, x_norm = f"{ax}+", _axis_normal(k, True)
        else:
            t_lo, t_hi = t_hi, t_lo
            n_name, n_norm = f"{ax}+", _axis_normal(k, True)
            x_name, x_norm = f"{ax}-", _axis_normal(k, False)
        nears.append((t_lo, n_name, n_norm))
        fars.append((t_hi, x_name, x_norm))

    t0 = max(n[0] for n in nears)
    t1 = min(n[0] for n in fars)

    if t0 == t1:
        # Contact only at an edge (2 faces) or corner (3 faces): a touch.
        faces = tuple((n[1], n[2]) for n in nears if n[0] == t0)
        for n in fars:
            if n[0] == t1 and all(n[1] != f[0] for f in faces):
                faces += ((n[1], n[2]),)
        return [_Event(t0, lid, "touch", faces)]
    if t0 > t1:
        return []

    enter_faces = tuple((n[1], n[2]) for n in nears if n[0] == t0)
    exit_faces = tuple((n[1], n[2]) for n in fars if n[0] == t1)
    return [
        _Event(t0, lid, "enter", enter_faces),
        _Event(t1, lid, "exit", exit_faces),
    ]


# --------------------------------------------------------------------------- #
# Boolean event sweep
# --------------------------------------------------------------------------- #


def _collect_leaf_events(leaves, origin, direction) -> list[_Event]:
    events: list[_Event] = []
    for node in leaves.values():
        if isinstance(node, Sphere):
            events.extend(_sphere_events(node, origin, direction))
        else:
            events.extend(_box_events(node, origin, direction))
    return events


def _sweep(tree, leaves, events):
    groups: dict[Fraction, list[_Event]] = {}
    for ev in events:
        groups.setdefault(ev.t, []).append(ev)

    inside = {lid: False for lid in leaves}
    boundaries: list[tuple[Fraction, str, list[tuple[str, str]]]] = []
    touches: list[tuple[Fraction, list[tuple[str, str]]]] = []

    def node_status(node, before, after, touch, memo):
        """(inside just before t, inside just after t, contains point t)."""
        key = id(node)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if isinstance(node, (Sphere, Box)):
            b = before[node.leaf_id]
            res = (b, after[node.leaf_id],
                   b or after[node.leaf_id] or touch[node.leaf_id])
        else:
            lb, la, lp = node_status(node.left, before, after, touch, memo)
            rb, ra, rp = node_status(node.right, before, after, touch, memo)
            if isinstance(node, Union):
                res = (lb or rb, la or ra, lp or rp)
            elif isinstance(node, Intersection):
                res = (lb and rb, la and ra, lp and rp)
            else:  # Difference: left \\ right
                res = (lb and not rb, la and not ra, lp and not rp)
        memo[key] = res
        return res

    def boundary_refs(node, role, before, after, touch, group, memo):
        """(leaf_id, face) pairs witnessing node's enter/exit transition."""
        key = (id(node), role)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if isinstance(node, (Sphere, Box)):
            res = [(node.leaf_id, fc[0])
                   for fc in group[node.leaf_id][role]]
            memo[key] = res
            return res

        lb, la, _ = node_status(node.left, before, after, touch, st_memo)
        rb, ra, _ = node_status(node.right, before, after, touch, st_memo)
        refs: list[tuple[str, str]] = []

        def left_child(want_role):
            return boundary_refs(node.left, want_role, before, after, touch,
                                 group, memo)

        def right_child(want_role):
            return boundary_refs(node.right, want_role, before, after, touch,
                                 group, memo)

        if isinstance(node, Union):
            if role == "enter":
                if not lb and la:
                    refs += left_child("enter")
                if not rb and ra:
                    refs += right_child("enter")
            else:
                if lb and not la:
                    refs += left_child("exit")
                if rb and not ra:
                    refs += right_child("exit")
        elif isinstance(node, Intersection):
            if role == "enter":
                if not lb and la:
                    refs += left_child("enter")
                if not rb and ra:
                    refs += right_child("enter")
            else:
                if lb and not la:
                    refs += left_child("exit")
                if rb and not ra:
                    refs += right_child("exit")
        else:  # Difference: left faces keep role; right faces invert role
            if role == "enter":  # result F->T: left F->T, or right T->F
                if not lb and la:
                    refs += left_child("enter")
                if rb and not ra:
                    refs += right_child("exit")
            else:  # result T->F: left T->F, or right F->T
                if lb and not la:
                    refs += left_child("exit")
                if not rb and ra:
                    refs += right_child("enter")
        memo[key] = refs
        return refs

    def point_refs(node, before, after, touch, group, memo):
        """(leaf_id, face) pairs witnessing that point t belongs to node."""
        key = id(node)
        if key in memo:
            return memo[key]
        if isinstance(node, (Sphere, Box)):
            lid = node.leaf_id
            b, a = before[lid], after[lid]
            if touch[lid]:
                refs = [(lid, fc[0]) for fc in group[lid]["touch"]]
            elif not b and a:
                refs = [(lid, fc[0]) for fc in group[lid]["enter"]]
            elif b and not a:
                refs = [(lid, fc[0]) for fc in group[lid]["exit"]]
            else:
                refs = []  # strictly interior on both sides: face-free point
            memo[key] = refs
            return refs

        lb, la, lp = node_status(node.left, before, after, touch, st_memo)
        rb, ra, rp = node_status(node.right, before, after, touch, st_memo)
        refs: list[tuple[str, str]] = []
        if isinstance(node, Union):
            if lp:
                refs += point_refs(node.left, before, after, touch, group, memo)
            if rp:
                refs += point_refs(node.right, before, after, touch,
                                   group, memo)
        elif isinstance(node, Intersection):
            if lp and rp:
                refs += point_refs(node.left, before, after, touch, group, memo)
                refs += point_refs(node.right, before, after, touch,
                                   group, memo)
        else:  # Difference
            if lp and not rp:
                refs += point_refs(node.left, before, after, touch, group, memo)
        memo[id(node)] = refs
        return refs

    for t in sorted(groups):
        group: dict[str, dict[str, list]] = {}
        touch = {lid: False for lid in leaves}
        for ev in groups[t]:
            slot = group.setdefault(
                ev.leaf_id, {"enter": [], "exit": [], "touch": []})
            slot[ev.kind].extend(ev.faces)
            if ev.kind == "touch":
                touch[ev.leaf_id] = True

        after = dict(inside)
        for lid, slot in group.items():
            if slot["enter"]:
                after[lid] = True
            if slot["exit"]:
                after[lid] = False

        st_memo: dict[int, tuple[bool, bool, bool]] = {}
        b_root, a_root, p_root = node_status(tree, inside, after, touch,
                                             st_memo)

        if b_root != a_root:
            role = "enter" if a_root else "exit"
            refs = boundary_refs(tree, role, inside, after, touch, group, {})
            boundaries.append((t, role, refs))
        elif not b_root and not a_root and p_root:
            # Outside on both sides but the exact point belongs to the
            # result: a zero-measure touch of the combined set.  A touch
            # buried strictly inside another solid (both sides solid) is
            # suppressed here, as required.
            refs = point_refs(tree, inside, after, touch, group, {})
            if refs:
                touches.append((t, refs))
        # otherwise interior run, or state-equal coincident events with no
        # point membership: no spurious boundary is emitted.

        inside = after

    return boundaries, touches


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def ray_intervals(tree, ray: Ray) -> RayIntervalResult:
    """Classify a ray against a CSG tree (pure function).

    Raises TypeError / InvalidGeometryError / InvalidRayError /
    BudgetExceededError on contract violations; a geometric miss returns an
    empty (but valid) result.  A failed call raises before returning, so no
    partial state can survive it.
    """
    leaves, flip, _count = _validate_and_index(tree)

    if not isinstance(ray, Ray):
        raise InvalidRayError(f"ray: must be Ray, got {type(ray).__name__}")
    _check_vec3(ray.origin, "ray.origin")
    _check_vec3(ray.direction, "ray.direction")
    origin = tuple(_to_fraction(v) for v in ray.origin)
    direction = tuple(_to_fraction(v) for v in ray.direction)
    if all(d == 0 for d in direction):
        raise InvalidRayError("ray.direction: must be non-zero")

    events = _collect_leaf_events(leaves, origin, direction)
    boundaries, touches = _sweep(tree, leaves, events)

    # (t, leaf_id, face name) -> (face name, exact normal)
    face_at: dict[tuple[Fraction, str, str],
                  tuple[str, tuple[Fraction, ...]]] = {}
    for ev in events:
        for fname, nvec in ev.faces:
            face_at[(ev.t, ev.leaf_id, fname)] = (fname, nvec)

    def resolve(t: Fraction, refs: list[tuple[str, str]]):
        sources = []
        seen: set[tuple[str, str]] = set()
        for lid, fname in sorted(set(refs)):
            if (lid, fname) in seen:
                continue
            seen.add((lid, fname))
            hit = face_at.get((t, lid, fname))
            if hit is None:
                continue
            name, nvec = hit
            n = tuple(float(-x) if flip[lid] else float(x) for x in nvec)
            sources.append(FaceSource(lid, name, n, bool(flip[lid])))
        return tuple(sources)

    intervals_out: list[Interval] = []
    pending: Optional[tuple[Fraction, tuple[FaceSource, ...]]] = None
    for t, role, refs in sorted(boundaries, key=lambda x: x[0]):
        sources = resolve(t, refs)
        if role == "enter":
            pending = (t, sources)
        else:
            t_enter, enter_sources = pending
            intervals_out.append(
                Interval(float(t_enter), float(t), enter_sources, sources))
            pending = None

    touches_out = tuple(
        Touch(float(t), resolve(t, refs))
        for t, refs in sorted(touches, key=lambda x: x[0]))
    touches_out = tuple(tc for tc in touches_out if tc.sources)

    nearest = None
    for t, role, refs in sorted(boundaries, key=lambda x: x[0]):
        if t > 0:  # strictly forward; a t==0 surface origin is not "positive"
            nearest = Boundary(float(t), role, resolve(t, refs))
            break

    return RayIntervalResult(tuple(intervals_out), touches_out, nearest)


def solid_at(result: RayIntervalResult, t: float) -> bool:
    """Closed-interval membership of parameter t in the result.

    Touch points are not interior, so this is True only within a solid
    interval (endpoints included).
    """
    _check_finite(t, "t")
    return any(iv.t_enter <= t <= iv.t_exit for iv in result.intervals)
