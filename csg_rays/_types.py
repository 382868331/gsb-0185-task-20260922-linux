"""Public data structures, input validation and error semantics for csg_rays.

Contract highlights (see README.md for the full specification):

* Numeric inputs must be ``int`` or ``float``; ``bool`` is rejected everywhere
  (it is an ``int`` subclass but never a valid geometric quantity here), and
  NaN / Infinity are rejected.  Valid numbers are stored exactly as
  ``fractions.Fraction``.
* All validation errors raise :class:`GeometryError` with a locatable path in
  the message (e.g. ``Sphere(leaf_id='s1').center[2]: ...``).  Nothing is
  silently corrected.
* :class:`CertificationBudgetExhausted` is raised when the ordering of two
  distinct event parameters cannot be certified within the Decimal precision
  budget.  This is deliberately separate from a legitimately empty result
  ("proven no intersection"), which is reported as empty intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

#: Maximum number of distinct declared nodes accepted in a CSG DAG.
MAX_TREE_NODES = 100

#: Maximum number of leaf occurrences produced after instance expansion.
#: A DAG with at most 100 distinct nodes can still expand exponentially when
#: a subtree references another shared subtree repeatedly; this cap bounds the
#: actual sweep work independently of the declared-node budget.
MAX_LEAF_OCCURRENCES = 2000


class CSGError(ValueError):
    """Base class for all errors raised by csg_rays."""


class GeometryError(CSGError):
    """Invalid geometric input (type, finiteness, shape or tree structure)."""


class CertificationBudgetExhausted(CSGError):
    """Event ordering could not be certified within the precision budget.

    Distinct from "no intersection": the geometry exists but two event
    parameters are so close that the built-in Decimal budget (up to 240
    digits) cannot prove their order.
    """


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _as_number(value, path):
    """Validate a single numeric input and return it as an exact Fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryError(
            f"{path}: expected int or float (bool is not accepted), "
            f"got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise GeometryError(f"{path}: NaN/Infinity is not allowed")
    return Fraction(value)


def _as_vec3(value, path):
    """Validate a 3-component numeric sequence; return a Fraction tuple."""
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{path}: expected a sequence of 3 numbers, got str/bytes")
    try:
        items = tuple(value)
    except TypeError:
        raise GeometryError(
            f"{path}: expected a sequence of 3 numbers, got {type(value).__name__}"
        )
    if len(items) != 3:
        raise GeometryError(f"{path}: expected 3 components, got {len(items)}")
    return tuple(_as_number(v, f"{path}[{i}]") for i, v in enumerate(items))


def _as_leaf_id(value, path):
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise GeometryError(f"{path}: leaf_id must be a non-empty str")
    return value


def _is_node(value):
    return isinstance(
        value, (Sphere, Box, Union, Intersection, Difference, Instance)
    )


def _as_mat3(value, path):
    """Validate a 3x3 numeric matrix; return a row-major 9-tuple of Fractions."""
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{path}: expected 3 rows of 3 numbers, got str/bytes")
    try:
        rows = tuple(value)
    except TypeError:
        raise GeometryError(
            f"{path}: expected a 3x3 matrix, got {type(value).__name__}"
        )
    if len(rows) != 3:
        raise GeometryError(f"{path}: expected 3 rows, got {len(rows)}")
    matrix = []
    for r, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise GeometryError(
                f"{path}[{r}]: expected a row of 3 numbers, got str/bytes"
            )
        try:
            items = tuple(row)
        except TypeError:
            raise GeometryError(
                f"{path}[{r}]: expected a row of 3 numbers, "
                f"got {type(row).__name__}"
            )
        if len(items) != 3:
            raise GeometryError(
                f"{path}[{r}]: expected 3 columns, got {len(items)}"
            )
        for c, item in enumerate(items):
            matrix.append(_as_number(item, f"{path}[{r}][{c}]"))
    return tuple(matrix)


def _check_children(children, path):
    if isinstance(children, (str, bytes)):
        raise GeometryError(f"{path}: expected an iterable of CSG nodes, got str/bytes")
    try:
        items = tuple(children)
    except TypeError:
        raise GeometryError(f"{path}: expected an iterable of CSG nodes")
    if not items:
        raise GeometryError(f"{path}: must contain at least one child")
    for i, child in enumerate(items):
        if not _is_node(child):
            raise GeometryError(
                f"{path}[{i}]: not a CSG node ({type(child).__name__})"
            )
    return items


# ---------------------------------------------------------------------------
# CSG tree nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sphere:
    """Solid sphere leaf.  ``center``/``radius`` are stored as exact Fractions."""

    center: object
    radius: object
    leaf_id: str

    def __post_init__(self):
        path = f"Sphere(leaf_id={self.leaf_id!r})"
        object.__setattr__(self, "leaf_id", _as_leaf_id(self.leaf_id, path))
        object.__setattr__(self, "center", _as_vec3(self.center, f"{path}.center"))
        radius = _as_number(self.radius, f"{path}.radius")
        if radius <= 0:
            raise GeometryError(f"{path}.radius: must be > 0, got {self.radius!r}")
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class Box:
    """Axis-aligned solid box leaf.  ``mins[i] < maxs[i]`` is required."""

    mins: object
    maxs: object
    leaf_id: str

    def __post_init__(self):
        path = f"Box(leaf_id={self.leaf_id!r})"
        object.__setattr__(self, "leaf_id", _as_leaf_id(self.leaf_id, path))
        mins = _as_vec3(self.mins, f"{path}.mins")
        maxs = _as_vec3(self.maxs, f"{path}.maxs")
        for i in range(3):
            if not mins[i] < maxs[i]:
                raise GeometryError(
                    f"{path}: maxs[{i}] must be > mins[{i}], "
                    f"got {self.mins[i]!r} .. {self.maxs[i]!r}"
                )
        object.__setattr__(self, "mins", mins)
        object.__setattr__(self, "maxs", maxs)


@dataclass(frozen=True)
class Union:
    """N-ary union of CSG nodes."""

    children: object

    def __post_init__(self):
        object.__setattr__(
            self, "children", _check_children(self.children, "Union.children")
        )


@dataclass(frozen=True)
class Intersection:
    """N-ary intersection of CSG nodes."""

    children: object

    def __post_init__(self):
        object.__setattr__(
            self, "children", _check_children(self.children, "Intersection.children")
        )


class Difference:
    """``Difference(positive, *subtractors)``: positive minus the subtractors.

    Subtractors are treated as closed solids (regularized difference); their
    boundary-face normals are reported flipped in the output.
    """

    __slots__ = ("positive", "subtractors")

    def __init__(self, positive, *subtractors):
        if not _is_node(positive):
            raise GeometryError(
                f"Difference.positive: not a CSG node ({type(positive).__name__})"
            )
        if not subtractors:
            raise GeometryError("Difference: at least one subtractor is required")
        for i, sub in enumerate(subtractors):
            if not _is_node(sub):
                raise GeometryError(
                    f"Difference.subtractors[{i}]: not a CSG node "
                    f"({type(sub).__name__})"
                )
        self.positive = positive
        self.subtractors = tuple(subtractors)

    def __repr__(self):
        return f"Difference({self.positive!r}, *{self.subtractors!r})"

    def __eq__(self, other):
        return (
            isinstance(other, Difference)
            and self.positive == other.positive
            and self.subtractors == other.subtractors
        )


class Instance:
    """``Instance(child, linear, translation, instance_id)``: one reference
    to an existing CSG subtree placed by the invertible affine map
    ``p_world = A @ p_local + b``.

    The transform belongs to this reference only: the shared child node is
    never modified and may be referenced from multiple ``Instance`` nodes.
    ``linear`` is a 3x3 matrix and ``translation`` a 3-vector, all entries
    ``int``/``float`` (stored exactly as Fractions); the matrix must be
    invertible (non-zero determinant) -- singular matrices, including
    reflections composed to rank-deficient maps, are rejected.
    """

    __slots__ = ("child", "linear", "translation", "instance_id")

    def __init__(self, child, linear, translation=(0, 0, 0),
                 instance_id="instance"):
        if not _is_node(child):
            raise GeometryError(
                f"Instance(instance_id={instance_id!r}).child: not a CSG node "
                f"({type(child).__name__})"
            )
        path = f"Instance(instance_id={instance_id!r})"
        if isinstance(instance_id, bool) or not isinstance(instance_id, str) \
                or not instance_id:
            raise GeometryError(f"{path}.instance_id: must be a non-empty str")
        linear = _as_mat3(linear, f"{path}.linear")
        translation = _as_vec3(translation, f"{path}.translation")
        det = (
            linear[0] * (linear[4] * linear[8] - linear[5] * linear[7])
            - linear[1] * (linear[3] * linear[8] - linear[5] * linear[6])
            + linear[2] * (linear[3] * linear[7] - linear[4] * linear[6])
        )
        if det == 0:
            raise GeometryError(
                f"{path}.linear: matrix must be invertible, got a singular "
                f"matrix (det = 0)"
            )
        self.child = child
        self.linear = linear
        self.translation = translation
        self.instance_id = instance_id

    def __repr__(self):
        return (
            f"Instance({self.child!r}, {self.linear!r}, "
            f"{self.translation!r}, {self.instance_id!r})"
        )


@dataclass(frozen=True)
class Ray:
    """Ray ``origin + t * direction`` over the whole real line (t in R).

    The direction must be non-zero.  Intervals are computed for all real t;
    ``nearest_boundary`` of the result restricts to t > 0.
    """

    origin: object
    direction: object

    def __post_init__(self):
        object.__setattr__(self, "origin", _as_vec3(self.origin, "Ray.origin"))
        direction = _as_vec3(self.direction, "Ray.direction")
        if all(component == 0 for component in direction):
            raise GeometryError("Ray.direction: must be a non-zero vector")
        object.__setattr__(self, "direction", direction)


# ---------------------------------------------------------------------------
# output structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """One face of one leaf occurrence participating in a boundary or touch.

    ``leaf_id`` identifies the original leaf definition.  ``path`` is the
    tuple of ``instance_id`` values of the :class:`Instance` nodes traversed
    from the scene root to that occurrence (empty for an un-instanced leaf):
    two references to the same shared subtree are therefore distinct sources
    ``(path, leaf_id)``.

    ``face`` is ``"sphere"`` for spheres and one of ``"x_lo"``, ``"x_hi"``,
    ``"y_lo"``, ``"y_hi"``, ``"z_lo"``, ``"z_hi"`` for boxes.  ``normal`` is
    the outward unit normal of that face at the hit point, in world
    coordinates (pushed through the inverse-transpose of the occurrence
    transform and renormalized), flipped when the occurrence acts as a
    subtractor inside :class:`Difference` (``flipped`` is then True).
    ``kind`` is the leaf-side event kind: ``"enter"``, ``"exit"`` or
    ``"touch"`` (tangential / face-riding contact).
    """

    leaf_id: str
    face: str
    kind: str
    normal: tuple
    flipped: bool
    path: tuple = ()


@dataclass(frozen=True)
class Boundary:
    """A solid boundary of the CSG result at parameter ``t``.

    ``kind`` is ``"enter"`` (empty -> solid) or ``"exit"`` (solid -> empty).
    ``sources`` holds every contributing face sorted by
    ``(path, leaf_id, face)``; ``source`` is the representative, the first
    source in that order.
    """

    t: float
    kind: str
    source: Source
    sources: tuple


@dataclass(frozen=True)
class Interval:
    """Open solid interval ``(t0, t1)`` with its boundary objects."""

    t0: float
    t1: float
    start: Boundary
    end: Boundary


@dataclass(frozen=True)
class Touch:
    """Zero-measure tangential contact, not part of the solid interior.

    An isolated tangent point has ``t0 == t1``; a ray riding along a box face
    produces a contact segment ``t0 < t1`` which still has zero thickness.
    """

    t0: float
    t1: float
    source: Source
    sources: tuple


@dataclass(frozen=True)
class RayIntervalsResult:
    """Result of :func:`csg_rays.ray_intervals`.

    ``intervals`` are sorted, pairwise disjoint open solid intervals.
    ``touches`` are zero-measure contacts (never solid interior).
    ``boundaries`` are all solid boundaries sorted by t.
    ``nearest_boundary`` is the boundary with the smallest t > 0, or None.
    """

    intervals: tuple
    touches: tuple
    boundaries: tuple
    nearest_boundary: object
