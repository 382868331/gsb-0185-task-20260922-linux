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

#: Maximum number of nodes (leaves + internal nodes) accepted in a CSG tree.
MAX_TREE_NODES = 100


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
    return isinstance(value, (Sphere, Box, Union, Intersection, Difference, Instance))


def _as_node_id(value, path):
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise GeometryError(f"{path}: must be a non-empty str")
    return value


def _as_matrix3(value, path):
    """Validate a 3x3 numeric matrix; return 3 rows of Fraction triples."""
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{path}: expected a 3x3 matrix (3 rows of 3 numbers)")
    try:
        rows = tuple(value)
    except TypeError:
        raise GeometryError(
            f"{path}: expected a 3x3 matrix, got {type(value).__name__}"
        )
    if len(rows) != 3:
        raise GeometryError(f"{path}: expected 3 rows, got {len(rows)}")
    return tuple(_as_vec3(row, f"{path}[{i}]") for i, row in enumerate(rows))


def _mat3_det(m):
    """Exact determinant of a Fraction 3x3 matrix."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _mat3_inverse(m):
    """Exact inverse (adjugate / determinant) of a nonsingular Fraction 3x3."""
    det = _mat3_det(m)

    def minor(i, j):
        rows = [r for r in range(3) if r != i]
        cols = [c for c in range(3) if c != j]
        return (
            m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
            - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]
        )

    return tuple(
        tuple(((-1) ** (i + j)) * minor(j, i) / det for j in range(3))
        for i in range(3)
    )


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


@dataclass(frozen=True)
class Instance:
    """Reference to an existing CSG subtree under an invertible affine transform.

    ``Instance(child, matrix, translation, instance_id)`` places one
    *reference* to ``child`` at ``matrix @ x + translation`` (the transform
    maps the child's local coordinates to the coordinates of this node).  The
    referenced subtree is shared, never copied or modified; several Instance
    nodes may reference the same subtree object.  ``matrix`` must be a
    nonsingular 3x3 matrix (singular matrices are rejected with
    :class:`GeometryError`).  Reference cycles (a subtree that, directly or
    transitively, references itself) are rejected by :func:`ray_intervals`.

    ``matrix``/``translation`` are stored as exact Fractions; ``inverse`` is
    the exact inverse matrix (Fractions), precomputed at construction.
    """

    child: object
    matrix: object
    translation: object
    instance_id: str

    def __post_init__(self):
        path = f"Instance(instance_id={self.instance_id!r})"
        object.__setattr__(
            self, "instance_id", _as_node_id(self.instance_id, f"{path}.instance_id")
        )
        if not _is_node(self.child):
            raise GeometryError(
                f"{path}.child: not a CSG node ({type(self.child).__name__})"
            )
        matrix = _as_matrix3(self.matrix, f"{path}.matrix")
        if _mat3_det(matrix) == 0:
            raise GeometryError(
                f"{path}.matrix: singular matrix (determinant is 0); "
                f"the instance transform must be invertible"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(
            self, "translation", _as_vec3(self.translation, f"{path}.translation")
        )
        object.__setattr__(self, "inverse", _mat3_inverse(matrix))
        inv = self.inverse
        object.__setattr__(
            self, "translation_inv",
            tuple(
                -sum(inv[i][j] * self.translation[j] for j in range(3))
                for i in range(3)
            ),
        )

    def __repr__(self):
        return (
            f"Instance(instance_id={self.instance_id!r}, "
            f"child={type(self.child).__name__})"
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
    """One face of one leaf participating in a boundary or touch.

    ``face`` is ``"sphere"`` for spheres and one of ``"x_lo"``, ``"x_hi"``,
    ``"y_lo"``, ``"y_hi"``, ``"z_lo"``, ``"z_hi"`` for boxes.  ``normal`` is
    the outward unit normal of that face at the hit point, flipped when the
    leaf acts as a subtractor inside :class:`Difference` (``flipped`` is then
    True).  ``kind`` is the leaf-side event kind: ``"enter"``, ``"exit"`` or
    ``"touch"`` (tangential / face-riding contact).  ``leaf_id`` is the id of
    the referenced leaf; ``path`` is the tuple of ``instance_id``s (outermost
    first) of the :class:`Instance` references through which the leaf was
    reached -- ``()`` for leaves placed directly in the tree, so two
    instances of one shared subtree yield distinct sources for the same
    original leaf.
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
    ``sources`` holds every contributing face sorted by ``(leaf_id, face)``;
    ``source`` is the representative, the first source in leaf-id
    lexicographic order.
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
