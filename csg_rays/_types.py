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


def _as_mat3(value, path):
    """Validate a 3x3 matrix of numbers; return a Fraction row tuple."""
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{path}: expected a 3x3 matrix, got str/bytes")
    try:
        rows = tuple(value)
    except TypeError:
        raise GeometryError(f"{path}: expected a 3x3 matrix, got {type(value).__name__}")
    if len(rows) != 3:
        raise GeometryError(f"{path}: expected 3 rows, got {len(rows)}")
    matrix = []
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise GeometryError(f"{path}[{i}]: expected a row of 3 numbers, got str/bytes")
        try:
            items = tuple(row)
        except TypeError:
            raise GeometryError(
                f"{path}[{i}]: expected a row of 3 numbers, got {type(row).__name__}"
            )
        if len(items) != 3:
            raise GeometryError(f"{path}[{i}]: expected 3 columns, got {len(items)}")
        matrix.append(
            tuple(_as_number(v, f"{path}[{i}][{j}]") for j, v in enumerate(items))
        )
    return tuple(matrix)


def _mat3_determinant(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _mat3_inverse(matrix):
    """Exact inverse of an invertible 3x3 Fraction matrix."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    det = _mat3_determinant(matrix)
    cofactors = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return tuple(
        tuple(cofactors[row][col] / det for col in range(3)) for row in range(3)
    )


def _mat3_mul(left, right):
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(3))
            for j in range(3)
        )
        for i in range(3)
    )


_IDENTITY3 = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
_ZERO3 = (Fraction(0), Fraction(0), Fraction(0))


def _is_node(value):
    return isinstance(
        value,
        (Sphere, Box, Union, Intersection, Difference, Transform),
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


class Transform:
    """Affine transform ``x_world = linear @ x_local + translation``.

    Nests freely: the child may be any CSG node, including another
    transform.  ``linear`` must be an invertible 3x3 matrix of numbers
    (its determinant is checked exactly); singular matrices are rejected at
    construction time because the inverse is needed to map the ray into the
    child's coordinates.  Use :class:`Translate`, :class:`Scale` and
    :class:`Rotate` for the named convenience forms.
    """

    __slots__ = ("child", "linear", "translation", "_inverse", "_label")

    def __init__(
        self, child, linear=None, translation=None, *,
        _label="Transform", _matrix=None,
    ):
        if _matrix is not None:
            matrix = _matrix
        else:
            matrix = _IDENTITY3 if linear is None else _as_mat3(
                linear, f"{_label}.linear"
            )
        if _mat3_determinant(matrix) == 0:
            raise GeometryError(
                f"{_label}.linear: matrix is singular (determinant 0); the "
                f"transform must be invertible"
            )
        offset = _ZERO3 if translation is None else _as_vec3(
            translation, f"{_label}.translation"
        )
        if not _is_node(child):
            raise GeometryError(
                f"{_label}.child: not a CSG node ({type(child).__name__})"
            )
        self.child = child
        self.linear = matrix
        self.translation = offset
        self._inverse = _mat3_inverse(matrix)
        self._label = _label

    def __repr__(self):
        return (
            f"{self._label}(child={self.child!r}, linear={self.linear!r}, "
            f"translation={self.translation!r})"
        )

    def __eq__(self, other):
        return (
            isinstance(other, Transform)
            and self.child == other.child
            and self.linear == other.linear
            and self.translation == other.translation
        )


class Translate(Transform):
    """Pure translation of a sub-combination by ``offset``."""

    def __init__(self, child, offset):
        super().__init__(child, translation=offset, _label="Translate")


class Scale(Transform):
    """Per-axis non-uniform scaling; every factor must be non-zero.

    Negative factors are accepted (scale combined with a reflection);
    a zero factor makes the transform singular and is rejected.
    """

    def __init__(self, child, factors):
        path = "Scale.factors"
        values = _as_vec3(factors, path)
        for i, factor in enumerate(values):
            if factor == 0:
                raise GeometryError(f"{path}[{i}]: scale factor must be non-zero")
        linear = tuple(
            tuple(values[i] if i == j else Fraction(0) for j in range(3))
            for i in range(3)
        )
        super().__init__(child, _label="Scale", _matrix=linear)


class Rotate(Transform):
    """Linear rotation/isometry given by an invertible 3x3 matrix.

    Only invertibility is required (exact determinant check); exact
    orthogonality is not enforced, so finite ``float`` rotation matrices
    with irrational entries (sin/cos) are accepted.
    """

    def __init__(self, child, matrix):
        super().__init__(child, linear=matrix, _label="Rotate")


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
    ``"touch"`` (tangential / face-riding contact).
    """

    leaf_id: str
    face: str
    kind: str
    normal: tuple
    flipped: bool


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
