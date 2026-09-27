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
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
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
        value,
        (Sphere, Box, Union, Intersection, Difference, Transform),
    )


def _as_matrix3(value, path):
    """Validate an invertible 3x3 numeric matrix and return Fractions."""
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{path}: expected a 3x3 sequence, got str/bytes")
    try:
        rows = tuple(value)
    except TypeError:
        raise GeometryError(f"{path}: expected a 3x3 sequence")
    if len(rows) != 3:
        raise GeometryError(f"{path}: expected 3 rows, got {len(rows)}")
    matrix = []
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise GeometryError(f"{path}[{i}]: expected a row of 3 numbers")
        try:
            items = tuple(row)
        except TypeError:
            raise GeometryError(f"{path}[{i}]: expected a row of 3 numbers")
        if len(items) != 3:
            raise GeometryError(
                f"{path}[{i}]: expected 3 columns, got {len(items)}"
            )
        matrix.append(
            tuple(_as_number(v, f"{path}[{i}][{j}]") for j, v in enumerate(items))
        )
    determinant = _det3(matrix)
    if determinant == 0:
        raise GeometryError(f"{path}: transform matrix is singular and cannot be inverted")
    return tuple(matrix)


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _identity3():
    return (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rotation_matrix(angle, axis, path):
    angle_value = _as_number(angle, f"{path}.angle")
    axis_value = _as_vec3(axis, f"{path}.axis")
    if all(component == 0 for component in axis_value):
        raise GeometryError(f"{path}.axis: rotation axis must be non-zero")

    angle_float = float(angle_value)
    quadrant = round(angle_float / (math.pi / 2))
    if math.isclose(angle_float, quadrant * math.pi / 2, rel_tol=1e-12, abs_tol=1e-12):
        c = Fraction((1, 0, -1, 0)[quadrant % 4])
        s = Fraction((0, 1, 0, -1)[quadrant % 4])
        axis_length_squared = sum(component * component for component in axis_value)
        axis_length = axis_length_squared.numerator
        denominator = axis_length_squared.denominator
        root_num = math.isqrt(axis_length)
        root_den = math.isqrt(denominator)
        if root_num * root_num == axis_length and root_den * root_den == denominator:
            length = Fraction(root_num, root_den)
            u = tuple(component / length for component in axis_value)
            zero = Fraction(0)
            one = Fraction(1)
            return tuple(
                tuple(
                    c * (one if i == j else zero)
                    + s * (
                        u[(i + 1) % 3] * u[(j + 2) % 3]
                        - u[(i + 2) % 3] * u[(j + 1) % 3]
                    )
                    + (one - c) * u[i] * u[j]
                    for j in range(3)
                )
                for i in range(3)
            )

    # A non-right-angle float rotation is stored as the exact rational value
    # of its binary-float trigonometric matrix.  This keeps later ray algebra
    # exact for that explicitly supplied floating-point transform.
    axis_float = tuple(float(v) for v in axis_value)
    length = math.sqrt(sum(v * v for v in axis_float))
    u = tuple(v / length for v in axis_float)
    c_float = math.cos(angle_float)
    s_float = math.sin(angle_float)
    values = []
    axis_squared = sum(v * v for v in axis_float)
    for i in range(3):
        row = []
        for j in range(3):
            cross = (
                u[(i + 1) % 3] * u[(j + 2) % 3]
                - u[(i + 2) % 3] * u[(j + 1) % 3]
            )
            row.append(Fraction(c_float * (1.0 if i == j else 0.0)
                                + s_float * cross
                                + (1.0 - c_float) * axis_float[i] * axis_float[j]
                                / axis_squared))
        values.append(tuple(row))
    return _as_matrix3(values, path + ".matrix")


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
    """Invertible affine transform ``p_world = M p_local + translation``."""

    __slots__ = ("child", "matrix", "translation")

    def __init__(self, child, matrix=None, translation=None):
        path = "Transform"
        if not _is_node(child):
            raise GeometryError(
                f"{path}.child: not a CSG node ({type(child).__name__})"
            )
        if matrix is None:
            matrix = _identity3()
        else:
            matrix = _as_matrix3(matrix, f"{path}.matrix")
        if translation is None:
            translation = (Fraction(0),) * 3
        else:
            translation = _as_vec3(translation, f"{path}.translation")
        self.child = child
        self.matrix = matrix
        self.translation = translation

    def __repr__(self):
        return (
            f"Transform({self.child!r}, matrix={self.matrix!r}, "
            f"translation={self.translation!r})"
        )

    def __eq__(self, other):
        return (
            isinstance(other, Transform)
            and self.child == other.child
            and self.matrix == other.matrix
            and self.translation == other.translation
        )


class Translate(Transform):
    """Translate a child subtree by ``offset``."""

    def __init__(self, child, offset):
        super().__init__(
            child,
            translation=_as_vec3(offset, "Translate.offset"),
        )


class Rotate(Transform):
    """Right-handed rotation by ``angle`` radians around ``axis``."""

    def __init__(self, child, angle, axis, center=None):
        path = "Rotate"
        rotation = _rotation_matrix(angle, axis, path)
        if center is None:
            translation = (Fraction(0),) * 3
        else:
            center_value = _as_vec3(center, f"{path}.center")
            translation = tuple(
                center_value[i]
                - sum(rotation[i][j] * center_value[j] for j in range(3))
                for i in range(3)
            )
        super().__init__(child, matrix=rotation, translation=translation)


class Scale(Transform):
    """Independent per-axis scaling; every factor must be non-zero."""

    def __init__(self, child, factors, center=None):
        path = "Scale"
        if isinstance(factors, (int, float)) and not isinstance(factors, bool):
            factor_value = _as_number(factors, f"{path}.factors")
            factors_value = (factor_value, factor_value, factor_value)
        else:
            factors_value = _as_vec3(factors, f"{path}.factors")
        for i, factor in enumerate(factors_value):
            if factor == 0:
                raise GeometryError(
                    f"{path}.factors[{i}]: scale factor must be non-zero"
                )
        matrix = tuple(
            tuple(factors_value[i] if i == j else Fraction(0) for j in range(3))
            for i in range(3)
        )
        if center is None:
            translation = (Fraction(0),) * 3
        else:
            center_value = _as_vec3(center, f"{path}.center")
            translation = tuple(
                center_value[i] - matrix[i][i] * center_value[i]
                for i in range(3)
            )
        super().__init__(child, matrix=matrix, translation=translation)


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
