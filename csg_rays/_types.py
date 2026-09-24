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
from types import MappingProxyType

#: Maximum number of nodes (leaves + internal nodes) accepted in a CSG tree.
MAX_TREE_NODES = 100

#: Material id assigned to a primitive surface when the caller overrides none.
DEFAULT_MATERIAL = "default"

#: The six axis-aligned box surface names (valid ``face_materials`` keys).
BOX_FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")


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


def _as_material_id(value, path):
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise GeometryError(f"{path}: material id must be a non-empty str")
    return value


def _resolve_materials(material, face_materials, faces, path):
    """Build the immutable ``{face: material_id}`` map of a primitive.

    ``material`` is the primitive-wide default; ``face_materials`` optionally
    overrides individual named faces.  Every surface (one face for a sphere,
    six for a box) ends up with a non-empty string material id.  Unknown face
    keys are rejected rather than silently ignored, so a typo can never leave
    a surface half-configured.
    """
    default = _as_material_id(material, f"{path}.material")
    if face_materials is None:
        overrides = {}
    else:
        if isinstance(face_materials, (str, bytes)) or not hasattr(
            face_materials, "items"
        ):
            raise GeometryError(
                f"{path}.face_materials: expected a mapping of face names to "
                f"material ids, got {type(face_materials).__name__}"
            )
        overrides = dict(face_materials)
    allowed = set(faces)
    for face in overrides:
        if face not in allowed:
            raise GeometryError(
                f"{path}.face_materials: unknown face {face!r}; "
                f"valid faces are {', '.join(faces)}"
            )
    resolved = {face: default for face in faces}
    for face, mat in overrides.items():
        resolved[face] = _as_material_id(mat, f"{path}.face_materials[{face!r}]")
    # a sorted tuple of pairs: value-based and hashable, so frozen primitive
    # nodes stay comparable and usable as dict/set members even with overrides
    stored = tuple(sorted(overrides.items())) or None
    return MappingProxyType(resolved), stored


def _is_node(value):
    return isinstance(value, (Sphere, Box, Union, Intersection, Difference))


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
    """Solid sphere leaf.  ``center``/``radius`` are stored as exact Fractions.

    ``material`` is the material id of the sphere surface (default
    :data:`DEFAULT_MATERIAL`).  ``face_materials`` is an optional mapping and
    only accepts the key ``"sphere"``; it is kept for interface symmetry with
    :class:`Box`.  The resolved per-face map is exposed as ``materials``.
    """

    center: object
    radius: object
    leaf_id: str
    material: str = DEFAULT_MATERIAL
    face_materials: object = None

    def __post_init__(self):
        path = f"Sphere(leaf_id={self.leaf_id!r})"
        object.__setattr__(self, "leaf_id", _as_leaf_id(self.leaf_id, path))
        object.__setattr__(self, "center", _as_vec3(self.center, f"{path}.center"))
        radius = _as_number(self.radius, f"{path}.radius")
        if radius <= 0:
            raise GeometryError(f"{path}.radius: must be > 0, got {self.radius!r}")
        object.__setattr__(self, "radius", radius)
        materials, overrides = _resolve_materials(
            self.material, self.face_materials, ("sphere",), path
        )
        object.__setattr__(self, "face_materials", overrides)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(
            self, "material", _as_material_id(self.material, f"{path}.material")
        )


@dataclass(frozen=True)
class Box:
    """Axis-aligned solid box leaf.  ``mins[i] < maxs[i]`` is required.

    ``material`` is the material id used for every face unless overridden via
    ``face_materials``, a mapping from face names (``"x_lo"`` ... ``"z_hi"``)
    to material ids.  The resolved per-face map is exposed as ``materials``.
    """

    mins: object
    maxs: object
    leaf_id: str
    material: str = DEFAULT_MATERIAL
    face_materials: object = None

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
        materials, overrides = _resolve_materials(
            self.material, self.face_materials, BOX_FACES, path
        )
        object.__setattr__(self, "face_materials", overrides)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "material", _as_material_id(self.material, f"{path}.material"))


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
    ``"y_lo"``, ``"y_hi"``, ``"z_lo"``, ``"z_hi"`` for boxes.  ``material`` is
    the material id assigned to that primitive surface (default
    :data:`DEFAULT_MATERIAL`; per-face overrides come from
    ``Sphere/Box(..., face_materials=...)``).

    ``kind`` is the leaf-side event kind: ``"enter"``, ``"exit"`` or
    ``"touch"`` (tangential / face-riding contact).

    ``normal`` is the geometric outward unit normal of that face at the hit
    point, flipped when the leaf acts as a subtractor inside
    :class:`Difference` (``flipped`` is then True).

    ``orientation`` records the direction the surface was authored with and
    never changes under Boolean composition: ``"outward"`` for a positive
    primitive, ``"inward"`` for a surface reached through an odd number of
    subtractor positions (the cavity wall of a difference).  Nested
    differences toggle it again: the positive of a difference nested inside a
    subtractor is authored inward, and its own subtractors toggle back to
    outward.  Note the distinction: ``orientation`` is the authored facing
    (parity of subtractor nesting), while ``flipped`` says whether the
    geometric ``normal`` was negated.  For ordinary boundaries they coincide;
    at a tangent contact on an inward-authored subtractor face the surface
    keeps its inward orientation but the reported normal is the outward one
    (the contact does not become a cavity wall), so ``flipped`` is False while
    ``orientation == "inward"``.
    """

    leaf_id: str
    face: str
    kind: str
    normal: tuple
    flipped: bool
    material: str = DEFAULT_MATERIAL
    orientation: str = "outward"


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
