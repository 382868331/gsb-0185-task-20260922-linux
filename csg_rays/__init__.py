"""csg_rays: CSG ray-interval intersection with cavities (offline, stdlib only).

Public interface:

* Tree nodes: :class:`Sphere`, :class:`Box` (leaves), :class:`Union`,
  :class:`Intersection`, :class:`Difference` (internal nodes) and
  :class:`Instance` (a reference to an existing subtree under an invertible
  affine transform; sharing without copying, no reference cycles).
* :class:`Ray` and the entry point :func:`ray_intervals`.
* Output: :class:`RayIntervalsResult` with :class:`Interval`, :class:`Touch`,
  :class:`Boundary` and :class:`Source` (a source is qualified by its instance
  ``path`` plus the original leaf id).
* Errors: :class:`CSGError` (base), :class:`GeometryError` (invalid input),
  :class:`CertificationBudgetExhausted` (precision budget exceeded; distinct
  from a legitimately empty result).

See README.md for the full contract, limits and error policy.
"""

from ._core import ray_intervals
from ._types import (
    MAX_LEAF_OCCURRENCES,
    MAX_TREE_NODES,
    Boundary,
    Box,
    CertificationBudgetExhausted,
    CSGError,
    Difference,
    GeometryError,
    Instance,
    Intersection,
    Interval,
    Ray,
    RayIntervalsResult,
    Source,
    Sphere,
    Touch,
    Union,
)

__all__ = [
    "MAX_LEAF_OCCURRENCES",
    "MAX_TREE_NODES",
    "Boundary",
    "Box",
    "CertificationBudgetExhausted",
    "CSGError",
    "Difference",
    "GeometryError",
    "Instance",
    "Intersection",
    "Interval",
    "Ray",
    "RayIntervalsResult",
    "Source",
    "Sphere",
    "Touch",
    "Union",
    "ray_intervals",
]

__version__ = "1.1.0"
