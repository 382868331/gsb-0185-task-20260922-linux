"""csg_rays: CSG ray-interval intersection with cavities (offline, stdlib only).

Public interface:

* Tree nodes: :class:`Sphere`, :class:`Box` (leaves) and :class:`Union`,
  :class:`Intersection`, :class:`Difference` (internal nodes).
* :class:`Ray` and the entry point :func:`ray_intervals`.
* Output: :class:`RayIntervalsResult` with :class:`Interval`, :class:`Touch`,
  :class:`Boundary` and :class:`Source`.
* Errors: :class:`CSGError` (base), :class:`GeometryError` (invalid input),
  :class:`CertificationBudgetExhausted` (precision budget exceeded; distinct
  from a legitimately empty result).

See README.md for the full contract, limits and error policy.
"""

from ._core import ray_intervals
from ._types import (
    MAX_TREE_NODES,
    ORIENT_BACK,
    ORIENT_FRONT,
    ORIENT_GRAZING,
    Boundary,
    Box,
    CertificationBudgetExhausted,
    CSGError,
    Difference,
    GeometryError,
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
    "MAX_TREE_NODES",
    "ORIENT_BACK",
    "ORIENT_FRONT",
    "ORIENT_GRAZING",
    "Boundary",
    "Box",
    "CertificationBudgetExhausted",
    "CSGError",
    "Difference",
    "GeometryError",
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
