"""csg_rays: CSG ray-interval intersection with cavities (offline, stdlib only).

Public interface:

* Tree nodes: :class:`Sphere`, :class:`Box` (leaves); :class:`Union`,
  :class:`Intersection`, :class:`Difference` (Boolean nodes) and nestable
  affine transforms :class:`Transform`, :class:`Translate`, :class:`Scale`,
  :class:`Rotate`.
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
    Rotate,
    Scale,
    Source,
    Sphere,
    Touch,
    Transform,
    Translate,
    Union,
)

__all__ = [
    "MAX_TREE_NODES",
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
    "Rotate",
    "Scale",
    "Source",
    "Sphere",
    "Touch",
    "Transform",
    "Translate",
    "Union",
    "ray_intervals",
]

__version__ = "1.0.0"
