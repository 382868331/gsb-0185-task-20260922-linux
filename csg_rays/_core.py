"""Core sweep algorithm: CSG ray-interval intersection with cavities.

Pipeline of :func:`ray_intervals`:

1. Validate the tree (node budget, no shared nodes, unique leaf ids) and
   record each leaf's polarity (flipped under an odd number of Difference
   subtractor positions).
2. Turn every leaf into events on the ray parameter t: ``enter``/``exit``
   (solid crossings), ``touch`` (isolated tangential contact) and
   ``tstart``/``tend`` (zero-thickness face-riding contact of a ray lying on
   a box face).  All event parameters are exact (Fraction or quadratic Root).
3. Cluster coincident event parameters exactly, then sweep the t-line with a
   three-valued state (IN / T / OUT) per leaf and per Boolean node.  Events
   at the same t are applied as one batch, so coincident events that cause
   no state change never create phantom boundaries.
4. Regularize: maximal T-runs flanked by solid on both sides are swallowed
   (a tangent touch does not puncture a solid); T-runs flanked by empty on
   both sides become zero-measure touches; the rest become ordinary
   boundaries.  Isolated empty points between solid segments are swallowed.
5. Emit sorted disjoint open intervals, touches, all boundaries (with full
   source lists and a lexicographic representative) and the nearest boundary
   with t > 0.
"""

from __future__ import annotations

import math
from decimal import localcontext
from fractions import Fraction

from ._exact import Root, cmp_tval, frac_to_decimal, sorted_unique, tval_to_float
from ._types import (
    MAX_TREE_NODES,
    ORIENT_BACK,
    ORIENT_FRONT,
    ORIENT_GRAZING,
    Boundary,
    Box,
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
    _is_node,
)

_IN = "IN"
_T = "T"
_OUT = "OUT"

_SOLID = "SOLID"
_EMPTY = "EMPTY"
_TOUCH = "TOUCH"

_BOX_NORMALS = {
    "x_lo": (-1.0, 0.0, 0.0),
    "x_hi": (1.0, 0.0, 0.0),
    "y_lo": (0.0, -1.0, 0.0),
    "y_hi": (0.0, 1.0, 0.0),
    "z_lo": (0.0, 0.0, -1.0),
    "z_hi": (0.0, 0.0, 1.0),
}


# ---------------------------------------------------------------------------
# tree validation
# ---------------------------------------------------------------------------

def _validate_and_collect(tree):
    """Check tree structure; return (leaves, polarities) in traversal order."""
    leaves = []
    polarities = []
    seen_nodes = set()
    leaf_paths = {}
    count = 0

    def walk(node, polarity, path):
        nonlocal count
        count += 1
        if count > MAX_TREE_NODES:
            raise GeometryError(
                f"{path}: CSG tree exceeds the limit of {MAX_TREE_NODES} nodes"
            )
        if id(node) in seen_nodes:
            raise GeometryError(
                f"{path}: node {node!r} is shared/reused; the CSG tree must be "
                f"a proper tree (no node may appear twice)"
            )
        seen_nodes.add(id(node))
        if isinstance(node, (Sphere, Box)):
            if node.leaf_id in leaf_paths:
                raise GeometryError(
                    f"{path}: duplicate leaf_id {node.leaf_id!r} "
                    f"(first used at {leaf_paths[node.leaf_id]})"
                )
            leaf_paths[node.leaf_id] = path
            leaves.append(node)
            polarities.append(polarity)
        elif isinstance(node, (Union, Intersection)):
            for i, child in enumerate(node.children):
                walk(child, polarity, f"{path}.children[{i}]")
        elif isinstance(node, Difference):
            walk(node.positive, polarity, f"{path}.positive")
            for i, sub in enumerate(node.subtractors):
                walk(sub, -polarity, f"{path}.subtractors[{i}]")
        else:
            raise GeometryError(
                f"{path}: not a CSG node ({type(node).__name__})"
            )

    walk(tree, 1, "tree")
    return leaves, polarities


# ---------------------------------------------------------------------------
# leaf events
# ---------------------------------------------------------------------------

def _sqrt_fraction(value):
    """Exact square root of a non-negative Fraction, or None if irrational."""
    num_root = math.isqrt(value.numerator)
    den_root = math.isqrt(value.denominator)
    if num_root * num_root == value.numerator and den_root * den_root == value.denominator:
        return Fraction(num_root, den_root)
    return None


def _sphere_events(sphere, origin, direction):
    """Events of a sphere leaf: enter/exit crossings or one tangent touch."""
    oc = [origin[i] - sphere.center[i] for i in range(3)]
    a = sum(component * component for component in direction)
    b = 2 * sum(oc[i] * direction[i] for i in range(3))
    c = sum(component * component for component in oc) - sphere.radius ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    if disc == 0:
        # tangent ray: zero-measure touch, not solid interior
        return [(-b / (2 * a), "touch", ("sphere",))]
    sqrt_disc = _sqrt_fraction(disc)
    if sqrt_disc is not None:
        t_lo = (-b - sqrt_disc) / (2 * a)
        t_hi = (-b + sqrt_disc) / (2 * a)
    else:
        t_lo = Root(a, b, c, -1)
        t_hi = Root(a, b, c, +1)
    return [(t_lo, "enter", ("sphere",)), (t_hi, "exit", ("sphere",))]


def _box_events(box, origin, direction):
    """Events of an axis-aligned box leaf.

    Generic rays produce enter/exit crossings (corner/edge hits list every
    coincident face).  A ray lying on a face plane never enters the interior:
    it produces a zero-thickness tstart/tend contact run (or one isolated
    touch when the overlap degenerates to a point).
    """
    enter_t = None
    enter_faces = []
    exit_t = None
    exit_faces = []
    riding_faces = []
    crossing = False

    for i in range(3):
        axis = "xyz"[i]
        o_i = origin[i]
        d_i = direction[i]
        lo = box.mins[i]
        hi = box.maxs[i]
        face_lo = f"{axis}_lo"
        face_hi = f"{axis}_hi"
        if d_i == 0:
            if o_i < lo or o_i > hi:
                return []  # parallel and outside the slab: miss
            if o_i == lo:
                riding_faces.append(face_lo)
            elif o_i == hi:
                riding_faces.append(face_hi)
            continue
        crossing = True
        t_lo = (lo - o_i) / d_i
        t_hi = (hi - o_i) / d_i
        if d_i > 0:
            enter_face, exit_face = face_lo, face_hi
        else:
            t_lo, t_hi = t_hi, t_lo
            enter_face, exit_face = face_hi, face_lo
        if enter_t is None or t_lo > enter_t:
            enter_t = t_lo
            enter_faces = [enter_face]
        elif t_lo == enter_t:
            enter_faces.append(enter_face)
        if exit_t is None or t_hi < exit_t:
            exit_t = t_hi
            exit_faces = [exit_face]
        elif t_hi == exit_t:
            exit_faces.append(exit_face)

    if riding_faces:
        # The ray lies on a face plane: no solid interior at all, only a
        # zero-thickness contact run over the overlap of the crossing axes.
        if not crossing:
            raise GeometryError(  # unreachable: direction is non-zero
                "Box: ray direction is non-zero but no axis is crossed"
            )
        if enter_t > exit_t:
            return []  # passes the face outside the box outline
        if enter_t == exit_t:
            faces = tuple(enter_faces + exit_faces + riding_faces)
            return [(enter_t, "touch", faces)]
        return [
            (enter_t, "tstart", tuple(enter_faces + riding_faces)),
            (exit_t, "tend", tuple(exit_faces + riding_faces)),
        ]

    if not crossing:
        raise GeometryError(  # unreachable: direction is non-zero
            "Box: ray direction is non-zero but no axis is crossed"
        )
    if enter_t > exit_t:
        return []  # slab intervals do not overlap: miss
    if enter_t == exit_t:
        # grazing an edge or corner: isolated zero-measure touch
        return [(enter_t, "touch", tuple(enter_faces + exit_faces))]
    return [(enter_t, "enter", tuple(enter_faces)), (exit_t, "exit", tuple(exit_faces))]


# ---------------------------------------------------------------------------
# three-valued Boolean evaluation
# ---------------------------------------------------------------------------

def _sub_state(positive, negative):
    """Regularized difference of two leaf states (subtractor is closed)."""
    if negative == _IN:
        return _OUT
    if negative == _OUT:
        return positive
    return _OUT if positive == _OUT else _T


def _make_evaluator(tree, leaf_index):
    def evaluate(node, leaf_state):
        if isinstance(node, (Sphere, Box)):
            return leaf_state(leaf_index[id(node)])
        if isinstance(node, Union):
            saw_touch = False
            for child in node.children:
                state = evaluate(child, leaf_state)
                if state == _IN:
                    return _IN
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _OUT
        if isinstance(node, Intersection):
            saw_touch = False
            for child in node.children:
                state = evaluate(child, leaf_state)
                if state == _OUT:
                    return _OUT
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _IN
        if isinstance(node, Difference):
            result = evaluate(node.positive, leaf_state)
            for sub in node.subtractors:
                result = _sub_state(result, evaluate(sub, leaf_state))
            return result
        raise GeometryError(f"tree: not a CSG node ({type(node).__name__})")

    return evaluate


# ---------------------------------------------------------------------------
# normals
# ---------------------------------------------------------------------------

def _sphere_normal(sphere, origin, direction, tval):
    """Outward unit normal at the hit point (exact when tval is rational)."""
    if isinstance(tval, Fraction):
        normal = [
            (origin[i] + tval * direction[i] - sphere.center[i]) / sphere.radius
            for i in range(3)
        ]
        return tuple(float(component) for component in normal)
    with localcontext() as ctx:
        ctx.prec = 70
        t_dec = tval.decimal(70)
        normal = [
            (
                frac_to_decimal(origin[i])
                + t_dec * frac_to_decimal(direction[i])
                - frac_to_decimal(sphere.center[i])
            )
            / frac_to_decimal(sphere.radius)
            for i in range(3)
        ]
        return tuple(float(component) for component in normal)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def ray_intervals(tree, ray):
    """Intersect a ray with a CSG tree; return a RayIntervalsResult.

    See module docstring and README.md for the full contract.  Raises
    GeometryError for invalid input and CertificationBudgetExhausted when
    event ordering exceeds the precision budget.
    """
    if not isinstance(ray, Ray):
        raise GeometryError(f"ray: expected a Ray, got {type(ray).__name__}")
    if not _is_node(tree):
        raise GeometryError(f"tree: not a CSG node ({type(tree).__name__})")
    origin = ray.origin
    direction = ray.direction

    leaves, polarities = _validate_and_collect(tree)
    leaf_index = {id(leaf): i for i, leaf in enumerate(leaves)}
    evaluate = _make_evaluator(tree, leaf_index)

    # -- per-leaf events -------------------------------------------------
    leaf_events = []
    for leaf in leaves:
        if isinstance(leaf, Sphere):
            leaf_events.append(_sphere_events(leaf, origin, direction))
        else:
            leaf_events.append(_box_events(leaf, origin, direction))

    # -- exact clustering of coincident event parameters -----------------
    all_tvals = [tval for events in leaf_events for (tval, _kind, _faces) in events]
    clusters = sorted_unique(all_tvals)
    n_clusters = len(clusters)

    def cluster_index(tval):
        lo, hi = 0, n_clusters
        while lo < hi:
            mid = (lo + hi) // 2
            order = cmp_tval(clusters[mid], tval)
            if order == 0:
                return mid
            if order < 0:
                lo = mid + 1
            else:
                hi = mid
        raise AssertionError("event parameter missing from its own clusters")

    leaf_events_at = [dict() for _ in leaves]
    for li, events in enumerate(leaf_events):
        for tval, kind, faces in events:
            leaf_events_at[li].setdefault(cluster_index(tval), []).append((kind, faces))

    # -- per-leaf segment / point states ----------------------------------
    # Segment j is the open interval between cluster j-1 and cluster j
    # (segment 0 lies before the first cluster, segment n after the last).
    leaf_seg = [[_OUT] * (n_clusters + 1) for _ in leaves]
    leaf_pt = [[_OUT] * n_clusters for _ in leaves]
    for li in range(len(leaves)):
        depth = 0
        riding = 0
        for ci in range(n_clusters):
            events = leaf_events_at[li].get(ci, [])
            kinds = {kind for kind, _faces in events}
            if "enter" in kinds or "exit" in kinds:
                leaf_pt[li][ci] = _OUT  # crossing boundary point
            elif events:
                leaf_pt[li][ci] = _T  # tangential / face-riding contact point
            else:
                leaf_pt[li][ci] = _IN if depth > 0 else (_T if riding > 0 else _OUT)
            for kind, _faces in events:
                if kind == "enter":
                    depth += 1
                elif kind == "exit":
                    depth -= 1
                elif kind == "tstart":
                    riding += 1
                elif kind == "tend":
                    riding -= 1
            leaf_seg[li][ci + 1] = _IN if depth > 0 else (_T if riding > 0 else _OUT)

    def seg_state(j, forced=None):
        def leaf_state(li):
            if forced is not None and forced[0] == li:
                return forced[1]
            return leaf_seg[li][j]

        return evaluate(tree, leaf_state)

    def pt_state(i, forced=None):
        def leaf_state(li):
            if forced is not None and forced[0] == li:
                return forced[1]
            return leaf_pt[li][i]

        return evaluate(tree, leaf_state)

    result_seg = [seg_state(j) for j in range(n_clusters + 1)]
    result_pt = [pt_state(i) for i in range(n_clusters)]

    # -- cell relabeling: regularize T-runs --------------------------------
    final_seg = [
        _SOLID if s == _IN else (_EMPTY if s == _OUT else _T) for s in result_seg
    ]
    final_pt = [
        _SOLID if s == _IN else (_EMPTY if s == _OUT else _T) for s in result_pt
    ]

    def cell_state(k):
        return final_seg[k // 2] if k % 2 == 0 else final_pt[k // 2]

    def set_cell(k, value):
        if k % 2 == 0:
            final_seg[k // 2] = value
        else:
            final_pt[k // 2] = value

    touch_runs = []  # (left_cluster_index, right_cluster_index)
    k = 0
    last_cell = 2 * n_clusters
    while k <= last_cell:
        if cell_state(k) != _T:
            k += 1
            continue
        start = k
        while k + 1 <= last_cell and cell_state(k + 1) == _T:
            k += 1
        end = k
        left_state = cell_state(start - 1) if start > 0 else _EMPTY
        right_state = cell_state(end + 1) if end < last_cell else _EMPTY
        if left_state == _SOLID and right_state == _SOLID:
            # tangent contact inside solid: swallowed, no phantom boundary
            for k2 in range(start, end + 1):
                set_cell(k2, _SOLID)
        elif left_state == _SOLID or right_state == _SOLID:
            # one side solid: the run collapses into an ordinary boundary
            for k2 in range(start, end + 1):
                set_cell(k2, _EMPTY)
        else:
            for k2 in range(start, end + 1):
                set_cell(k2, _TOUCH)
            left_cluster = start // 2 - 1 if start % 2 == 0 else start // 2
            right_cluster = end // 2
            touch_runs.append((left_cluster, right_cluster))
        k += 1

    # swallow isolated empty points between solid segments (zero-width gaps)
    for i in range(n_clusters):
        if (
            final_pt[i] == _EMPTY
            and final_seg[i] == _SOLID
            and final_seg[i + 1] == _SOLID
        ):
            final_pt[i] = _SOLID

    # -- source collection --------------------------------------------------
    def sources_at(ci):
        """All leaf faces at cluster ci whose state matters to the result."""
        found = []
        for li, leaf in enumerate(leaves):
            events = leaf_events_at[li].get(ci)
            if not events:
                continue
            pair_in = (seg_state(ci, (li, _IN)), seg_state(ci + 1, (li, _IN)))
            pair_out = (seg_state(ci, (li, _OUT)), seg_state(ci + 1, (li, _OUT)))
            if pair_in == pair_out:
                continue  # coincident event with no effect: not a source
            for kind, faces in events:
                source_kind = kind if kind in ("enter", "exit") else "touch"
                for face in faces:
                    found.append(
                        _make_source(
                            leaf, face, source_kind, clusters[ci],
                            origin, direction, polarities[li],
                        )
                    )
        found.sort(key=lambda s: (s.leaf_id, s.face))
        return found

    # -- boundaries ---------------------------------------------------------
    boundary_by_cluster = {}
    ordered_boundaries = []
    for ci in range(n_clusters):
        left = final_seg[ci]
        right = final_seg[ci + 1]
        if left == _EMPTY and right == _SOLID:
            kind = "enter"
        elif left == _SOLID and right == _EMPTY:
            kind = "exit"
        else:
            continue
        sources = sources_at(ci)
        if not sources:
            raise GeometryError(  # unreachable: a transition needs a cause
                f"internal error: boundary at cluster {ci} has no source"
            )
        boundary = Boundary(
            t=tval_to_float(clusters[ci]),
            kind=kind,
            source=sources[0],
            sources=tuple(sources),
        )
        boundary_by_cluster[ci] = boundary
        ordered_boundaries.append(boundary)

    # -- intervals ----------------------------------------------------------
    intervals = []
    j = 0
    while j <= n_clusters:
        if final_seg[j] != _SOLID:
            j += 1
            continue
        j2 = j
        while j2 + 1 <= n_clusters and final_seg[j2 + 1] == _SOLID:
            j2 += 1
        start_boundary = boundary_by_cluster[j - 1]
        end_boundary = boundary_by_cluster[j2]
        intervals.append(
            Interval(
                t0=start_boundary.t,
                t1=end_boundary.t,
                start=start_boundary,
                end=end_boundary,
            )
        )
        j = j2 + 1

    # -- touches --------------------------------------------------------------
    touches = []
    for left_cluster, right_cluster in touch_runs:
        merged = []
        for ci in range(left_cluster, right_cluster + 1):
            merged.extend(sources_at(ci))
        merged.sort(key=lambda s: (s.leaf_id, s.face, s.kind))
        unique = []
        seen = set()
        for source in merged:
            key = (source.leaf_id, source.face, source.kind)
            if key not in seen:
                seen.add(key)
                unique.append(source)
        touches.append(
            Touch(
                t0=tval_to_float(clusters[left_cluster]),
                t1=tval_to_float(clusters[right_cluster]),
                source=unique[0],
                sources=tuple(unique),
            )
        )

    # -- nearest forward boundary --------------------------------------------
    nearest = None
    zero = Fraction(0)
    for ci, boundary in boundary_by_cluster.items():
        if cmp_tval(clusters[ci], zero) <= 0:
            continue
        if nearest is None or cmp_tval(clusters[ci], clusters[nearest]) < 0:
            nearest = ci

    return RayIntervalsResult(
        intervals=tuple(intervals),
        touches=tuple(touches),
        boundaries=tuple(ordered_boundaries),
        nearest_boundary=boundary_by_cluster[nearest] if nearest is not None else None,
    )


def _face_orientation(leaf, face, kind, tval, origin, direction):
    """Classify a face against the ray direction from the leaf's own outward
    normal (before any Difference flip), using exact arithmetic.

    Returns one of ORIENT_FRONT / ORIENT_BACK / ORIENT_GRAZING according to
    the sign of the outward-normal . ray-direction product.
    """
    if isinstance(leaf, Box):
        axis = "xyz".index(face[0])
        sign = -1 if face.endswith("_lo") else +1
        dot = sign * direction[axis]  # axis-aligned unit normal . direction
    else:  # Sphere
        if kind == "touch":
            # A sphere touch is the tangent point; the tangent normal is
            # exactly perpendicular to the ray.
            return ORIENT_GRAZING
        if isinstance(tval, Fraction):
            # Outward (not necessarily unit) radial vector at the hit point.
            dot = Fraction(0)
            for i in range(3):
                radial_i = origin[i] + tval * direction[i] - leaf.center[i]
                dot += radial_i * direction[i]
        else:
            # Irrational quadratic root: a transversal crossing is never
            # grazing; the smaller root is the entry face, larger the exit.
            return ORIENT_FRONT if kind == "enter" else ORIENT_BACK
    if dot < 0:
        return ORIENT_FRONT
    if dot > 0:
        return ORIENT_BACK
    return ORIENT_GRAZING


def _make_source(leaf, face, kind, tval, origin, direction, polarity):
    if isinstance(leaf, Sphere):
        normal = _sphere_normal(leaf, origin, direction, tval)
    else:
        normal = _BOX_NORMALS[face]
    flipped = polarity < 0
    if flipped:
        normal = tuple(-component for component in normal)
    orientation = _face_orientation(
        leaf, face, kind, tval, origin, direction
    )
    return Source(
        leaf_id=leaf.leaf_id,
        face=face,
        kind=kind,
        # "+ 0.0" normalizes -0.0 produced by flipping zero components
        normal=tuple(float(component) + 0.0 for component in normal),
        flipped=flipped,
        material=leaf.material,
        orientation=orientation,
    )
