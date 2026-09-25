"""Core sweep algorithm: CSG ray-interval intersection with cavities.

Pipeline of :func:`ray_intervals`:

1. Validate the tree and *expand* Instance references into effective leaf
   occurrences (leaf node, composed inverse affine, difference polarity,
   instance path).  Structural sharing inside one expansion root and
   reference cycles are rejected; the 100-node budget counts occurrences.
2. Map the world ray into each occurrence's local frame with the composed
   inverse affine (the local direction is NOT renormalized, so event t
   stays the world-ray parameter), and turn every leaf into events on t:
   ``enter``/``exit`` (solid crossings), ``touch`` (isolated tangential
   contact) and ``tstart``/``tend`` (zero-thickness face-riding contact of
   a ray lying on a box face).  All event parameters are exact (Fraction
   or quadratic Root).
3. Cluster coincident event parameters exactly, then sweep the t-line with
   a three-valued state (IN / T / OUT) per leaf occurrence and per Boolean
   node.  Events at the same t are applied as one batch, so coincident
   events that cause no state change never create phantom boundaries.
4. Regularize: maximal T-runs flanked by solid on both sides are swallowed
   (a tangent touch does not puncture a solid); T-runs flanked by empty on
   both sides become zero-measure touches; the rest become ordinary
   boundaries.  Isolated empty points between solid segments are swallowed.
5. Map local normals to world with the inverse transpose B^T plus
   renormalization (Difference polarity flips independently, at the Boolean
   level), and emit sorted disjoint open intervals, touches, all boundaries
   (with full path-qualified source lists and a lexicographic
   representative) and the nearest boundary with t > 0.
"""

from __future__ import annotations

import math
from decimal import localcontext
from fractions import Fraction

from ._exact import Root, cmp_tval, frac_to_decimal, sorted_unique, tval_to_float
from ._types import (
    MAX_TREE_NODES,
    Boundary,
    Box,
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
    _is_node,
)

_IN = "IN"
_T = "T"
_OUT = "OUT"

_SOLID = "SOLID"
_EMPTY = "EMPTY"
_TOUCH = "TOUCH"


# ---------------------------------------------------------------------------
# tree expansion: instances unfold into effective leaves
# ---------------------------------------------------------------------------
#
# An Instance references an existing CSG subtree under an invertible affine
# transform x_world = A x_local + t.  The tree is *expanded*: every leaf
# occurrence reached through k instance references becomes one effective
# leaf carrying
#
#   * the leaf node itself (shared node object, never modified),
#   * the composed inverse transform (B | v) mapping the world ray to the
#     leaf's local ray,
#   * the difference polarity along the occurrence path,
#   * the instance path (tuple of instance_ids) identifying the reference.
#
# The shared node objects are reused read-only, but all sweep state is keyed
# by occurrence, so a transform only affects its own reference.

def _identity_affine():
    b = [
        [Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]
    return b, [Fraction(0), Fraction(0), Fraction(0)]


def _compose_affine(outer, inner_inv):
    """Compose inverse transforms for a nested instance.

    If the outer instance maps by ``A1`` (inverse ``B1 x + v1``) and the
    newly entered instance maps locally by ``A2`` (inverse ``B2 x + v2``),
    the world->deep-local map is ``B2 (B1 x + v1) + v2``.
    """
    b1, v1 = outer
    b2, v2 = inner_inv
    b = [
        [sum(b2[i][k] * b1[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    v = [v2[i] + sum(b2[i][k] * v1[k] for k in range(3)) for i in range(3)]
    return b, v


def _validate_and_expand(tree):
    """Validate the tree and unfold instances into effective leaves.

    Returns ``(occurrences, expr)`` where each occurrence is a
    ``(leaf, inv_matrix_rows, inv_translation, polarity, instance_path)``
    tuple and ``expr`` is the Boolean expression tree evaluated with one
    state per leaf occurrence (shared leaves get one state per reference).

    Structural rules enforced here:

    * every traversal rooted at the main tree or at one referenced subtree
      must be a proper tree -- a node appearing twice inside one such
      traversal (sibling sharing or a diamond) is rejected;
    * reference cycles (a referenced subtree that references one of its own
      ancestors, including the main tree) are rejected via the active-path
      set, which spans instance boundaries;
    * a node referenced by an Instance must not also be placed directly in
      the same traversal (shared geometry lives behind references only);
    * ``instance_id``s are globally unique, so every instance path is a
      unique identity and ``(path, leaf_id)`` uniquely fixes one occurrence.
    """
    occurrences = []
    instance_paths = set()  # full instance paths already entered
    count = 0

    def walk(node, polarity, path, affine, instance_path, seen, active, leaf_ids):
        nonlocal count
        count += 1
        if count > MAX_TREE_NODES:
            raise GeometryError(
                f"{path}: CSG tree exceeds the limit of {MAX_TREE_NODES} nodes "
                f"(instance nodes and referenced occurrences are counted)"
            )
        if not _is_node(node):
            raise GeometryError(f"{path}: not a CSG node ({type(node).__name__})")
        nid = id(node)
        if nid in active:
            raise GeometryError(
                f"{path}: reference cycle or reused node {node!r}: the node is "
                f"one of its own ancestors (reference cycles are forbidden)"
            )
        if nid in seen:
            raise GeometryError(
                f"{path}: node {node!r} appears more than once in the same "
                f"(sub)tree; place shared geometry behind separate Instance "
                f"references"
            )
        seen.add(nid)
        active.add(nid)
        try:
            if isinstance(node, (Sphere, Box)):
                if node.leaf_id in leaf_ids:
                    raise GeometryError(
                        f"{path}: duplicate leaf_id {node.leaf_id!r} within one "
                        f"(sub)tree (shared leaf ids across instances are "
                        f"allowed; distinguish them via the instance path)"
                    )
                leaf_ids.add(node.leaf_id)
                occurrences.append(
                    (node, affine[0], affine[1], polarity, instance_path)
                )
                return ("leaf", len(occurrences) - 1)

            if isinstance(node, (Union, Intersection)):
                children = [
                    walk(
                        child, polarity, f"{path}.children[{i}]",
                        affine, instance_path, seen, active, leaf_ids,
                    )
                    for i, child in enumerate(node.children)
                ]
                return (type(node).__name__, children)

            if isinstance(node, Difference):
                positive = walk(
                    node.positive, polarity, f"{path}.positive",
                    affine, instance_path, seen, active, leaf_ids,
                )
                subtractors = [
                    walk(
                        sub, -polarity, f"{path}.subtractors[{i}]",
                        affine, instance_path, seen, active, leaf_ids,
                    )
                    for i, sub in enumerate(node.subtractors)
                ]
                return ("Difference", positive, subtractors)

            # Instance: validate identity, then traverse the referenced
            # subtree as a fresh root (its own `seen`) while keeping the
            # active path so reference cycles spanning the boundary fail.
            full_path = instance_path + (node.instance_id,)
            if full_path in instance_paths:
                raise GeometryError(
                    f"{path}: two instance references share the path "
                    f"{full_path!r} (duplicate instance_id at the same nesting)"
                )
            sub = node.child
            # A cycle spanning an instance boundary is caught by the active
            # path (walk() would reject it too, but give a specific message).
            if id(sub) in active:
                raise GeometryError(
                    f"{path}.child: reference cycle through "
                    f"Instance(instance_id={node.instance_id!r}): the referenced "
                    f"subtree is one of its own ancestors"
                )
            instance_paths.add(full_path)
            nested = _compose_affine(
                affine, (node.inverse, node.translation_inv)
            )
            expr = walk(
                sub, polarity, f"{path}.child",
                nested, full_path, set(), active, set(),
            )
            return ("Instance", expr)
        finally:
            active.discard(nid)

    expr = walk(tree, 1, "tree", _identity_affine(), (), set(), set(), set())

    # Unique effective-leaf identities (guaranteed structurally; checked
    # defensively to keep source keys unambiguous).
    identities = {}
    for leaf, _bm, _bv, _pol, ipath in occurrences:
        key = (ipath, leaf.leaf_id)
        if key in identities:
            raise GeometryError(  # pragma: no cover - structural checks above
                f"tree: duplicate effective leaf {leaf.leaf_id!r} at path {ipath!r}"
            )
        identities[key] = leaf
    return occurrences, expr


def _make_evaluator(expr):
    """Boolean evaluator over the expanded expression; leaf state keyed by
    occurrence index."""
    def evaluate(node_expr, leaf_state):
        tag = node_expr[0]
        if tag == "leaf":
            return leaf_state(node_expr[1])
        if tag == "Instance":
            return evaluate(node_expr[1], leaf_state)
        if tag == "Union":
            saw_touch = False
            for child in node_expr[1]:
                state = evaluate(child, leaf_state)
                if state == _IN:
                    return _IN
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _OUT
        if tag == "Intersection":
            saw_touch = False
            for child in node_expr[1]:
                state = evaluate(child, leaf_state)
                if state == _OUT:
                    return _OUT
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _IN
        if tag == "Difference":
            result = evaluate(node_expr[1], leaf_state)
            for sub in node_expr[2]:
                result = _sub_state(result, evaluate(sub, leaf_state))
            return result
        raise GeometryError(f"tree: internal evaluator error ({tag})")

    return evaluate


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


# ---------------------------------------------------------------------------
# normals
# ---------------------------------------------------------------------------

_BOX_FACE_AXIS = {
    "x_lo": (0, -1), "x_hi": (0, +1),
    "y_lo": (1, -1), "y_hi": (1, +1),
    "z_lo": (2, -1), "z_hi": (2, +1),
}


def _local_box_normal(face):
    """Signed axis unit normal of a box face in local coordinates (Fractions)."""
    axis, sign = _BOX_FACE_AXIS[face]
    return tuple(Fraction(sign) if i == axis else Fraction(0) for i in range(3))


def _local_sphere_normal(sphere, origin_l, direction_l, tval):
    """Outward unit normal of a sphere in local coordinates.

    The ray arguments are the *local* ray; ``(p - c)/r`` is already a unit
    normal.  Fraction components for rational event t, Decimal (70 digits)
    for an irrational root.
    """
    if isinstance(tval, Fraction):
        return tuple(
            (origin_l[i] + tval * direction_l[i] - sphere.center[i]) / sphere.radius
            for i in range(3)
        ), True
    with localcontext() as ctx:
        ctx.prec = 70
        t_dec = tval.decimal(70)
        normal = tuple(
            (
                frac_to_decimal(origin_l[i])
                + t_dec * frac_to_decimal(direction_l[i])
                - frac_to_decimal(sphere.center[i])
            )
            / frac_to_decimal(sphere.radius)
            for i in range(3)
        )
    return normal, False


def _world_normal(b_inv, n_local, rational):
    """Transform a local outward normal to world coordinates and renormalize.

    A local normal covector maps by the inverse transpose of the local->world
    map, i.e. the transpose ``B^T`` of the world->local matrix B, followed by
    normalization in world metric.  This works unchanged for mirrored frames
    (negative determinant): the outward normal of a mirrored surface is the
    image of the local outward normal; enter/exit are never swapped.
    """
    if rational:
        w = [
            sum(b_inv[j][i] * n_local[j] for j in range(3))
            for i in range(3)
        ]
        norm = math.sqrt(sum(float(x) * float(x) for x in w))
        return tuple(float(x) / norm + 0.0 for x in w)
    with localcontext() as ctx:
        ctx.prec = 70
        bd = [[frac_to_decimal(b_inv[j][i]) for j in range(3)] for i in range(3)]
        w = [sum(bd[i][j] * n_local[j] for j in range(3)) for i in range(3)]
        norm = sum(x * x for x in w).sqrt()
        return tuple(float(x / norm) + 0.0 for x in w)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def ray_intervals(tree, ray):
    """Intersect a ray with a CSG tree (possibly containing Instance refs);
    return a RayIntervalsResult.

    See module docstring and README.md for the full contract.  Raises
    GeometryError for invalid input (including singular transforms and
    reference cycles) and CertificationBudgetExhausted when event ordering
    exceeds the precision budget.
    """
    if not isinstance(ray, Ray):
        raise GeometryError(f"ray: expected a Ray, got {type(ray).__name__}")
    if not _is_node(tree):
        raise GeometryError(f"tree: not a CSG node ({type(tree).__name__})")
    origin = ray.origin
    direction = ray.direction

    occurrences, expr = _validate_and_expand(tree)
    n_leaves = len(occurrences)
    evaluate = _make_evaluator(expr)

    # -- per-occurrence local rays and events ------------------------------
    # The world ray O + t*D is mapped into each leaf's local frame by the
    # composed inverse affine: p_l = B*(O + t*D) + v.  The direction is
    # deliberately NOT renormalized: scaling it keeps the very same t on the
    # local line (a scaled sphere would otherwise shift its hit parameters),
    # so every event t below is simultaneously the world-ray parameter.
    leaf_events = []
    local_rays = []
    for leaf, b_inv, v_inv, _pol, _ipath in occurrences:
        origin_l = [
            v_inv[i] + sum(b_inv[i][j] * origin[j] for j in range(3))
            for i in range(3)
        ]
        direction_l = [
            sum(b_inv[i][j] * direction[j] for j in range(3))
            for i in range(3)
        ]
        local_rays.append((origin_l, direction_l))
        if isinstance(leaf, Sphere):
            leaf_events.append(_sphere_events(leaf, origin_l, direction_l))
        else:
            leaf_events.append(_box_events(leaf, origin_l, direction_l))

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

    leaf_events_at = [dict() for _ in range(n_leaves)]
    for li, events in enumerate(leaf_events):
        for tval, kind, faces in events:
            leaf_events_at[li].setdefault(cluster_index(tval), []).append((kind, faces))

    # -- per-leaf segment / point states ----------------------------------
    # Segment j is the open interval between cluster j-1 and cluster j
    # (segment 0 lies before the first cluster, segment n after the last).
    leaf_seg = [[_OUT] * (n_clusters + 1) for _ in range(n_leaves)]
    leaf_pt = [[_OUT] * n_clusters for _ in range(n_leaves)]
    for li in range(n_leaves):
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

        return evaluate(expr, leaf_state)

    def pt_state(i, forced=None):
        def leaf_state(li):
            if forced is not None and forced[0] == li:
                return forced[1]
            return leaf_pt[li][i]

        return evaluate(expr, leaf_state)

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
        """All effective-leaf faces at cluster ci whose state matters."""
        found = []
        for li, (leaf, b_inv, _v_inv, polarity, ipath) in enumerate(occurrences):
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
                            local_rays[li][0], local_rays[li][1],
                            b_inv, polarity, ipath,
                        )
                    )
        # instance path first: two references of the same original leaf are
        # distinct sources, ordered deterministically by reference path
        found.sort(key=lambda s: (s.path, s.leaf_id, s.face))
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
        merged.sort(key=lambda s: (s.path, s.leaf_id, s.face, s.kind))
        unique = []
        seen = set()
        for source in merged:
            key = (source.path, source.leaf_id, source.face, source.kind)
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


def _make_source(leaf, face, kind, tval, origin_l, direction_l, b_inv,
                 polarity, instance_path):
    if isinstance(leaf, Sphere):
        n_local, rational = _local_sphere_normal(
            leaf, origin_l, direction_l, tval
        )
    else:
        n_local, rational = _local_box_normal(face), True
    normal = _world_normal(b_inv, n_local, rational)
    flipped = polarity < 0
    if flipped:
        normal = tuple(-component for component in normal)
    return Source(
        leaf_id=leaf.leaf_id,
        face=face,
        kind=kind,
        # "+ 0.0" normalizes -0.0 produced by flipping zero components
        normal=tuple(float(component) + 0.0 for component in normal),
        flipped=flipped,
        path=tuple(instance_path),
    )
