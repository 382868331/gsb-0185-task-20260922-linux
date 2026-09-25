"""Core sweep algorithm: CSG ray-interval intersection with cavities and
reusable, transformable subtree instances.

Pipeline of :func:`ray_intervals`:

1. Validate the scene DAG (declared-node budget, instance-leaf-occurrence
   budget, no reference cycles, unique leaf and instance ids, sharing only
   through :class:`Instance` edges) and expand it into *occurrences*: every
   walk from the root reaches one leaf occurrence carrying its Difference
   polarity, its instance ``path`` and its composed world-from-local affine
   map ``p_world = A @ p_local + b``.  The shared child nodes are never
   copied or mutated; each reference just records another frame.
2. For every occurrence, inverse-transform the world ray into the leaf's
   local frame: ``o_local = A^-1 (o - b)``, ``d_local = A^-1 d``.  The
   direction is deliberately NOT renormalized, so the ray parameter t is
   preserved (also under reflections: enter/exit are never swapped).  Each
   occurrence then yields the same exact events as a plain leaf:
   enter/exit, touch, tstart/tend.
3. Cluster coincident event parameters exactly, then sweep the t-line with a
   three-valued state (IN / T / OUT) per leaf occurrence and per Boolean
   node.  Events at the same t are applied as one batch, so coincident
   events that cause no state change never create phantom boundaries.
4. Regularize: maximal T-runs flanked by solid on both sides are swallowed
   (a tangent touch does not puncture a solid); T-runs flanked by empty on
   both sides become zero-measure touches; the rest become ordinary
   boundaries.  Isolated empty points between solid segments are swallowed.
5. Transform local face normals to world coordinates with the inverse
   transpose A^-T and renormalize (exact Fractions stay exact; irrational
   normals get one float rounding), apply the Difference polarity flip at
   the occurrence's own level, and emit sorted disjoint open intervals,
   touches, boundaries (sources qualified by instance path + original leaf
   id) and the nearest boundary with t > 0.
"""

from __future__ import annotations

import math
from decimal import localcontext
from fractions import Fraction

from ._exact import Root, cmp_tval, frac_to_decimal, sorted_unique, tval_to_float
from ._types import (
    MAX_LEAF_OCCURRENCES,
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

# Evaluator node tags (small ints make the hot sweep loop cheap).
_E_LEAF = 0
_E_UNION = 1
_E_INTER = 2
_E_DIFF = 3

_FRAC0 = Fraction(0)
_FRAC1 = Fraction(1)
_IDENTITY = (
    _FRAC1, _FRAC0, _FRAC0,
    _FRAC0, _FRAC1, _FRAC0,
    _FRAC0, _FRAC0, _FRAC1,
)


# ---------------------------------------------------------------------------
# exact 3x3 affine helpers (row-major tuples of Fractions)
# ---------------------------------------------------------------------------

def _mat_mul(A, B):
    """C = A @ B for row-major 3x3 tuples."""
    return tuple(
        sum(A[3 * r + k] * B[3 * k + c] for k in range(3))
        for r in range(3) for c in range(3)
    )


def _mat_vec(A, v):
    return tuple(
        A[3 * r] * v[0] + A[3 * r + 1] * v[1] + A[3 * r + 2] * v[2]
        for r in range(3)
    )


def _invert3(A):
    """Exact inverse of a non-singular 3x3 Fraction matrix (adj/det)."""
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = A
    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20
    c10 = a02 * a21 - a01 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a01 * a20 - a00 * a21
    c20 = a01 * a12 - a02 * a11
    c21 = a02 * a10 - a00 * a12
    c22 = a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c01 + a02 * c02
    if det == 0:
        raise GeometryError(  # unreachable: Instance rejects singular maps
            "Instance: composed transform is singular"
        )
    # adjugate = cofactor transpose
    adj = (c00, c10, c20, c01, c11, c21, c02, c12, c22)
    return tuple(value / det for value in adj)


# ---------------------------------------------------------------------------
# DAG validation + expansion into leaf occurrences
# ---------------------------------------------------------------------------

def _prepare(tree):
    """Validate the scene DAG and expand it into leaf occurrences.

    Returns ``(occurrences, evaluator)`` where each occurrence is a dict with
    keys ``leaf``, ``polarity``, ``path``, ``A``, ``b``, ``Ainv`` and the
    evaluator is a nested ``(tag, *args)`` structure whose leaf nodes index
    the occurrence list.

    Rules (see README.md):
    * distinct declared nodes (leaves + internal + instances) <= 100;
    * leaf occurrences after expansion <= MAX_LEAF_OCCURRENCES;
    * non-Instance edges form a proper tree (sharing is allowed only through
      an Instance reference);
    * following Instance child edges must never revisit a node on the current
      walk (no reference cycles);
    * leaf_id / instance_id are unique among distinct declared nodes.
    """
    occurrences = []
    declared = set()                 # ids of distinct declared nodes
    leaf_id_owner = {}               # leaf_id -> node id of its first declaration
    instance_id_owner = {}           # instance_id -> node id of its first declaration
    active = set()                   # node ids on the current DFS walk

    def expand(node, polarity, path, A, b, static_path, skeleton):
        """Walk the DAG.  ``skeleton`` is the set of node ids reached by
        non-Instance edges in the current skeleton expansion: the scene
        root starts one, and every Instance child starts a fresh one, so the
        same shared subtree may be walked many times while plain sharing
        (two static edges into one node) stays forbidden.
        """
        nid = id(node)
        first_sight = nid not in declared
        if first_sight:
            if len(declared) >= MAX_TREE_NODES:
                raise GeometryError(
                    f"{static_path}: CSG scene exceeds the limit of "
                    f"{MAX_TREE_NODES} distinct declared nodes"
                )
            declared.add(nid)
        if nid in skeleton:
            raise GeometryError(
                f"{static_path}: node {node!r} is shared/reused outside of "
                f"an Instance (a node may be referenced twice only through "
                f"an Instance node)"
            )
        skeleton.add(nid)

        if nid in active:
            raise GeometryError(
                f"{static_path}: reference cycle detected; an Instance may "
                f"not (transitively) reference its own subtree"
            )
        active.add(nid)
        try:
            if isinstance(node, (Sphere, Box)):
                if first_sight:
                    owner = leaf_id_owner.get(node.leaf_id)
                    if owner is not None and owner != nid:
                        raise GeometryError(
                            f"{static_path}: duplicate leaf_id "
                            f"{node.leaf_id!r} (first used by another node)"
                        )
                    leaf_id_owner[node.leaf_id] = nid
                if len(occurrences) >= MAX_LEAF_OCCURRENCES:
                    raise GeometryError(
                        f"{static_path}: instance expansion exceeds the limit "
                        f"of {MAX_LEAF_OCCURRENCES} leaf occurrences"
                    )
                index = len(occurrences)
                occurrences.append(
                    {
                        "leaf": node,
                        "polarity": polarity,
                        "path": path,
                        "A": A,
                        "b": b,
                        "Ainv": _IDENTITY if A is _IDENTITY else _invert3(A),
                    }
                )
                return (_E_LEAF, index)

            if isinstance(node, (Union, Intersection)):
                children = [
                    expand(
                        child, polarity, path, A, b,
                        f"{static_path}.children[{i}]", skeleton,
                    )
                    for i, child in enumerate(node.children)
                ]
                tag = _E_UNION if isinstance(node, Union) else _E_INTER
                return (tag, tuple(children))

            if isinstance(node, Difference):
                positive = expand(
                    node.positive, polarity, path, A, b,
                    f"{static_path}.positive", skeleton,
                )
                subtractors = tuple(
                    expand(
                        sub, -polarity, path, A, b,
                        f"{static_path}.subtractors[{i}]", skeleton,
                    )
                    for i, sub in enumerate(node.subtractors)
                )
                return (_E_DIFF, positive, subtractors)

            if isinstance(node, Instance):
                if first_sight:
                    owner = instance_id_owner.get(node.instance_id)
                    if owner is not None and owner != nid:
                        raise GeometryError(
                            f"{static_path}.instance_id: duplicate instance_id "
                            f"{node.instance_id!r} (first used by another node)"
                        )
                    instance_id_owner[node.instance_id] = nid
                # Compose frames: p_world = A (L p_child + t) + b
                #                       = (A L) p_child + (A t + b)
                A2 = _mat_mul(A, node.linear)
                At = _mat_vec(A, node.translation)
                b2 = tuple(At[i] + b[i] for i in range(3))
                return expand(
                    node.child, polarity, path + (node.instance_id,), A2, b2,
                    f"{static_path}.child[Instance {node.instance_id!r}]",
                    set(),
                )

            raise GeometryError(
                f"{static_path}: not a CSG node ({type(node).__name__})"
            )
        finally:
            active.discard(nid)

    root_eval = expand(tree, 1, (), _IDENTITY, (_FRAC0,) * 3, "tree", set())
    return occurrences, root_eval


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
# three-valued Boolean evaluation over the expanded occurrence tree
# ---------------------------------------------------------------------------

def _sub_state(positive, negative):
    """Regularized difference of two leaf states (subtractor is closed)."""
    if negative == _IN:
        return _OUT
    if negative == _OUT:
        return positive
    return _OUT if positive == _OUT else _T


def _make_evaluator(root_eval):
    def evaluate(node, leaf_state):
        tag = node[0]
        if tag == _E_LEAF:
            return leaf_state(node[1])
        if tag == _E_UNION:
            saw_touch = False
            for child in node[1]:
                state = evaluate(child, leaf_state)
                if state == _IN:
                    return _IN
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _OUT
        if tag == _E_INTER:
            saw_touch = False
            for child in node[1]:
                state = evaluate(child, leaf_state)
                if state == _OUT:
                    return _OUT
                if state == _T:
                    saw_touch = True
            return _T if saw_touch else _IN
        # difference: (tag, positive, subtractors)
        result = evaluate(node[1], leaf_state)
        for sub in node[2]:
            result = _sub_state(result, evaluate(sub, leaf_state))
        return result

    return evaluate


# ---------------------------------------------------------------------------
# normals (local frame -> world frame)
# ---------------------------------------------------------------------------

def _local_sphere_normal(sphere, origin, direction, tval):
    """Outward unit normal of a sphere in *local* coordinates.

    Returns exact Fraction components when ``tval`` is rational (the hit
    point divided by the radius is already a unit vector); for an
    irrational quadratic root it returns Decimal components at 70 digits.
    """
    if isinstance(tval, Fraction):
        return [
            (origin[i] + tval * direction[i] - sphere.center[i]) / sphere.radius
            for i in range(3)
        ], True
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
        return normal, False


def _normalize_exact(components):
    """Normalize a Fraction 3-vector exactly when its length^2 is a square."""
    length2 = sum(c * c for c in components)
    num_root = math.isqrt(length2.numerator)
    den_root = math.isqrt(length2.denominator)
    if num_root * num_root == length2.numerator and \
            den_root * den_root == length2.denominator:
        length = Fraction(num_root, den_root)
        return tuple(float(c / length) for c in components), True
    return None, False


def _normalize_decimal(components):
    with localcontext() as ctx:
        ctx.prec = 70
        length = sum(c * c for c in components).sqrt()
        return tuple(float(c / length) for c in components)


def _world_normal(local_normal, exact_local, Ainv):
    """Push a local outward normal to world space: n_w = normalize(A^-T n_l).

    The inverse transpose is required so that the transformed normal stays
    perpendicular to the transformed tangent plane (shears and anisotropic
    scaling would break a plain ``A n`` / matrix-vector transform).  For a
    reflection (det A < 0) this yields the outward normal of the mirrored
    solid directly -- no enter/exit swapping is involved.
    """
    if Ainv is None:
        # untransformed occurrence: local == world
        if exact_local:
            return tuple(float(c) + 0.0 for c in local_normal)
        return _normalize_decimal(local_normal)
    if exact_local:
        # Ainv row-major: (A^-T n)_r = sum_c Ainv[c*3+r] n_c (columns of Ainv)
        transformed = [
            sum(Ainv[c * 3 + r] * local_normal[c] for c in range(3))
            for r in range(3)
        ]
        normalized, ok = _normalize_exact(transformed)
        if ok:
            return tuple(c + 0.0 for c in normalized)
        dec = [frac_to_decimal(c) for c in transformed]
        return tuple(c + 0.0 for c in _normalize_decimal(dec))
    with localcontext() as ctx:
        ctx.prec = 70
        dAinv = [frac_to_decimal(v) for v in Ainv]
        transformed = [
            sum(dAinv[c * 3 + r] * local_normal[c] for c in range(3))
            for r in range(3)
        ]
    return tuple(c + 0.0 for c in _normalize_decimal(transformed))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def ray_intervals(tree, ray):
    """Intersect a ray with a CSG scene (possibly instanced); return a result.

    See module docstring and README.md for the full contract.  Raises
    GeometryError for invalid input (including reference cycles and singular
    instance matrices) and CertificationBudgetExhausted when event ordering
    exceeds the precision budget.
    """
    if not isinstance(ray, Ray):
        raise GeometryError(f"ray: expected a Ray, got {type(ray).__name__}")
    if not _is_node(tree):
        raise GeometryError(f"tree: not a CSG node ({type(tree).__name__})")
    origin = ray.origin
    direction = ray.direction

    occurrences, root_eval = _prepare(tree)
    evaluate = _make_evaluator(root_eval)

    # -- per-occurrence local ray and events --------------------------------
    # The world ray is inverse-transformed into each leaf frame.  The local
    # direction is A^-1 d WITHOUT renormalization: p(o + t d) = p_l + t d_l,
    # so the hit parameter stays the same world t (preserving t-intervals,
    # and keeping enter/exit correct under negative determinants).
    occurrence_events = []
    for occ in occurrences:
        Ainv = occ["Ainv"]
        if Ainv is _IDENTITY:
            o_local = origin
            d_local = direction
        else:
            shifted = tuple(origin[i] - occ["b"][i] for i in range(3))
            o_local = _mat_vec(Ainv, shifted)
            d_local = _mat_vec(Ainv, direction)
        leaf = occ["leaf"]
        if isinstance(leaf, Sphere):
            events = _sphere_events(leaf, o_local, d_local)
        else:
            events = _box_events(leaf, o_local, d_local)
        occurrence_events.append(events)

    # -- exact clustering of coincident event parameters -------------------
    all_tvals = [
        tval for events in occurrence_events
        for (tval, _kind, _faces) in events
    ]
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

    n_occ = len(occurrences)
    events_at = [dict() for _ in occurrences]
    for li, events in enumerate(occurrence_events):
        for tval, kind, faces in events:
            events_at[li].setdefault(cluster_index(tval), []).append((kind, faces))

    # -- per-occurrence segment / point states ------------------------------
    # Segment j is the open interval between cluster j-1 and cluster j
    # (segment 0 lies before the first cluster, segment n after the last).
    leaf_seg = [[_OUT] * (n_clusters + 1) for _ in occurrences]
    leaf_pt = [[_OUT] * n_clusters for _ in occurrences]
    for li in range(n_occ):
        depth = 0
        riding = 0
        for ci in range(n_clusters):
            events = events_at[li].get(ci, [])
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
            leaf_seg[li][ci + 1] = _IN if depth > 0 else (
                _T if riding > 0 else _OUT
            )

    def seg_state(j, forced=None):
        def leaf_state(li):
            if forced is not None and forced[0] == li:
                return forced[1]
            return leaf_seg[li][j]

        return evaluate(root_eval, leaf_state)

    def pt_state(i, forced=None):
        def leaf_state(li):
            if forced is not None and forced[0] == li:
                return forced[1]
            return leaf_pt[li][i]

        return evaluate(root_eval, leaf_state)

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
        """All occurrence faces at cluster ci whose state matters."""
        found = []
        for li, occ in enumerate(occurrences):
            events = events_at[li].get(ci)
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
                            occ, face, source_kind, clusters[ci],
                            origin, direction,
                        )
                    )
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

    # -- touches ------------------------------------------------------------
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


def _make_source(occ, face, kind, tval, origin, direction):
    """Build a world-space Source for one occurrence face event."""
    leaf = occ["leaf"]
    Ainv = occ["Ainv"]
    if Ainv is _IDENTITY:
        o_local = origin
        d_local = direction
    else:
        o_local = _mat_vec(Ainv, tuple(origin[i] - occ["b"][i] for i in range(3)))
        d_local = _mat_vec(Ainv, direction)
    if isinstance(leaf, Sphere):
        local_normal, exact_local = _local_sphere_normal(
            leaf, o_local, d_local, tval
        )
        normal = _world_normal(local_normal, exact_local, Ainv)
    else:
        # Box face normals are signed axis vectors in the leaf frame; push
        # the (unsigned) axis through A^-T and reapply the face sign.
        col = "xyz".index(face[0])
        sign = 1.0 if face.endswith("hi") else -1.0
        local = [Fraction(1) if c == col else Fraction(0) for c in range(3)]
        world = _world_normal(local, True, Ainv)
        normal = tuple(sign * c for c in world)
    flipped = occ["polarity"] < 0
    if flipped:
        normal = tuple(-component for component in normal)
    return Source(
        leaf_id=leaf.leaf_id,
        face=face,
        kind=kind,
        # "+ 0.0" normalizes -0.0 produced by flipping zero components
        normal=tuple(float(component) + 0.0 for component in normal),
        flipped=flipped,
        path=occ["path"],
    )
