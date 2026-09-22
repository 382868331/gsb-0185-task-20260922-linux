"""Fixed-input demonstration for csg_ray (no network, no randomness).

Run:  python demo.py

Prints (1) a real computation on a hollow sphere, showing the cavity entry/
exit faces (flipped normals), and (2) one input-contract rejection that is
actually triggered.
"""

from csg_ray import (
    Box,
    Difference,
    InvalidGeometryError,
    Ray,
    Sphere,
    Union,
    ray_intervals,
)


def _fmt_sources(sources):
    parts = []
    for s in sources:
        n = "(" + ", ".join(f"{v:+.3f}" for v in s.normal) + ")"
        flag = " [FLIPPED cavity wall]" if s.flipped else ""
        parts.append(f"{s.leaf_id}:{s.face} normal={n}{flag}")
    return " ; ".join(parts)


def main() -> None:
    print("=== 1. Normal result: concentric hollow sphere with a cavity ===")
    outer = Sphere("outer", (0.0, 0.0, 0.0), 5.0)
    inner = Sphere("inner", (0.0, 0.0, 0.0), 2.0)
    hollow = Difference(outer, inner)

    ray = Ray((-7.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    print("tree: outer sphere r=5  -  inner sphere r=2")
    print("ray : origin=(-7,0,0), direction=(1,0,0)")
    result = ray_intervals(hollow, ray)

    print(f"solid intervals (closed, {len(result.intervals)}):")
    for iv in result.intervals:
        print(f"  t in [{iv.t_enter:.6f}, {iv.t_exit:.6f}]")
        print(f"    enter face(s): {_fmt_sources(iv.enter_sources)}")
        print(f"    exit  face(s): {_fmt_sources(iv.exit_sources)}")
    print(f"tangent touches: {len(result.touches)}")
    nb = result.nearest_boundary
    print("nearest strictly forward solid boundary: "
          f"t={nb.t:.6f} kind={nb.kind} "
          f"face={nb.representative.leaf_id}:{nb.representative.face}")
    print("cavity walls visible above: the exit at t=5 and the entry at "
          "t=9 both come from leaf 'inner' with flipped normals.")

    print()
    print("=== 2. Extra: non-spherical cavity made by a box union ===")
    # The ray threads through the box 'vent' (y in [2,6]) as well as the
    # inner sphere, so the printed exit faces include box cavity faces.
    cavity = Union(inner,
                   Box("vent", (-1.0, 2.0, -0.5), (1.0, 6.0, 0.5)))
    tree = Difference(outer, cavity)
    ray2 = Ray((-7.0, 3.0, 0.1), (1.0, 0.0, 0.0))
    result2 = ray_intervals(tree, ray2)
    for iv in result2.intervals:
        print(f"  t in [{iv.t_enter:.6f}, {iv.t_exit:.6f}]")
        print(f"    enter via {_fmt_sources(iv.enter_sources)}")
        print(f"    exit  via {_fmt_sources(iv.exit_sources)}")

    print()
    print("=== 3. Actually triggered contract rejection ===")
    bad_sphere = Sphere("bad", (0.0, 0.0, 0.0), float("nan"))
    try:
        ray_intervals(bad_sphere, Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    except InvalidGeometryError as exc:
        print(f"InvalidGeometryError raised as expected: {exc}")
    else:  # pragma: no cover
        raise AssertionError("rejection did not fire")


if __name__ == "__main__":
    main()
