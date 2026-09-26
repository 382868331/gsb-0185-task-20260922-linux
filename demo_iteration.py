"""Offline demo for the finite z-aligned cylinder extension.

Run:  python demo_iteration.py

All results are computed when the script runs.  The checks below use a small
independent membership classifier; they do not reuse ray_intervals to create
expected values.
"""

from fractions import Fraction
import time

from csg_rays import (
    Box,
    Cylinder,
    Difference,
    GeometryError,
    Intersection,
    Ray,
    Sphere,
    ray_intervals,
)


def point(ray, t):
    t = Fraction(t)
    return tuple(Fraction(ray.origin[i]) + t * Fraction(ray.direction[i]) for i in range(3))


def inside_cylinder(node, p):
    return (
        (p[0] - node.center[0]) ** 2 + (p[1] - node.center[1]) ** 2 < node.radius**2
        and node.center[2] - node.lower_height < p[2] < node.center[2] + node.upper_height
    )


def inside_sphere(node, p):
    return sum((p[i] - node.center[i]) ** 2 for i in range(3)) < node.radius**2


def inside_box(node, p):
    return all(node.mins[i] < p[i] < node.maxs[i] for i in range(3))


def reference_inside(tree, p):
    if isinstance(tree, Cylinder):
        return inside_cylinder(tree, p)
    if isinstance(tree, Sphere):
        return inside_sphere(tree, p)
    if isinstance(tree, Box):
        return inside_box(tree, p)
    if isinstance(tree, Difference):
        return reference_inside(tree.positive, p) and not inside_sphere(
            tree.subtractors[0], p
        )
    if isinstance(tree, Intersection):
        return all(reference_inside(child, p) for child in tree.children)
    raise AssertionError("unsupported demo tree")


def midpoint_samples(result):
    if not result.intervals:
        return [Fraction(-1, 2), Fraction(0), Fraction(1, 2)]
    samples = []
    first = Fraction(result.intervals[0].t0)
    last = Fraction(result.intervals[-1].t1)
    samples.extend((first - Fraction(3, 2), last + Fraction(3, 2)))
    for left, right in zip(result.intervals, result.intervals[1:]):
        samples.append((Fraction(left.t1) + Fraction(right.t0)) / 2)
    for interval in result.intervals:
        samples.append((Fraction(interval.t0) + Fraction(interval.t1)) / 2)
    return samples


def show(label, tree, ray):
    result = ray_intervals(tree, ray)
    print(f"--- {label} ---")
    for interval in result.intervals:
        print(f"  solid open interval: ({interval.t0:g}, {interval.t1:g})")
    for touch in result.touches:
        faces = ", ".join(sorted({source.face for source in touch.sources}))
        print(
            f"  zero-measure contact: t={touch.t0:g}..{touch.t1:g} "
            f"faces=[{faces}]"
        )
    for boundary in result.boundaries:
        sources = ", ".join(
            f"{source.leaf_id}:{source.face}{' flipped' if source.flipped else ''}"
            for source in boundary.sources
        )
        print(f"  t={boundary.t:g} {boundary.kind:<5} sources=[{sources}]")
    for sample_t in midpoint_samples(result):
        expected = any(
            Fraction(iv.t0) < sample_t < Fraction(iv.t1)
            for iv in result.intervals
        )
        actual = reference_inside(tree, point(ray, sample_t))
        if expected != actual:
            raise AssertionError((label, sample_t, expected, actual))
    print("  open-interval midpoint/outside checks: independent classifier matches")


def main():
    started = time.perf_counter()

    tube = Cylinder((0, 0, 0), 3, 4, 6, "tube")
    ball = Sphere((0, 0, 0), 2, "ball")
    show("cylinder minus sphere", Difference(tube, ball), Ray((-10, 0, 0), (1, 0, 0)))
    print()

    short_tube = Cylinder((0, 0, 0), 2, 1, 3, "tube")
    slab = Box((-2, -2, -1), (2, 2, 3), "slab")
    show("box intersect cylinder", Intersection([short_tube, slab]), Ray((-5, 0, 0), (1, 0, 0)))
    print()

    rim_box = Box((2, -2, 3), (4, 2, 5), "rim_box")
    show(
        "zero-measure upper-rim contact",
        Intersection([short_tube, rim_box]),
        Ray((2, 0, -5), (0, 0, 1)),
    )
    print()

    try:
        Cylinder((0, 0, 0), 1, 0, 2, "bad")
    except GeometryError as exc:
        print(f"--- invalid cylinder rejected ---\n  {exc}")

    elapsed = time.perf_counter() - started
    print(f"\ncomputed in {elapsed:.4f} seconds (limit: 90 seconds)")


if __name__ == "__main__":
    main()
