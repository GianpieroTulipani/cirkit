"""Region graphs built from sliding image windows and their complements."""

from collections import defaultdict

from cirkit.templates.region_graph.algorithms.utils import HypercubeToScope
from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)

Rectangle = tuple[int, int, int, int]  # Top, left, bottom, right; bottom/right are exclusive.


# pylint: disable-next=invalid-name
def WindowPartitionGraph(
    shape: tuple[int, int, int], *, window_shape: tuple[int, int] = (2, 2), stride: int = 1
) -> RegionGraph:
    """Build a region graph with alternative placements of a sliding window.

    Each window placement partitions its parent rectangle into the window and up to four
    non-overlapping rectangles covering the remainder. Centered horizontal and vertical
    splits are also included, ensuring that smaller or equal-sized windows can reach pixels.
    Rectangles reached through different partitions share a single region node.

    Args:
        shape: Image shape ``(channels, height, width)``.
        window_shape: Height and width of the sliding window.
        stride: Step between consecutive window placements in either direction.

    Returns:
        A region graph with one root covering the full image.
    """
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("Image shape must contain three positive dimensions")
    if len(window_shape) != 2 or any(size <= 0 for size in window_shape):
        raise ValueError("Window shape must contain two positive dimensions")
    if stride <= 0:
        raise ValueError("Stride must be positive")

    channels, height, width = shape
    window_height, window_width = window_shape
    hypercube_to_scope = HypercubeToScope(shape)
    nodes: list[RegionGraphNode] = []
    in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)
    regions: dict[Rectangle, RegionNode] = {}

    def build(rectangle: Rectangle) -> RegionNode:
        if rectangle in regions:
            return regions[rectangle]

        top, left, bottom, right = rectangle
        scope = hypercube_to_scope[((0, top, left), (channels, bottom, right))]
        region = RegionNode(scope)
        regions[rectangle] = region
        nodes.append(region)

        if bottom - top == 1 and right - left == 1:
            return region

        seen_partitions: set[tuple[Rectangle, ...]] = set()

        def add_partition(children: list[Rectangle]) -> None:
            children = [child for child in children if child[0] < child[2] and child[1] < child[3]]
            if len(children) < 2:
                return
            key = tuple(sorted(children))
            if key in seen_partitions:
                return
            seen_partitions.add(key)

            partition = PartitionNode(scope)
            nodes.append(partition)
            in_nodes[region].append(partition)
            in_nodes[partition].extend(build(child) for child in children)

        if bottom - top > 1:
            middle = (top + bottom) // 2
            add_partition([(top, left, middle, right), (middle, left, bottom, right)])
        if right - left > 1:
            middle = (left + right) // 2
            add_partition([(top, left, bottom, middle), (top, middle, bottom, right)])

        if bottom - top >= window_height and right - left >= window_width:
            for window_top in range(top, bottom - window_height + 1, stride):
                window_bottom = window_top + window_height
                for window_left in range(left, right - window_width + 1, stride):
                    window_right = window_left + window_width
                    add_partition(
                        [
                            (window_top, window_left, window_bottom, window_right),
                            (top, left, window_top, right),
                            (window_bottom, left, bottom, right),
                            (window_top, left, window_bottom, window_left),
                            (window_top, window_right, window_bottom, right),
                        ]
                    )

        return region

    root = build((0, 0, height, width))
    return RegionGraph(nodes, in_nodes, outputs=[root])
