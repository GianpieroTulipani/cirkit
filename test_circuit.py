from pathlib import Path

import torch
from cirkit.templates.data_modalities import image_data
from cirkit.templates.region_graph import QuadGraph, QuadTree, WindowPartitionGraph
from cirkit.templates.region_graph.io import plot_region_graph
from cirkit.pipeline import compile


IMAGE_SHAPE = (1, 4, 4)
SEED = 7


def plot_region_graphs() -> None:
    """Save the region graphs associated with ``IMAGE_SHAPE`` as SVG files."""
    output_dir = Path(__file__).parent
    region_graphs = {
        "quad_graph.svg": QuadGraph(IMAGE_SHAPE),
        "quad_tree_2.svg": QuadTree(IMAGE_SHAPE, num_patch_splits=2),
        "window_partition_graph.svg": WindowPartitionGraph(IMAGE_SHAPE),
    }

    for filename, region_graph in region_graphs.items():
        output_path = output_dir / filename
        plot_region_graph(region_graph, out_path=output_path)
        root = region_graph.outputs[0]
        print(
            f"Region graph saved to {output_path} "
            f"({len(list(region_graph.region_nodes))} regions, "
            f"{len(list(region_graph.partition_nodes))} partitions, "
            f"{len(region_graph.region_inputs(root))} root partitions)"
        )


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    image = torch.arange(16, dtype=torch.long).reshape(1, 1, 4, 4)
    flat_image = image.flatten(start_dim=1)  # TorchCircuit richiede forma (B, D)

    symbolic_circuit = image_data(
        IMAGE_SHAPE,
        region_graph="quad-graph",
        input_layer="categorical",
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        num_classes=1,
    )

    symbolic_layers = list(symbolic_circuit.layers)

    compiled_circuit = compile(symbolic_circuit)


if __name__ == "__main__":
    plot_region_graphs()
    # main()
