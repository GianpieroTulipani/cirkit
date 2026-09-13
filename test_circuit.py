import torch
from cirkit.templates.data_modalities import image_data
from cirkit.templates.region_graph import QuadGraph
from cirkit.pipeline import compile


IMAGE_SHAPE = (1, 2, 2) 
SEED = 7


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
    print(compiled_circuit)


if __name__ == "__main__":
    main()
