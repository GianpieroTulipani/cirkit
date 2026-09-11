"""Small, reproducible torch.compile diagnostics; run from the repository root.

    python -m scripts.diagnose_torch_compile --backend aot_eager
    python -m scripts.diagnose_torch_compile --backend inductor --quick

Uses existing test fixtures. No datasets or model checkpoints are needed.
"""

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.symbolic.layers import MultichannelCategoricalLayer
from cirkit.symbolic.parameters import mixing_weight_factory
from cirkit.templates.region_graph import QuadGraph
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from tests.symbolic.test_from_region_graph import categorical_layer_factory
from tests.symbolic.test_utils import (
    build_monotonic_bivariate_gaussian_hadamard_dense_pc,
    build_monotonic_structured_categorical_cpt_pc,
)


def rgb_shared_factory():
    """Same symbolic construction as LearnSPN's RGB/full-sharing baseline.

    Builds directly to avoid importing the unused k-means/MIWAE dependencies.
    """
    shared = None

    def input_factory(scope, num_units):
        nonlocal shared
        layer = MultichannelCategoricalLayer(
            scope, num_units, num_channels=3, num_categories=256,
            probs=None if shared is None else shared.ref())
        if shared is None:
            shared = layer.probs
        return layer

    weights = parameterization_to_factory(
        Parameterization(activation='none', initialization='uniform'))
    return QuadGraph((3, 3, 3)).build_circuit(
        num_input_units=3, num_sum_units=3, sum_product='cp',
        input_factory=input_factory, factorize_multivariate=False,
        sum_weight_factory=weights,
        nary_sum_weight_factory=lambda shape: mixing_weight_factory(shape, param_factory=weights))


def check_case(name, factory, fold, optimize, semiring, backend, gaussian=False):
    torch.compiler.reset()
    torch.manual_seed(42)
    compiler = TorchCompiler(fold=fold, optimize=optimize, semiring=semiring)
    symbolic = factory()
    circuit = compiler.compile(symbolic)
    partition = compiler.compile(SF.integrate(symbolic))
    x = (torch.randn(4, circuit.num_variables) if gaussian else
         torch.randint(0, 2, (4, circuit.num_variables)))

    def loss_step(data):
        return (circuit(data) - partition()).sum()

    params = list(circuit.parameters())
    reference = loss_step(x)
    assert torch.isfinite(reference), 'Non-finite eager loss: check the fixture initialization'
    reference_grads = torch.autograd.grad(reference, params, allow_unused=True)
    explanation = torch._dynamo.explain(loss_step)(x)
    result = dict(name=name, fold=fold, optimize=optimize, semiring=semiring,
                  layers=[type(layer).__name__ for layer in circuit.layers],
                  graphs=explanation.graph_count,
                  graph_breaks=explanation.graph_break_count,
                  reasons=[str(r) for r in explanation.break_reasons])
    compiled = torch.compile(loss_step, backend=backend, fullgraph=True)
    output = compiled(x)
    gradients = torch.autograd.grad(output, params, allow_unused=True)
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
    for expected, actual in zip(reference_grads, gradients):
        if expected is None or actual is None:
            assert expected is actual
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # A different input and batch size also have to give the eager result.
    new_x = torch.randn(3, circuit.num_variables) if gaussian else 1 - x[:3]
    torch.testing.assert_close(compiled(new_x), loss_step(new_x), rtol=1e-4, atol=1e-5)
    result['status'] = 'PASS forward, parameter gradients, changed input/batch'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='aot_eager')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--rgb-only', action='store_true')
    parser.add_argument('--quad-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = dict(torch=torch.__version__, python=platform.python_version(),
                   platform=platform.platform(), cuda=torch.cuda.is_available(),
                   backend=args.backend, cases=[])
    configs = [(True, True, 'lse-sum')] if args.quick else list(
        itertools.product((False, True), (False, True), ('sum-product', 'lse-sum')))
    cases = [('categorical', build_monotonic_structured_categorical_cpt_pc, *c, False)
             for c in configs]
    if not args.quick:
        cases.append(('gaussian', build_monotonic_bivariate_gaussian_hadamard_dense_pc,
                      True, True, 'lse-sum', True))
        for product in ('cp', 'tucker'):
            def factory(product=product):
                return QuadGraph((1, 3, 3)).build_circuit(
                    num_input_units=3, num_sum_units=3, sum_product=product,
                    sum_weight_factory=parameterization_to_factory(
                        Parameterization(activation='none', initialization='uniform')),
                    input_factory=categorical_layer_factory)
            cases.append((f'quadgraph-{product}', factory, True, True, 'lse-sum', False))
        cases.append(('rgb-full-sharing', rgb_shared_factory, True, True, 'lse-sum', False))
    if args.rgb_only:
        cases = [('rgb-full-sharing', rgb_shared_factory, True, True, 'lse-sum', False)]
    if args.quad_only:
        cases = [case for case in cases if case[0].startswith('quadgraph-')]
    for name, factory, fold, optimize, semiring, gaussian in cases:
        start = time.perf_counter()
        try:
            result = check_case(name, factory, fold, optimize, semiring, args.backend, gaussian)
        except Exception as exc:
            result = dict(name=name, fold=fold, optimize=optimize, semiring=semiring,
                          status='FAIL', error=f'{type(exc).__name__}: {exc}')
        result['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        results['cases'].append(result)
        print(json.dumps(result), flush=True)

    if not args.quick and not args.rgb_only and not args.quad_only:
        circuit = TorchCompiler(fold=True, optimize=True, semiring='lse-sum').compile(
            build_monotonic_structured_categorical_cpt_pc())
        query = IntegrateQuery(circuit)
        x = torch.randint(0, 2, (4, circuit.num_variables))
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, 0] = True
        def query_step(data, integration_mask):
            return query(data, integrate_vars=integration_mask)
        # Verify that the query itself works eagerly before testing capture.
        query_step(x, mask)
        for name, operation in (
            ('runtime_integrate_query', lambda: torch.compile(
                query_step, backend='eager', fullgraph=True)(x, mask)),
            ('torchscript_script', lambda: torch.jit.script(circuit)),
        ):
            torch.compiler.reset()
            try:
                operation()
                result = dict(name=name, status='PASS')
            except Exception as exc:
                result = dict(name=name, status='FAIL', error=f'{type(exc).__name__}: {exc}')
            results['cases'].append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
