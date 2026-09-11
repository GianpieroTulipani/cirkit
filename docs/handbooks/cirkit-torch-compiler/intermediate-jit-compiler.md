# Intermediate JIT Compiler for Circuits

## Motivation

The current Torch backend compiles a symbolic circuit into a generic
`TorchCircuit`. This representation is flexible: it can execute arbitrary DAGs,
support folded and unfolded graphs, reuse the same layer classes, and evaluate
queries through the address-book mechanism.

However, after compilation the forward pass is still interpreted by Python. A
`TorchCircuit` evaluates layers by iterating over an `AddressBook`, gathering
inputs from a mutable list of intermediate tensors, calling the current module,
and appending its output. Folding reduces the number of modules and improves
batching, but it does not remove this generic execution layer.

This can limit the effectiveness of `torch.compile`. PyTorch's compiler works
best when it sees a static tensor program made mostly of direct PyTorch
operations. The current representation contains Python control flow,
generators, dynamic module lookup, list mutation, conditional branches, and
generic semiring dispatch. These patterns may cause graph breaks or prevent
kernel fusion.

An intermediate JIT compiler would sit between Cirkit's existing symbolic
compiler and low-level backends. Its purpose would be to specialize a compiled
circuit into a static executable module.

Conceptually:

```text
current path:
  symbolic Circuit
    -> generic TorchCircuit
    -> interpreted DAG execution

proposed path:
  symbolic Circuit
    -> generic TorchCircuit, optionally optimized/folded
    -> specialized static Torch module
    -> torch.compile / export / custom backend
```

## Main Goal

The intermediate compiler should generate a circuit-specific forward function
where:

- the circuit topology is fixed;
- tensor shapes and fold indices are known as much as possible;
- the semiring is specialized;
- parameter graphs are inlined or cached explicitly;
- address-book iteration is replaced by straight-line code;
- repeated algebraic patterns are simplified before reaching `torch.compile`.

Instead of evaluating:

```python
module_outputs = []
for module, inputs in address_book.lookup(module_outputs, in_graph=x):
    y = module(*inputs)
    module_outputs.append(y)
return output
```

the generated module should look closer to:

```python
def forward(self, x):
    y0 = self.input_0(x[..., self.scope_0].permute(1, 0, 2))
    y1 = self.input_1(x[..., self.scope_1].permute(1, 0, 2))
    y2 = torch.sum(torch.stack((y0, y1), dim=1), dim=1)
    y3 = torch.logsumexp(y2 + self.weight_0(), dim=...)
    return y3
```

The exact generated code can still call selected layer modules at first. The
important step is removing the generic DAG interpreter and making the execution
trace static.

## Required Properties

### Semantic Equivalence

The generated module must compute the same function as the original
`TorchCircuit`, including:

- identical outputs for the same input batch;
- identical semiring interpretation;
- identical parameter sharing;
- identical folded/unfolded behavior;
- compatible gradients with respect to learnable parameters.

The first implementation should include numerical equivalence tests between:

```text
original TorchCircuit(x)
specialized module(x)
torch.compile(specialized module)(x)
```

### Semiring Specialization

The compiler should specialize operations for the selected semiring.

For `sum-product`:

```text
semiring sum  -> torch.sum / torch.einsum
semiring prod -> torch.prod
semiring mul  -> torch.mul
```

For `lse-sum`:

```text
semiring sum  -> torch.logsumexp / torch.logaddexp
semiring prod -> torch.sum
semiring mul  -> torch.add
```

This avoids calls such as:

```python
self.semiring.sum(...)
self.semiring.prod(...)
self.semiring.einsum(...)
```

inside the hot path. The semiring choice is known at compile time, so the
generated program should not dispatch on it at runtime.

### Static Topology

The compiler should freeze:

- topological order;
- layer input dependencies;
- output layer selection;
- fold indices;
- input scope indices;
- arities and output dimensions.

This turns circuit evaluation from a graph traversal into straight-line tensor
code.

### Parameter Handling

Parameters require special care because Cirkit parameterizations are themselves
small computational graphs.

The intermediate compiler should support two modes:

- **module mode**: keep each compiled `TorchParameter` as a submodule and call
  it from generated code;
- **inline mode**: emit the parameter graph as static tensor code too.

Module mode is easier and should be the MVP. Inline mode is more powerful
because it exposes operations such as softmax, log-softmax, reduce-sum, and
outer-product directly to `torch.compile`.

Parameter sharing must be preserved. If two symbolic layers share the same
parameter, the generated module must reference the same compiled parameter, not
create independent tensors.

### Shape Discipline

The generated code should assume static rank and mostly static non-batch
dimensions. The batch dimension can remain dynamic if needed.

The compiler should record guards such as:

```text
input rank == 2
input variable dimension == max(scope) + 1
known folded dimensions match compiled metadata
```

This makes failures explicit and reduces accidental recompilation.

## Architecture

### 1. Intermediate Representation

Introduce a small IR representing static circuit execution. Example node types:

```text
InputGather
ConstantLayerCall
LayerCall
ParameterCall
SemiringSum
SemiringProd
SemiringEinsum
ConcatFolds
IndexFolds
Return
```

Each IR node should store:

- output name;
- input names;
- operation type;
- static attributes;
- output shape when known;
- reference to the original layer or parameter when useful.

The IR should be produced from a compiled `TorchCircuit`, not directly from the
symbolic circuit. This lets the new compiler reuse existing layer compilation,
optimization, folding, and parameter registration.

### 2. Lowering From `TorchCircuit`

The first lowering pass can use the existing `address_book` once, at compile
time, to produce a static sequence of IR nodes.

Runtime logic like:

```python
for entry in address_book.lookup(...):
    ...
```

should become compile-time logic:

```text
entry 0 -> emit input gather + layer call
entry 1 -> emit gather previous output + layer call
entry 2 -> emit output selection
```

After lowering, the generated module should no longer need `AddressBook.lookup`
in its forward pass.

### 3. Code Generation

For an MVP, code generation can build an `nn.Module` dynamically with a
manually written `forward` closure or generated Python source.

There are two possible approaches:

- **FX graph builder**: construct a `torch.fx.GraphModule`;
- **Python source emitter**: generate a Python class/function and `exec` it.

The FX path is cleaner for integration with PyTorch compiler tools. The source
emitter is often simpler for a prototype and easier to inspect.

Recommended MVP path:

```text
TorchCircuit
  -> StaticCircuitIR
  -> Python source module
  -> nn.Module
  -> optional torch.compile
```

Once the design stabilizes, replace the source emitter with FX generation if it
proves useful.

### 4. Algebraic Optimization Passes

Before emitting code, the IR can apply simple circuit-specific rewrites:

- remove identity products;
- remove identity sums;
- fuse consecutive fold indexing operations;
- fuse `cat` followed by immediate indexing when possible;
- specialize weighted sums in log-space;
- replace generic semiring calls with direct Torch operations;
- precompute constant fold/scoping tensors as buffers;
- cache parameter graph outputs reused multiple times in one forward pass.

For `lse-sum`, useful rewrites include:

```text
prod(x, dim) -> torch.sum(x, dim)
mul(a, b)    -> a + b
sum(x, dim)  -> torch.logsumexp(x, dim)
```

Weighted sums should be emitted directly in log-space when possible:

```text
log sum_i exp(log_w_i + child_i)
```

rather than going through a generic semiring `einsum` implementation.

## MVP Implementation Plan

### Phase 1: Static Forward Without Inlining

Implement a function:

```python
compile_static_forward(circuit: TorchCircuit) -> torch.nn.Module
```

The returned module:

- owns the original layers in a `ModuleList`;
- owns any static index tensors as buffers;
- emits a straight-line `forward`;
- still calls layer modules directly;
- avoids runtime `AddressBook.lookup`.

This phase tests whether removing the DAG interpreter is enough to make
`torch.compile` more effective.

### Phase 2: Semiring-Aware Layer Lowering

Lower the most common layer types directly:

- `TorchHadamardLayer`;
- `TorchSumLayer`;
- `TorchCategoricalLayer`;
- `TorchGaussianLayer`;
- optimized sum/product layers.

At this stage, the generated code starts replacing layer calls with direct
tensor operations.

Example:

```text
TorchHadamardLayer + lse-sum
  -> torch.sum(x, dim=1)

TorchHadamardLayer + sum-product
  -> torch.prod(x, dim=1)
```

### Phase 3: Parameter Graph Inlining

Inline `TorchParameter` graphs into the generated module.

Start with:

- tensor parameters;
- constants;
- references;
- softmax/log-softmax;
- reduce-sum/reduce-logsumexp;
- outer product / outer sum;
- indexing.

This exposes more operations to PyTorch's compiler and avoids repeated
parameter graph interpretation.

### Phase 4: `torch.compile` Integration

Add an optional flag:

```python
PipelineContext(
    backend="torch",
    semiring="lse-sum",
    fold=True,
    optimize=True,
    static_codegen=True,
    torch_compile=True,
)
```

Possible modes:

```text
static_codegen=False:
  current behavior

static_codegen=True:
  generate specialized module, run eager

static_codegen=True, torch_compile=True:
  generate specialized module, then call torch.compile
```

The implementation should keep the current backend as the reference path.

## Testing Strategy

Tests should compare the original and generated modules across:

- `sum-product` and `lse-sum`;
- folded and unfolded circuits;
- optimized and non-optimized circuits;
- categorical and Gaussian inputs;
- shared parameters;
- evidence and integration circuits;
- CPU and GPU when available;
- forward values and backward gradients.

Minimum tests:

```text
test_static_codegen_matches_torch_circuit_forward
test_static_codegen_preserves_parameter_sharing
test_static_codegen_matches_gradients
test_static_codegen_with_lse_sum
test_static_codegen_with_folded_circuit
test_static_codegen_torch_compile_no_graph_break_smoke
```

For performance, benchmark:

```text
TorchCircuit eager
TorchCircuit + torch.compile
StaticCircuit eager
StaticCircuit + torch.compile
```

The expected early win is not necessarily faster single kernels, but fewer graph
breaks, less Python overhead, and more opportunities for PyTorch Inductor to
fuse operations.

## Likely Blockers

The main technical risks are:

- dynamic indexing introduced by folding;
- custom autograd functions such as safe log variants;
- generic semiring `einsum`;
- multivariate input layers;
- query-specific execution paths such as integration masks;
- shape polymorphism across different batch sizes;
- preserving parameter sharing after folding.

These should be handled incrementally. The first target should be standard
forward evaluation of common smooth and decomposable probabilistic circuits in
`lse-sum`.

## Suggested Starting Point

Start by building a prototype outside the main compiler:

```text
cirkit/backend/torch/static_codegen.py
```

with:

```python
class StaticTorchCircuit(torch.nn.Module):
    ...

def compile_static_forward(circuit: TorchCircuit) -> StaticTorchCircuit:
    ...
```

Do not replace `TorchCircuit` immediately. Keep it as the reference
implementation and use the static compiler as an optional acceleration path.

Once the prototype is correct and measurable, integrate it into `TorchCompiler`
as a post-processing stage after optimization and folding.

## Summary

The current compiler translates symbolic circuits into a flexible PyTorch DAG.
Folding improves this DAG by batching compatible layers, but the runtime still
contains a generic Python graph interpreter.

An intermediate JIT compiler should specialize the compiled circuit into a
static tensor program. Its main responsibilities are freezing topology,
specializing semiring operations, reducing runtime bookkeeping, preserving
parameter semantics, and exposing a cleaner graph to `torch.compile`.

This would not replace Cirkit's symbolic compiler. It would complement it:
Cirkit would still use the existing compiler for correctness and modularity,
then optionally lower the result into a specialized execution form for speed.
