# Developer Overview

!!! note "Initial draft"

    This page is an initial draft synthesised from the codebase, open issues, and pull
    request history. Some sections may be incomplete or reflect the maintainer's current
    thinking rather than a finalised decision. See [Sources and Freshness](@ref) at the
    bottom for the list of references consulted.

This page gives contributors and curious users a narrative map of the package: how it is
structured today, what works, what is explicitly left out, and where the project is headed.
For low-level API details see [Tensor Kernels](@ref) and [Private functions](@ref).

## Architecture

The package is organised into four layers that build on each other.

```
src/
├── base-kernels/       # Pure linear-algebra primitives (no GCP knowledge)
│   ├── khatrirao.jl   #   Khatri-Rao product
│   ├── mttkrp.jl      #   Matricized tensor × Khatri-Rao product (single mode)
│   └── mttkrps.jl     #   MTTKRP for all modes at once
│
├── base-types/         # Core data structures
│   ├── cpd.jl         #   CPD — the decomposition object
│   └── sparsearraycoo.jl  # SparseArrayCOO — N-d sparse array in COO format
│
├── losses.jl           # AbstractLoss + built-in entrywise loss functions
├── constraints.jl      # AbstractConstraint + built-in projection types
│
└── decomp-gcp/         # GCP decomposition logic
    ├── main.jl         #   gcp() entry point + default_gcp_* helpers
    ├── obj-grad.jl     #   Full (deterministic) objective and gradient
    ├── stoch-obj-grad/ #   Stochastic objective and gradient
    │   ├── abstract.jl #     AbstractGCPSampler + GCPSampleOnce
    │   ├── uniform.jl
    │   ├── stratified.jl
    │   └── semistratified.jl
    └── algorithms/     #   Concrete optimisers
        ├── abstract.jl #     AbstractGCPAlgorithm + _gcp! interface
        ├── cp-als.jl
        ├── cp-fastals.jl
        ├── gcp-lbfgsb.jl
        └── gcp-adam.jl
```

### Call flow for `gcp(X, r; ...)`

```
gcp(X, r; loss, constraints, algorithm, init)
 │
 ├─ default_gcp_constraints(X, r, loss)   # if not supplied
 ├─ default_gcp_algorithm(X, r, loss, constraints)  # if not supplied
 ├─ default_gcp_init(rng, X, r, loss, constraints, algorithm)  # if not supplied
 │
 └─ _gcp!(rng, M, X, loss, constraints, algorithm)  ← dispatched on types
```

`_gcp!` is an internal function with no default implementation; each algorithm adds a
new method specialised on particular combinations of `(M, X, loss, constraints,
algorithm)` types. This design makes the package easy to extend: adding a new algorithm
only requires defining new `_gcp!` methods without touching `gcp()` itself.

### Extension points

| Abstract type | Implement to add |
|---|---|
| `AbstractLoss` | A new entrywise loss function |
| `AbstractConstraint` | A new constraint with a projection |
| `AbstractGCPAlgorithm` | A new optimisation algorithm |
| `AbstractGCPSampler` | A new stochastic sampler |

## Current Capabilities

### Algorithms

| Algorithm | Supported data | Supported loss | Supported constraints | Default for |
|---|---|---|---|---|
| `CP_FastALS` | Dense `Array{<:Real}` | `LeastSquaresLoss` | `()` (none) | Unconstrained least-squares |
| `CP_ALS` | Dense `Array{<:Real}` | `LeastSquaresLoss` | `()` (none) | — |
| `GCP_LBFGSB` | Dense `Array{<:Union{Real,Missing}}` | Any with domain `[-∞,∞]` or `[0,∞]` | `Tuple{Vararg{LowerBoundConstraint}}` | All other losses |
| `GCP_Adam` | Dense or `SparseArrayCOO` | Any with domain `[-∞,∞]` or `[0,∞]` | `Tuple{Vararg{LowerBoundConstraint}}` | — (must opt in) |

!!! note "Float64 only for GCP algorithms"

    Both `GCP_LBFGSB` and `GCP_Adam` currently require `Float64` factor matrices.
    `CP_FastALS` and `CP_ALS` also presently operate in `Float64`.
    Support for other element types is tracked in [#159](https://github.com/dahong67/GCPDecompositions.jl/issues/159).

`CP_FastALS` implements the efficient sequence-of-MTTKRP algorithm from
[Phan et al. (2013)](https://doi.org/10.1109/TSP.2013.2269903).
It is the default for unconstrained least-squares because it substantially reduces the
number of expensive tensor-matrix products.
`CP_ALS` is retained as a simpler reference implementation.

`GCP_Adam` uses Adam with a few common modifications (epoch-based failure detection and
step-size decay). It is the only stochastic algorithm and the only one that natively
handles `SparseArrayCOO` data tensors.

### Loss functions

Eleven built-in entrywise losses are provided:

| Type | Statistical model | Link | Domain |
|---|---|---|---|
| `LeastSquaresLoss` | Gaussian | identity | ``\mathbb{R}`` |
| `NonnegativeLeastSquaresLoss` | Gaussian (nonneg.) | identity | ``[0,\infty)`` |
| `PoissonLoss` | Poisson | identity | ``[0,\infty)`` |
| `PoissonLogLoss` | Poisson | log | ``\mathbb{R}`` |
| `GammaLoss` | Gamma | identity | ``[0,\infty)`` |
| `RayleighLoss` | Rayleigh | identity | ``[0,\infty)`` |
| `BernoulliOddsLoss` | Bernoulli | odds | ``[0,\infty)`` |
| `BernoulliLogitLoss` | Bernoulli | logit | ``\mathbb{R}`` |
| `NegativeBinomialOddsLoss` | Negative Binomial | odds | ``[0,\infty)`` |
| `HuberLoss` | Robust regression | identity | ``\mathbb{R}`` |
| `BetaDivergenceLoss` | General (β=0: IS, β=1: Poisson, β=2: Gaussian) | identity | ``[0,\infty)`` |

`CustomLoss` allows users to supply any entrywise loss as a Julia function; the
derivative is computed automatically via ForwardDiff if not provided explicitly.
`WrappedLoss` wraps losses from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
(loaded as a package extension).

Adding more losses (e.g., from the GCP book chapter, tensor textbook, or AO-ADMM paper)
is tracked in [#161](https://github.com/dahong67/GCPDecompositions.jl/issues/161).

### Constraints

Only `LowerBoundConstraint(value)` is currently built in. It enforces that every entry
of every factor matrix satisfies `U[k][i,j] >= value`.

The default constraint set is determined by `default_gcp_constraints`:
- Domain ``\mathbb{R}`` → no constraints (`()`)
- Domain ``[0, \infty)`` → `(LowerBoundConstraint(0.0),)`
- Any other domain → error (not yet supported)

### Stochastic samplers (for `GCP_Adam`)

| Sampler | Data | Samples | Notes |
|---|---|---|---|
| `UniformGCPSampler(s)` | Dense `Array` | `s` entries uniformly at random | |
| `StratifiedGCPSampler(p, q)` | `SparseArrayCOO` | `p` stored nonzeros + `q` true zeros | Uses rejection sampling for zeros |
| `SemistratifiedGCPSampler(p, q)` | `SparseArrayCOO` | `p` stored nonzeros + `q` assumed zeros | No rejection; samples may include stored entries |

`GCPSampleOnce` wraps any sampler so that the same index set is reused for the function
value estimate across an epoch, while a fresh set is drawn for each gradient step.

Several sampler improvements and missing combinations are tracked in
[#172](https://github.com/dahong67/GCPDecompositions.jl/issues/172).

### Data types

- **`Array{T,N}`** (standard Julia dense arrays) — including `Array{Union{T,Missing},N}`
  for data with missing entries. Missing entries are skipped in the objective and treated
  as zero gradient contribution.
- **`SparseArrayCOO{Tv,Ti,N}`** — COO-format sparse array. Duplicate indices are
  **summed** (not overwritten). Supported by `GCP_Adam` only.

## Design Decisions

The following decisions are worth documenting explicitly so they do not need to be
rediscovered.

### `_gcp!` as the dispatch target

Rather than overloading `gcp` directly, the public function validates and normalises its
inputs and then forwards to internal `_gcp!` methods. Algorithm authors extend `_gcp!`,
not `gcp`. This was introduced during the reorganisation of GCP functions in
[PR #106](https://github.com/dahong67/GCPDecompositions.jl/pull/106).

### Entrywise losses only (for now)

`AbstractLoss` currently represents entrywise functions ``f(x_i, m_i)``. More general
losses (e.g., group-level or non-separable) are not yet supported. Extending the
hierarchy to support non-entrywise losses is under exploration in
[PR #96](https://github.com/dahong67/GCPDecompositions.jl/pull/96).

### Default algorithm as hardcoded dispatch

`default_gcp_algorithm` is currently a set of Julia methods rather than a named,
user-selectable object. This means a user cannot explicitly pass `algorithm =
AutoAlgorithm()` to opt in to the automatic choice. Making the default a first-class
type is tracked in [#170](https://github.com/dahong67/GCPDecompositions.jl/issues/170).

### `rand` is not defined for `CPD`

It is not obvious what the canonical behaviour of `rand(::CPD)` should be (sample
entries from the model distribution? sample random factor entries?). Until there is a
clear answer, `rand` is intentionally left undefined. The rationale is documented in
[#169](https://github.com/dahong67/GCPDecompositions.jl/issues/169).

### Mode ordering and performance

The MTTKRP algorithms are more efficient when tensor modes are sorted by size in
increasing order. Users can improve performance by permuting their tensor accordingly.
A discussion of this and a potential hint in verbose mode is tracked in
[#168](https://github.com/dahong67/GCPDecompositions.jl/issues/168).

### SparseArrayCOO sums duplicate indices

When constructing a `SparseArrayCOO`, entries with the same index tuple are summed
rather than overwritten. This is consistent with the intended use for accumulating
stochastic gradient contributions.

## Known Limitations

- **Float64 only for GCP algorithms.** Both `GCP_LBFGSB` (due to the underlying
  Fortran LBFGSB library) and `GCP_Adam` work with `Float64` factor matrices only.
- **Constraint support is narrow.** Only `LowerBoundConstraint` is supported by the
  optimisers, and only domains of ``\mathbb{R}`` or ``[0,\infty)`` are handled by
  `default_gcp_constraints`.
- **Sampler coverage for GCP-Adam is incomplete.** For example, `UniformGCPSampler` is
  not implemented for `SparseArrayCOO`, and `StratifiedGCPSampler` is not implemented
  for dense `Array`. See [#172](https://github.com/dahong67/GCPDecompositions.jl/issues/172)
  for the full list.
- **Zero sampling uses rejection.** `StratifiedGCPSampler` samples true zeros via a
  rejection loop, which can be slow when the tensor is very dense.
- **No GPU support in mainline.** A draft PR exists ([#37](https://github.com/dahong67/GCPDecompositions.jl/pull/37))
  but it has not been merged.
- **CPD views may not work correctly.** It has not been fully tested that all `CPD`
  operations behave correctly when the weight vector or factor matrices are views into a
  larger array ([#164](https://github.com/dahong67/GCPDecompositions.jl/issues/164)).
- **Non-public names in use.** Some internal Julia names that are not part of the
  public API are currently used; auditing and eliminating these is tracked in
  [#160](https://github.com/dahong67/GCPDecompositions.jl/issues/160).

## Roadmap

Items are grouped roughly by time horizon and labelled with their status:

| Label | Meaning |
|---|---|
| 🚧 In progress | Open PR with active work |
| 📋 Planned | Agreed-upon open issue |
| 💡 Exploratory | Idea under discussion, no firm commitment |

### Short-term

- 📋 Complete remaining GCP-Adam items: missing sampler methods, more efficient
  implementations, without-replacement samplers, auto-selection of hyperparameters
  ([#172](https://github.com/dahong67/GCPDecompositions.jl/issues/172))
- 📋 Introduce `AutoAlgorithm()` as an explicit, user-passable algorithm type
  ([#170](https://github.com/dahong67/GCPDecompositions.jl/issues/170))
- 📋 Audit and tighten all method signatures
  ([#159](https://github.com/dahong67/GCPDecompositions.jl/issues/159))
- 📋 Eliminate use of non-public Julia names
  ([#160](https://github.com/dahong67/GCPDecompositions.jl/issues/160))
- 📋 Audit size-product arithmetic (consider `CheckedSizeProduct.jl`)
  ([#181](https://github.com/dahong67/GCPDecompositions.jl/issues/181))
- 📋 Document design choice about not defining `rand(::CPD)`
  ([#169](https://github.com/dahong67/GCPDecompositions.jl/issues/169))
- 📋 Document and implement performance hint for mode permutation
  ([#168](https://github.com/dahong67/GCPDecompositions.jl/issues/168))
- 📋 Set up release automation with release-please
  ([#180](https://github.com/dahong67/GCPDecompositions.jl/issues/180))
- 📋 Switch from CompatHelper to dependabot
  ([#163](https://github.com/dahong67/GCPDecompositions.jl/issues/163))

### Medium-term

- 🚧 **Symmetric CPD** (`SymCPD` type + symmetric GCP decompositions) — milestone 1.0
  ([PR #70](https://github.com/dahong67/GCPDecompositions.jl/pull/70))
- 🚧 **Restructured `AbstractLoss`** with support for non-entrywise losses and
  regularisers ([PR #96](https://github.com/dahong67/GCPDecompositions.jl/pull/96))
- 🚧 **Generalized Multilinear Models (GMLM)**
  ([PR #146](https://github.com/dahong67/GCPDecompositions.jl/pull/146))
- 🚧 **Faster MTTKRPS algorithm** ([PR #40](https://github.com/dahong67/GCPDecompositions.jl/pull/40))
- 🚧 **CUDA support** for ALS ([PR #37](https://github.com/dahong67/GCPDecompositions.jl/pull/37))
- 📋 More built-in loss functions ([#161](https://github.com/dahong67/GCPDecompositions.jl/issues/161))
- 📋 Decomposition plotting function
  ([PR #59](https://github.com/dahong67/GCPDecompositions.jl/pull/59))
- 📋 Python and R calling interfaces
  ([#151](https://github.com/dahong67/GCPDecompositions.jl/issues/151))
- 📋 More demos (traffic, fishing, PCA-inspired)
  ([#153](https://github.com/dahong67/GCPDecompositions.jl/issues/153),
   [#177](https://github.com/dahong67/GCPDecompositions.jl/issues/177),
   [#152](https://github.com/dahong67/GCPDecompositions.jl/issues/152))
- 📋 Interface more directly with the underlying LBFGSB Fortran library
  ([#158](https://github.com/dahong67/GCPDecompositions.jl/issues/158))
- 📋 Fix demo versioning so they track the corresponding package release
  ([#162](https://github.com/dahong67/GCPDecompositions.jl/issues/162))

### Research-track / exploratory

- 💡 **Blocked MTTKRP** for cache efficiency
  ([#167](https://github.com/dahong67/GCPDecompositions.jl/issues/167))
- 💡 **Matrix-free dense MTTKRP**
  ([#166](https://github.com/dahong67/GCPDecompositions.jl/issues/166))
- 💡 Investigate other optimisers: muon, PALM (for non-smooth regularisation)
  ([#175](https://github.com/dahong67/GCPDecompositions.jl/issues/175))
- 💡 Integration with [LowRankModels.jl](https://github.com/madeleineudell/LowRankModels.jl)
  ([#173](https://github.com/dahong67/GCPDecompositions.jl/issues/173))
- 💡 Coupled matrix-tensor factorisation / tensor data fusion (CMTF, AO-ADMM framework)
  ([#157](https://github.com/dahong67/GCPDecompositions.jl/issues/157))
- 💡 Terminal user interface (TUI)
  ([#176](https://github.com/dahong67/GCPDecompositions.jl/issues/176))
- 💡 Interactive 3D plots in the docs
  ([#171](https://github.com/dahong67/GCPDecompositions.jl/issues/171))
- 💡 Reference and integrate ideas from existing tools (Vagelis group, Copenhagen group,
  HOTTBOX, scikit-learn decompositions)
  ([#174](https://github.com/dahong67/GCPDecompositions.jl/issues/174))

## Sources and Freshness

**Last reviewed:** 2026-07-06

**Sources consulted:**

- Codebase: `master` branch (all `src/` and `test/` files, `docs/` structure)
- Open issues: #151, #157–161, #163–164, #166–177, #180–181
- Open PRs: #37, #40, #59, #70, #96, #146
- Closed PRs: #67, #90, #91, #98, #102–106, #122–124, #136, #138, #143, #147, #154, #156

**Update triggers:** This page should be refreshed after

- Merging a significant PR (new algorithm, new loss type, new data format)
- Closing a milestone
- A release tag

**Known gaps:** The following items lack clear canonical references and need follow-up:

- The exact rationale for choosing `CP_FastALS` as the default over `CP_ALS` once
  `CP_FastALS` was introduced (implementation credit: PR #106 / earlier work)
- Design constraints on the LBFGSB interface that force Float64
- The original motivation for stratified vs. semistratified sampling distinction
