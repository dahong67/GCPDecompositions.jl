# API Reference

## Main functions and types

```@docs
GCPDecompositions
gcp
CPD
ncomponents
GCPDecompositions.default_constraints
GCPDecompositions.default_algorithm
GCPDecompositions.default_init
```

## Loss functions

```@docs
GCPLosses
GCPLosses.AbstractLoss
```

```@autodocs
Modules = [GCPLosses]
Filter = t -> t in subtypes(GCPLosses.AbstractLoss)
```

```@docs
GCPLosses.value
GCPLosses.deriv
GCPLosses.domain
GCPLosses.objective
GCPLosses.grad_U!
```

## Constraints

```@docs
GCPConstraints
GCPConstraints.AbstractConstraint
```

```@autodocs
Modules = [GCPConstraints]
Filter = t -> t in subtypes(GCPConstraints.AbstractConstraint)
```

## Algorithms

```@docs
GCPAlgorithms
GCPAlgorithms.AbstractAlgorithm
```

```@autodocs
Modules = [GCPAlgorithms]
Filter = t -> t in subtypes(GCPAlgorithms.AbstractAlgorithm)
```

## Tensor Kernels

```@docs
GCPDecompositions.TensorKernels
GCPDecompositions.TensorKernels.khatrirao
GCPDecompositions.TensorKernels.khatrirao!
GCPDecompositions.TensorKernels.mttkrp
GCPDecompositions.TensorKernels.mttkrp!
GCPDecompositions.TensorKernels.create_mttkrp_buffer
GCPDecompositions.TensorKernels.mttkrps
GCPDecompositions.TensorKernels.mttkrps!
```

## Private functions

```@docs
GCPAlgorithms._gcp
GCPDecompositions.TensorKernels._checked_khatrirao_dims
GCPDecompositions.TensorKernels._checked_mttkrp_dims
GCPDecompositions.TensorKernels._checked_mttkrps_dims
```
