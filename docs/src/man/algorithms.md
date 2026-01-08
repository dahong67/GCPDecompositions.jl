# Algorithms

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

```@docs
AbstractGCPAlgorithm
gcp_objective
gcp_grad_U!
gcp_stoch_objective
gcp_stoch_grad_U!
AbstractGCPSampler
GCPSampleOnce
UniformGCPSampler
StratifiedGCPSampler
SemistratifiedGCPSampler
```

```@autodocs
Modules = [GCPDecompositions]
Filter = t -> t in subtypes(AbstractGCPAlgorithm)
```

```@docs
GCPDecompositions.CP_FastALS_iter!
```
