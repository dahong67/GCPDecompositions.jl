# Algorithms

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

```@docs
AbstractAlgorithm
gcp_objective
gcp_grad_U!
gcp_stoch_objective
gcp_stoch_grad_U!
AbstractSampler
SampleOnce
UniformSampler
StratifiedSampler
SemistratifiedSampler
```

```@autodocs
Modules = [GCPDecompositions]
Filter = t -> t in subtypes(AbstractAlgorithm) || (t isa Function && t != GCPDecompositions._gcp!)
```
