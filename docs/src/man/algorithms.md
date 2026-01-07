# Algorithms

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

```@docs
GCPAlgorithms
GCPAlgorithms.AbstractAlgorithm
GCPAlgorithms.gcp_objective
GCPAlgorithms.gcp_grad_U!
GCPAlgorithms.gcp_stoch_objective
GCPAlgorithms.gcp_stoch_grad_U!
GCPAlgorithms.AbstractSampler
GCPAlgorithms.SampleOnce
GCPAlgorithms.UniformSampler
GCPAlgorithms.StratifiedSampler
GCPAlgorithms.SemistratifiedSampler
```

```@autodocs
Modules = [GCPAlgorithms]
Filter = t -> t in subtypes(GCPAlgorithms.AbstractAlgorithm) || (t isa Function && t != GCPAlgorithms._gcp!)
```
