# Loss functions

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

See the [Developer Overview](@ref) for notes on the current design of `AbstractLoss`
and planned extensions.

```@docs
AbstractLoss
value
deriv
domain
```

```@autodocs
Modules = [GCPDecompositions]
Filter = t -> t in subtypes(AbstractLoss)
```
