# Constraints

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

See the [Developer Overview](@ref) for notes on the current constraint design and
which algorithm–constraint combinations are supported.

```@docs
AbstractConstraint
satisfies
project!
```

```@autodocs
Modules = [GCPDecompositions]
Filter = t -> t in subtypes(AbstractConstraint)
```
