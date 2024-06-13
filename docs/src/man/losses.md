# Loss functions

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

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
