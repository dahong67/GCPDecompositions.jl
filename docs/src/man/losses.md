# Loss functions

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
