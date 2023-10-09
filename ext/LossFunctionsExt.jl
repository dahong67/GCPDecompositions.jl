module LossFunctionsExt

using GCPDecompositions
using LossFunctions

"""
    gcp(X::Array, r, loss::LossFunctions.DistanceLoss, lower]) -> CPD

Compute an approximate rank-`r` CP decomposition of the tensor `X`
with respect to the loss function `loss` and return a `CPD` object.

# Inputs
+ `X` : multi-dimensional tensor/array to approximate/decompose
+ `r` : number of components for the CPD
+ `loss` : loss function from LossFunctions.jl
+ `lower` : lower bound for factor matrix entries, `default = -Inf`
"""
GCPDecompositions.gcp(X::Array, r, loss::LossFunctions.DistanceLoss, lower=-Inf) =
    GCPDecompositions._gcp(X, r, (x, m) -> loss(x, m), (x, m) -> LossFunctions.deriv(loss, m, x), lower, (;))

end