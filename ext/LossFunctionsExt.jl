module LossFunctionsExt

using GCPDecompositions
using LossFunctions

GCPDecompositions.gcp(X::Array, r, loss::LossFunctions.DistanceLoss, lower=-Inf) =
    GCPDecompositions._gcp(X, r, (x, m) -> loss(x, m), (x, m) -> LossFunctions.deriv(loss, m, x), lower, (;))

end