module LossFunctinonsExt

using GCPDecompositions

gcp(X::Array, r, loss::LossFunctions.DistanceLoss, lower=-Inf) =
    _gcp(X, r, (x, m) -> loss(x, m), (x, m) -> LossFunctions.deriv(loss, m, x), lower, (;))

end