module LossFunctionsExt

using GCPDecompositions, LossFunctions
import GCPDecompositions: _factor_matrix_lower_bound

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

GCPDecompositions.gcp(X::Array, r, loss::SupportedLosses) = GCPDecompositions._gcp(
    X,
    r,
    (x, m) -> loss(m, x),
    (x, m) -> LossFunctions.deriv(loss, m, x),
    _factor_matrix_lower_bound(loss),
    (;),
)

_factor_matrix_lower_bound(::LossFunctions.DistanceLoss) = -Inf
_factor_matrix_lower_bound(::LossFunctions.MarginLoss)   = -Inf

end
