module LossFunctionsExt

using GCPDecompositions, LossFunctions
using IntervalSets

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

GCPDecompositions.value(loss::SupportedLosses, x, m)   = loss(m, x)
GCPDecompositions.deriv(loss::SupportedLosses, x, m)   = LossFunctions.deriv(loss, m, x)
GCPDecompositions.domain(::LossFunctions.DistanceLoss) = Interval(-Inf, Inf)
GCPDecompositions.domain(::LossFunctions.MarginLoss)   = Interval(-Inf, Inf)

end
