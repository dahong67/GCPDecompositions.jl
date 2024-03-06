module LossFunctionsExt

using GCPDecompositions, LossFunctions
using IntervalSets

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

GCPLosses.value(loss::SupportedLosses, x, m)   = loss(m, x)
GCPLosses.deriv(loss::SupportedLosses, x, m)   = LossFunctions.deriv(loss, m, x)
GCPLosses.domain(::LossFunctions.DistanceLoss) = Interval(-Inf, Inf)
GCPLosses.domain(::LossFunctions.MarginLoss)   = Interval(-Inf, Inf)

end
