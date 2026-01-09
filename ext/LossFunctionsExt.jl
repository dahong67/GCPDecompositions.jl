module LossFunctionsExt

using GCPDecompositions
using LossFunctions: LossFunctions
using IntervalSets: Interval

const SupportedLosses = Union{LossFunctions.DistanceLoss,LossFunctions.MarginLoss}

Base.convert(::Type{AbstractLoss}, loss::SupportedLosses) = WrappedLoss(loss, LossFunctions)

GCPDecompositions.value(loss::SupportedLosses, x, m) = loss(m, x)
GCPDecompositions.deriv(loss::SupportedLosses, x, m) = LossFunctions.deriv(loss, m, x)
GCPDecompositions.domain(::SupportedLosses)          = Interval(-Inf, Inf)

end
