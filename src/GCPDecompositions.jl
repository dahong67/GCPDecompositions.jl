"""
Generalized CP Decomposition module. Provides approximate CP tensor decomposition with respect to general losses.
"""
module GCPDecompositions

# Imports
import Base: require_one_based_indexing
import Base: ndims, size, show, summary
import Base: getindex
import ForwardDiff
using Compat
using IntervalSets
using LinearAlgebra: mul!, rmul!, Diagonal, norm
using LBFGSB: lbfgsb

# Exports
export CPD
export ncomponents
export gcp
export AbstractLoss,
    LeastSquaresLoss,
    NonnegativeLeastSquaresLoss,
    PoissonLoss,
    PoissonLogLoss,
    GammaLoss,
    RayleighLoss,
    BernoulliOddsLoss,
    BernoulliLogitLoss,
    NegativeBinomialOddsLoss,
    HuberLoss,
    BetaDivergenceLoss,
    UserDefinedLoss
export GCPConstraints, GCPAlgorithms

include("type-cpd.jl")
include("type-losses.jl")
include("type-constraints.jl")
include("type-algorithms.jl")
include("kernels.jl")
include("gcp-opt.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LossFunctionsExt.jl")
end

end
