"""
Generalized CP Decomposition module. Provides approximate CP tensor decomposition with respect to general losses.
"""
module GCPDecompositions

# Imports
import Base: require_one_based_indexing
import Base: ndims, size, show, summary
import Base: getindex
import ForwardDiff
using IntervalSets
using LinearAlgebra: mul!, rmul!, Diagonal
using LBFGSB: lbfgsb

# Exports
export CPD
export ncomponents
export gcp
export AbstractLoss, LeastSquaresLoss, NonnegativeLeastSquaresLoss, PoissonLoss, UserDefinedLoss

include("type-cpd.jl")
include("type-losses.jl")
include("gcp-opt.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LossFunctionsExt.jl")
end

end
