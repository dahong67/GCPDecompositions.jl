"""
Generalized CP Decomposition module. Provides approximate CP tensor decomposition with respect to general losses.
"""
module GCPDecompositions

# Imports
import Base: ndims, size, show, summary
import Base: getindex
import LinearAlgebra: norm
using IntervalSets: Interval

# Exports
export CPD
export ncomponents
export gcp
export GCPLosses, GCPConstraints, GCPAlgorithms

include("tensor-kernels.jl")
include("cpd.jl")
include("gcp-losses.jl")
include("gcp-constraints.jl")
include("gcp-algorithms.jl")
include("gcp-opt.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LossFunctionsExt.jl")
end

end
