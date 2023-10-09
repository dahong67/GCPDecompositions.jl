"Generalized CP Decomposition module. Provides approximate CP tensor decomposition with respect to general losses."
module GCPDecompositions

# Imports
import Base: require_one_based_indexing
import Base: ndims, size, show, summary
import Base: getindex
import ForwardDiff
using LinearAlgebra: mul!, rmul!, Diagonal
using LBFGSB: lbfgsb

# Exports
export CPD
export ncomponents
export gcp

include("type-cpd.jl")
include("gcp-opt.jl")

if !isdefined(Base, :get_extension)
    include("ext/LossFunctinonsExt.jl")
  end

end
