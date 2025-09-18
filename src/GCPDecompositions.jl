"""
Generalized CP Decomposition module.
Provides approximate CP tensor decomposition with respect to general losses.
"""
module GCPDecompositions

# Imports
import Base: ndims, size, show, summary
import Base: getindex
import Base: permutedims
import Base: AbstractArray, Array
import LinearAlgebra: norm
using Base.Order: Ordering, Reverse
using IntervalSets: Interval
using Random: default_rng

# Import sparse array module and export data type
include("SparseArrayCOOs.jl")
using .SparseArrayCOOs
export SparseArrayCOO

# Exports
export CPD
export ncomps,
    normalizecomps, normalizecomps!, permutecomps, permutecomps!, sortcomps, sortcomps!
export gcp, default_gcp_constraints, default_gcp_algorithm, default_gcp_init
export GCPLosses, GCPConstraints, GCPAlgorithms

include("tensor-kernels.jl")
include("cpd.jl")
include("gcp-losses.jl")
include("gcp-constraints.jl")
include("gcp-algorithms.jl")
include("api-gcp.jl")

end
