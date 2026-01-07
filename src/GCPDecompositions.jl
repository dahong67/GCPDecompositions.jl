"""
Generalized CP Decomposition module.
Provides approximate CP tensor decompositions with respect to general losses.
"""
module GCPDecompositions

# Imports
using Base.Order: Ordering, Reverse
using LinearAlgebra: Diagonal, LinearAlgebra, mul!, norm
using IntervalSets: Interval
using Random: default_rng
using SparseArrays: sparse

# Tensor Kernels
include("kernels/khatrirao.jl")
include("kernels/mttkrp.jl")
include("kernels/mttkrps.jl")
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!

# Tensor Types
include("types/sparsearraycoo.jl")
include("types/cpd.jl")
export SparseArrayCOO, numstored
export CPD,
    ncomps,
    normalizecomps,
    normalizecomps!,
    permutecomps,
    permutecomps!,
    sortcomps,
    sortcomps!

# Exports
export gcp, default_gcp_constraints, default_gcp_algorithm, default_gcp_init
export GCPLosses, GCPConstraints, GCPAlgorithms

include("gcp-losses.jl")
include("gcp-constraints.jl")
include("gcp-algorithms.jl")
include("api-gcp.jl")

end
