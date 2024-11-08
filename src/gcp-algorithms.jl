## Algorithm types

"""
Algorithms for Generalized CP Decomposition.
"""
module GCPAlgorithms

using ..GCPDecompositions
using ..TensorKernels: create_mttkrp_buffer, mttkrp!
using ..TensorKernels: khatrirao!, khatrirao
using ..TensorKernels: checksym
using IntervalSets: Interval
using LinearAlgebra: lu!, mul!, norm, rdiv!
using LBFGSB: lbfgsb

"""
    AbstractAlgorithm

Abstract type for GCP algorithms.

Concrete types `ConcreteAlgorithm <: AbstractAlgorithm` should implement
`_gcp(X, r, loss, constraints, algorithm::ConcreteAlgorithm)`
that returns a `CPD`.
"""
abstract type AbstractAlgorithm end

"""
    _gcp(X, r, loss, constraints, algorithm)

Internal function to compute an approximate rank-`r` CP decomposition
of the tensor `X` with respect to the loss function `loss` and the
constraints `constraints` using the algorithm `algorithm`, returning
a `CPD` object.
"""
function _gcp end

include("gcp-algorithms/lbfgsb.jl")
include("gcp-algorithms/als.jl")
include("gcp-algorithms/fastals.jl")

end
