## GCP Algorithms

"""
Algorithms for Generalized CP Decomposition.
"""
module GCPAlgorithms

using ..GCPDecompositions
using ..GCPLosses: value, deriv, domain
using ..GCPConstraints: project!
using ..TensorKernels: create_mttkrp_buffer, mttkrp!, mttkrps!
using ..TensorKernels: khatrirao!, khatrirao
using IntervalSets: Interval
using LinearAlgebra: lu!, mul!, norm, rdiv!, rmul!, Diagonal
using LBFGSB: lbfgsb
using Random: AbstractRNG, default_rng
using StatsBase: sample!

# Objective and gradient functions

"""
    gcp_objective(M::CPD, X::AbstractArray, loss)

Compute the GCP objective function for the model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function gcp_objective(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

"""
    gcp_grad_U!(GU, M::CPD, X::AbstractArray, loss)

Compute the GCP gradient with respect to the factor matrices `U = (U[1],...,U[N])`
for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU = (GU[1],...,GU[N])`.
"""
function gcp_grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]
    mttkrps!(GU, Y, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end

# Stochastic objective and gradient functions: Abstract types and functions

"""
    AbstractSampler

Abstract type for samplers to use in stochastic evaluation
of the objective and gradients.

Concrete types `ConcreteSampler <: AbstractSampler` should implement

+ `gcp_stoch_objective(rng, M, X, loss, sampler::ConcreteSampler)`
+ `gcp_stoch_grad_U!(rng, GU, M, X, loss, sampler::ConcreteSampler)`
"""
abstract type AbstractSampler end

"""
    gcp_stoch_objective([rng=default_rng()], M::CPD, X::AbstractArray, loss, sampler)

Compute stochastic estimate of the GCP objective function for the
model tensor `M`, data tensor `X`, and loss function `loss`
using the sampler `sampler` with random number generator `rng`.
"""
gcp_stoch_objective(M::CPD, X::AbstractArray, loss, sampler) =
    gcp_stoch_objective(default_rng(), M, X, loss, sampler)

"""
    gcp_stoch_grad_U!([rng=default_rng()], GU, M::CPD, X::AbstractArray, loss, sampler)

Compute stochastic estimate of the GCP gradient with respect to the
factor matrices `U = (U[1],...,U[N])` for the model tensor `M`,
data tensor `X`, and loss function `loss` using the sampler `sampler`
with random number generator `rng`, and store the result in `GU = (GU[1],...,GU[N])`.
"""
gcp_stoch_grad_U!(GU, M::CPD, X::AbstractArray, loss, sampler) =
    gcp_stoch_grad_U!(default_rng(), GU, M::CPD, X::AbstractArray, loss, sampler)

"""
    SampleOnce(X::AbstractArray, sampler::AbstractSampler)

Wrapped sampler that samples entries from `X` using `sampler` only
the first time, then reuses the same indices every time after that.
For use with `gcp_stoch_objective`.

The internal field `cache` stores cached values - the particular choice
of what is stored is an implementation detail defined by each `sampler`.
"""
struct SampleOnce{S<:AbstractSampler,C} <: AbstractSampler
    sampler::S
    cache::C
end
Base.iterate(wrapped::SampleOnce) = (wrapped.sampler, Val(:cache))
Base.iterate(wrapped::SampleOnce, ::Val{:cache}) = (wrapped.cache, Val(:done))
Base.iterate(::SampleOnce, ::Val{:done}) = nothing

# Stochastic objective and gradient functions: Uniform sampler

"""
    UniformSampler(numsamples::Int)

Uniform sampling of `numsamples` entries with replacement.
"""
struct UniformSampler <: AbstractSampler
    numsamples::Int
end

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    sampler::UniformSampler,
) where {T,TX,N}
    return gcp_stoch_objective(rng, M, X, loss, SampleOnce(X, sampler))
end

SampleOnce(X::Array, sampler::UniformSampler) =
    SampleOnce(sampler, Vector{NTuple{ndims(X),Int}}())
function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    (sampler, inds)::SampleOnce{<:UniformSampler},
) where {T,TX,N}
    n, ω, s = size(X), length(X), sampler.numsamples
    if isempty(inds)
        sample!(rng, CartesianIndices(n), resize!(inds, s))
    end
    return sum(
        (ω / s) * value(loss, X[CartesianIndex(I)], M[CartesianIndex(I)]) for
        I in inds if !ismissing(X[CartesianIndex(I)])
    )
end

function gcp_stoch_grad_U!(
    rng::AbstractRNG,
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    sampler::UniformSampler,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    n, ω, s = size(X), length(X), sampler.numsamples
    inds = sample!(rng, CartesianIndices(n), Vector{NTuple{ndims(X),Int}}(undef, s))
    vals = [
        ismissing(X[CartesianIndex(I)]) ? zero(nonmissingtype(eltype(X))) :
        (ω / s) * deriv(loss, X[CartesianIndex(I)], M[CartesianIndex(I)]) for I in inds
    ]
    Yt = SparseArrayCOO(n, inds, vals)
    mttkrps!(GU, Yt, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end

# Abstract algorithm type and associated functions

"""
    AbstractAlgorithm

Abstract type for GCP algorithms.

Concrete types `ConcreteAlgorithm <: AbstractAlgorithm` should implement
`_gcp!(rng, M, X, loss, constraints, algorithm::ConcreteAlgorithm)`
that modifies the initialization `M` and returns the modified version.
"""
abstract type AbstractAlgorithm end

"""
    _gcp!(rng, M, X, loss, constraints, algorithm)

Internal function to compute an approximate rank-`r` CP decomposition
of the data tensor `X` with respect to the loss function `loss` and the
constraints `constraints` using the algorithm `algorithm` with random
numbers generated by `rng`, modifying the initialization `M` and
returning the modified version.
"""
function _gcp! end

# Built-in algorithms

include("gcp-algorithms/adam.jl")
include("gcp-algorithms/lbfgsb.jl")
include("gcp-algorithms/als.jl")
include("gcp-algorithms/fastals.jl")

end
