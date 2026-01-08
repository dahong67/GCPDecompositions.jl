## Stochastic GCP objective and gradient functions: Uniform sampler

"""
    UniformGCPSampler(numsamples::Int)

Uniform sampling of `numsamples` entries with replacement.
"""
struct UniformGCPSampler <: AbstractGCPSampler
    numsamples::Int
end

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    sampler::UniformGCPSampler,
) where {T,TX,N}
    return gcp_stoch_objective(rng, M, X, loss, GCPSampleOnce(X, sampler))
end

GCPSampleOnce(X::Array, sampler::UniformGCPSampler) =
    GCPSampleOnce(sampler, Vector{NTuple{ndims(X),Int}}())
function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    (; sampler, cache)::GCPSampleOnce{<:UniformGCPSampler},
) where {T,TX,N}
    n, ω, s, inds = size(X), length(X), sampler.numsamples, cache
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
    sampler::UniformGCPSampler,
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
