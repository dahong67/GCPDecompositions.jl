## Stochastic GCP objective and gradient functions: Semistratified sampler

"""
    SemistratifiedGCPSampler(num_nonzeros::Int, num_zeros::Int)

Semistratified sampling of `num_nonzeros` nonzero entries
and `num_zeros` assumed "zero" entries with replacement.
For `SparseArrayCOO` tensors, stored entries are all treated as nonzero.
"""
struct SemistratifiedGCPSampler <: AbstractGCPSampler
    num_nonzeros::Int
    num_zeros::Int
end

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    sampler::SemistratifiedGCPSampler,
) where {T,TX,TI,N}
    return gcp_stoch_objective(rng, M, X, loss, GCPSampleOnce(X, sampler))
end

GCPSampleOnce(
    ::SparseArrayCOO{TX,TI,N},
    sampler::SemistratifiedGCPSampler,
) where {TX,TI,N} =
    GCPSampleOnce(sampler, (; nzptrs = Vector{Int}(), azinds = Vector{NTuple{N,TI}}()))
function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    (; sampler, cache)::GCPSampleOnce{<:SemistratifiedGCPSampler},
) where {T,TX,TI,N}
    # Extract parameters
    n, η, ω = size(X), numstored(X), length(X)
    p, q = sampler.num_nonzeros, sampler.num_zeros
    (; nzptrs, azinds) = cache

    # Sample entries if not already done
    if isempty(nzptrs) || isempty(azinds)
        # Sample nonzeros
        sample!(rng, 1:η, resize!(nzptrs, p))

        # Sample assumed zeros
        sample!(rng, CartesianIndices(n), resize!(azinds, q))
    end

    # Compute and return estimated objective function value
    nzsum = sum(nzptrs) do ptr
        MI = M[CartesianIndex(X.inds[ptr])]
        return (η / p) * (value(loss, X.vals[ptr], MI) - value(loss, zero(TX), MI))
    end
    zsum = sum((ω / q) * value(loss, zero(TX), M[CartesianIndex(ind)]) for ind in azinds)
    return nzsum + zsum
end

function gcp_stoch_grad_U!(
    rng::AbstractRNG,
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    sampler::SemistratifiedGCPSampler,
) where {T,TX,TI,N,TGU<:AbstractMatrix{T}}
    # Extract parameters
    n, η, ω = size(X), numstored(X), length(X)
    p, q = sampler.num_nonzeros, sampler.num_zeros

    # Sample nonzeros
    nzptrs = sample!(rng, 1:η, Vector{Int}(undef, p))

    # Sample assumed zeros
    azinds = sample!(rng, CartesianIndices(n), Vector{NTuple{N,TI}}(undef, q))

    # Form sparse stochastic derivative tensor
    inds = [X.inds[nzptrs]; azinds]
    nzvals = map(nzptrs) do ptr
        MI = M[CartesianIndex(X.inds[ptr])]
        return (η / p) * (deriv(loss, X.vals[ptr], MI) - deriv(loss, zero(TX), MI))
    end
    azvals = [(ω / q) * deriv(loss, zero(TX), M[CartesianIndex(ind)]) for ind in azinds]
    vals = [nzvals; azvals]
    Yt = SparseArrayCOO(n, inds, vals)
    mttkrps!(GU, Yt, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end
