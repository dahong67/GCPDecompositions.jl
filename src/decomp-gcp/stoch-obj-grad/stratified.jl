## Stochastic GCP objective and gradient functions: Stratified sampler

"""
    StratifiedGCPSampler(num_nonzeros::Int, num_zeros::Int)

Stratified sampling of `num_nonzeros` nonzero entries
and `num_zeros` zero entries with replacement.
For `SparseArrayCOO` tensors, stored entries are all treated as nonzero.
"""
struct StratifiedGCPSampler <: AbstractGCPSampler
    num_nonzeros::Int
    num_zeros::Int
end

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    sampler::StratifiedGCPSampler,
) where {T,TX,TI,N}
    return gcp_stoch_objective(rng, M, X, loss, GCPSampleOnce(X, sampler))
end

GCPSampleOnce(::SparseArrayCOO{TX,TI,N}, sampler::StratifiedGCPSampler) where {TX,TI,N} =
    GCPSampleOnce(sampler, (; nzptrs = Vector{Int}(), zinds = Vector{NTuple{N,TI}}()))
function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    (; sampler, cache)::GCPSampleOnce{<:StratifiedGCPSampler},
) where {T,TX,TI,N}
    # Extract parameters
    n, η, ζ = size(X), numstored(X), length(X) - numstored(X)
    p, q = sampler.num_nonzeros, sampler.num_zeros
    (; nzptrs, zinds) = cache

    # Sample entries if not already done
    if isempty(nzptrs) || isempty(zinds)
        # Sample nonzeros
        sample!(rng, 1:η, resize!(nzptrs, p))

        # Sample zeros (naive rejection sampling loop for now)
        while length(zinds) < q
            ind = convert(NTuple{N,TI}, sample(rng, CartesianIndices(n)))
            if !(ind in X.inds)
                push!(zinds, ind)
            end
        end
    end

    # Compute and return estimated objective function value
    nzsum = sum(
        (η / p) * value(loss, X.vals[ptr], M[CartesianIndex(X.inds[ptr])]) for
        ptr in nzptrs
    )
    zsum = sum((ζ / q) * value(loss, zero(TX), M[CartesianIndex(ind)]) for ind in zinds)
    return nzsum + zsum
end

function gcp_stoch_grad_U!(
    rng::AbstractRNG,
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::SparseArrayCOO{TX,TI,N},
    loss,
    sampler::StratifiedGCPSampler,
) where {T,TX,TI,N,TGU<:AbstractMatrix{T}}
    # Extract parameters
    n, η, ζ = size(X), numstored(X), length(X) - numstored(X)
    p, q = sampler.num_nonzeros, sampler.num_zeros

    # Sample nonzeros
    nzptrs = sample!(rng, 1:η, Vector{Int}(undef, p))

    # Sample zeros (naive rejection sampling loop for now)
    zinds = Vector{NTuple{N,TI}}()
    while length(zinds) < q
        ind = convert(NTuple{N,TI}, sample(rng, CartesianIndices(n)))
        if !(ind in X.inds)
            push!(zinds, ind)
        end
    end

    # Form sparse stochastic derivative tensor
    inds = [X.inds[nzptrs]; zinds]
    nzvals = map(nzptrs) do ptr
        return (η / p) * deriv(loss, X.vals[ptr], M[CartesianIndex(X.inds[ptr])])
    end
    zvals = [(ζ / q) * deriv(loss, zero(TX), M[CartesianIndex(ind)]) for ind in zinds]
    vals = [nzvals; zvals]
    Yt = SparseArrayCOO(n, inds, vals)
    mttkrps!(GU, Yt, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end
