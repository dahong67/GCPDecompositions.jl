## CP decomposition type

"""
    SymCPD

Tensor decomposition type for the symmetric canonical polyadic decomposition (Sym-CPD)
of a tensor (i.e., a multi-dimensional array) `A`.

If `M::SymCPD` is the decomposition object,
the weights `λ` and the factor matrices `U = (U[1],...,U[K])`
can be obtained via `M.λ` and `M.U`,
such that --------------->change `A = Σ_j λ[j] U[1][:,j] ∘ ⋯ ∘ U[N][:,j]`.
"""
# I = index classes?
# k = number of index classes (k<= N, k=1 for fully symmetric, k=N for no symmetry)
struct SymCPD{T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{K,TU}
    S::NTuple{N,Int}
    function SymCPD{T,N,K,Tλ,TU}(λ, U, S) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        Base.require_one_based_indexing(λ, U...)
        for k in Base.OneTo(K)
            size(U[k], 2) == length(λ) || throw(
                DimensionMismatch(
                    "U[$k] has dimensions $(size(U[k])) but λ has length $(length(λ))",
                ),
            )
            S[k] <= K || throw(
                DimensionMismatch(
                    "Mode $(k) is mapped to an index class > than the given $(K)",
                ),
            )
        end
        return new{T,N,K,Tλ,TU}(λ, U, S)
    end
end
SymCPD(λ::Tλ, U::NTuple{K,TU}, S::NTuple{N,Int}) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    SymCPD{T,N,K,Tλ,TU}(λ, U, S)

"""
    ncomps(M::SymCPD)

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomps(M::SymCPD) = length(M.λ)
ndims(M::SymCPD) = length(M.S)

size(M::SymCPD{T,N,K}, dim::Integer) where {T,N,K} = dim <= N ? size(M.U[S[dim]], 1) : 1
size(M::SymCPD{T,N,K}) where {T,N,K} = ntuple(d -> size(M, d), N)


function getindex(M::SymCPD{T,N,K}, I::Vararg{Int,N}) where {T,N,K}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    val = zero(eltype(T))
    for j in Base.OneTo(ncomps(M))
        val += M.λ[j] * prod(M.U[S[k]][I[k], j] for k in Base.OneTo(ndims(M)))
    end
    return val
end
getindex(M::SymCPD{T,N,K}, I::CartesianIndex{N}) where {T,N,K} = getindex(M, Tuple(I)...)