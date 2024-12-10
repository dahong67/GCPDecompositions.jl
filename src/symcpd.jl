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
    function SymCPD{T,N,K,Tλ,TU}(
        λ,
        U,
        S,
    ) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        Base.require_one_based_indexing(λ, U...)
        if K > 0
            minimum([S...]) == 1 && maximum([S...]) <= K || throw(
                DimensionMismatch("Symmetric Groups must be numbered 1,2,... (max N)"),
            )
        end
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
SymCPD(
    λ::Tλ,
    U::NTuple{K,TU},
    S::NTuple{N,Int},
) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} = SymCPD{T,N,K,Tλ,TU}(λ, U, S)
SymCPD(M_cpd::CPD) = SymCPD(M_cpd)

"""
    ncomps(M::SymCPD)

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomps(M::SymCPD) = length(M.λ)
ndims(M::SymCPD) = length(M.S)

size(
    M::SymCPD{T,N,K,Tλ,TU},
    dim::Integer,
) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    dim <= N ? size(M.U[M.S[dim]], 1) : 1
size(M::SymCPD{T,N,K,Tλ,TU}) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    ntuple(d -> size(M, d), N)

"""
    ngroups(M::SymCPD)

Return the number of symmetric groups in `M`.
"""
ngroups(M::SymCPD) = length(M.U)

function getindex(
    M::SymCPD{T,N,K,Tλ,TU},
    I::Vararg{Int,N},
) where {T,N,K,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    val = zero(eltype(T))
    for j in Base.OneTo(ncomps(M))
        val += M.λ[j] * prod(M.U[M.S[k]][I[k], j] for k in Base.OneTo(ndims(M)))
    end
    return val
end
getindex(M::SymCPD{T,N,K}, I::CartesianIndex{N}) where {T,N,K} = getindex(M, Tuple(I)...)

AbstractArray(A::SymCPD) = reshape(
    TensorKernels.khatrirao(reverse([A.U[A.S[k]] for k in 1:ndims(A)])...) * A.λ,
    size(A),
)
Array(A::SymCPD) = Array(AbstractArray(A))

norm(M::SymCPD, p::Real = 2) =
    p == 2 ? norm2(M) : norm((M[I] for I in CartesianIndices(size(M))), p)
function norm2(M::SymCPD{T,N,K}) where {T,N,K}
    V = reduce(.*, M.U[M.S[i]]'M.U[M.S[i]] for i in 1:ndims(M))
    return sqrt(abs(M.λ' * V * M.λ))
end

"""
    normalizecomps(M::SymCPD, p::Real = 2)

Normalize the components of `M` so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:K` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:K]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:K`,
or a collection of these.

See also: `normalizecomps!`, `norm`.
"""
normalizecomps(M::SymCPD, p::Real = 2; dims = [:λ; 1:length(M.U)], distribute_to = :λ) =
    normalizecomps!(deepcopy(M), p; dims, distribute_to)

"""
    normalizecomps!(M::SymCPD, p::Real = 2)

Normalize the components of `M` in-place so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:K` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:K]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:K`,
or a collection of these.

See also: `normalizecomps`, `norm`.
"""
function normalizecomps!(
    M::SymCPD{T,N,K},
    p::Real = 2;
    dims = [:λ; 1:K],
    distribute_to = :λ,
) where {T,N,K}
    # Check dims and put into standard (mask) form
    dims_iterable = dims isa Symbol ? (dims,) : dims
    all(d -> d === :λ || (d isa Integer && d in 1:ngroups(M)), dims_iterable) || throw(
        ArgumentError(
            "`dims` must be `:λ`, an integer specifying a group, or a collection, got $dims",
        ),
    )
    dims_λ = :λ in dims_iterable
    dims_U = ntuple(in(dims_iterable), ngroups(M))

    # Check distribute_to and put into standard (mask) form
    dist_iterable = distribute_to isa Symbol ? (distribute_to,) : distribute_to
    all(d -> d === :λ || (d isa Integer && d in 1:ngroups(M)), dist_iterable) || throw(
        ArgumentError(
            "`distribute_to` must be `:λ`, an integer specifying a group, or a collection, got $distribute_to",
        ),
    )
    dist_λ = :λ in dist_iterable
    dist_U = ntuple(in(dist_iterable), ngroups(M))

    # Call inner function
    return _normalizecomps!(M, p, dims_λ, dims_U, dist_λ, dist_U)
end

function _normalizecomps!(
    M::SymCPD{T,N,K},
    p::Real,
    dims_λ::Bool,
    dims_U::NTuple{K,Bool},
    dist_λ::Bool,
    dist_U::NTuple{K,Bool},
) where {T,N,K}
    # Utility function to handle zero weights and norms
    zero_to_one(x) = iszero(x) ? oneunit(x) : x

    # Normalize components and collect excess weight
    excess = ones(T, 1, ncomps(M))
    if dims_λ
        norms = map(zero_to_one ∘ abs, M.λ)
        M.λ ./= norms
        excess .*= reshape(norms, 1, ncomps(M))
    end
    for k in Base.OneTo(ngroups(M))
        if dims_U[k]
            norms = mapslices(zero_to_one ∘ Base.Fix2(norm, p), M.U[k]; dims = 1)
            M.U[k] ./= norms
            excess .*= norms
        end
    end

    # Distribute excess weight (uniformly across specified parts)
    excess .= excess .^ (1 / count((dist_λ, dist_U...)))
    if dist_λ
        M.λ .*= dropdims(excess; dims = 1)
    end
    for k in Base.OneTo(ngroups(M))
        if dist_U[k]
            M.U[k] .*= excess
        end
    end

    # Return normalized SymCPD
    return M
end

"""
    permutecomps(M::SymCPD, perm)

Permute the components of `M`.
`perm` is a vector or a tuple of length `ncomps(M)` specifying the permutation.

See also: `permutecomps!`.
"""
permutecomps(M::SymCPD, perm) = permutecomps!(deepcopy(M), perm)

"""
    permutecomps!(M::SymCPD, perm)

Permute the components of `M` in-place.
`perm` is a vector or a tuple of length `ncomps(M)` specifying the permutation.

See also: `permutecomps`.
"""
permutecomps!(M::SymCPD, perm) = permutecomps!(M, collect(perm))
function permutecomps!(M::SymCPD, perm::Vector)
    # Check that perm is a valid permutation
    (length(perm) == ncomps(M) && isperm(perm)) ||
        throw(ArgumentError("`perm` is not a valid permutation of the components"))

    # Permute weights and factor matrices
    M.λ .= M.λ[perm]
    for k in Base.OneTo(ngroups(M))
        M.U[k] .= M.U[k][:, perm]
    end

    # Return CPD with permuted components
    return M
end

"""
    convertCPD(M::SymCPD)

Create a CPD object from a SymCPD

"""
function convertCPD(M::SymCPD)
    # Make a copy of corresponding factor matrix in M for each new factor matrix
    return CPD(deepcopy(M.λ), Tuple([deepcopy(M.U[M.S[dim]]) for dim in M.S]))
end