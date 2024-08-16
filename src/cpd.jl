## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `A`.
This is the return type of `gcp(_)`,
the corresponding tensor decomposition function.

If `M::CPD` is the decomposition object,
the weights `λ` and the factor matrices `U = (U[1],...,U[N])`
can be obtained via `M.λ` and `M.U`,
such that `A = Σ_j λ[j] U[1][:,j] ∘ ⋯ ∘ U[N][:,j]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        Base.require_one_based_indexing(λ, U...)
        for k in Base.OneTo(N)
            size(U[k], 2) == length(λ) || throw(
                DimensionMismatch(
                    "U[$k] has dimensions $(size(U[k])) but λ has length $(length(λ))",
                ),
            )
        end
        return new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

"""
    ncomps(M::CPD)

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomps(M::CPD) = length(M.λ)
ndims(::CPD{T,N}) where {T,N} = N

size(M::CPD{T,N}, dim::Integer) where {T,N} = dim <= N ? size(M.U[dim], 1) : 1
size(M::CPD{T,N}) where {T,N} = ntuple(d -> size(M, d), N)

function show(io::IO, mime::MIME{Symbol("text/plain")}, M::CPD{T,N}) where {T,N}
    # Compute displaysize for showing fields
    LINES, COLUMNS = displaysize(io)
    LINES_FIELD = max(LINES - 2 - N, 0) ÷ (1 + N)
    io_field = IOContext(io, :displaysize => (LINES_FIELD, COLUMNS))

    # Show summary and fields
    summary(io, M)
    println(io)
    println(io, "λ weights:")
    show(io_field, mime, M.λ)
    for k in Base.OneTo(N)
        println(io, "\nU[$k] factor matrix:")
        show(io_field, mime, M.U[k])
    end
end

function summary(io::IO, M::CPD)
    dimstring =
        ndims(M) == 0 ? "0-dimensional" :
        ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    _ncomps = ncomps(M)
    return print(
        io,
        dimstring,
        " ",
        typeof(M),
        " with ",
        _ncomps,
        _ncomps == 1 ? " component" : " components",
    )
end

function getindex(M::CPD{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    val = zero(eltype(T))
    for j in Base.OneTo(ncomps(M))
        val += M.λ[j] * prod(M.U[k][I[k], j] for k in Base.OneTo(ndims(M)))
    end
    return val
end
getindex(M::CPD{T,N}, I::CartesianIndex{N}) where {T,N} = getindex(M, Tuple(I)...)

norm(M::CPD, p::Real = 2) =
    p == 2 ? norm2(M) : norm((M[I] for I in CartesianIndices(size(M))), p)
function norm2(M::CPD{T,N}) where {T,N}
    V = reduce(.*, M.U[i]'M.U[i] for i in 1:N)
    return sqrt(abs(M.λ' * V * M.λ))
end

"""
    normalizecomps(M::CPD, p::Real = 2)

Normalize the components of `M` so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:ndims(M)` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
"""
normalizecomps(M::CPD, p::Real = 2) = normalizecomps!(deepcopy(M), p)

"""
    normalizecomps!(M::CPD, p::Real = 2)

Normalize the components of `M` in-place so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:ndims(M)` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.
"""
function normalizecomps!(M::CPD{T,N}, p::Real = 2) where {T,N}
    for k in 1:N
        norms = mapslices(Base.Fix2(norm, p), M.U[k]; dims = 1)
        M.U[k] ./= norms
        M.λ .*= dropdims(norms; dims = 1)
    end
    return M
end
