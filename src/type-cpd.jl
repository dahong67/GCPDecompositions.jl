## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `A`.

If `F::CPD` is the decomposition object,
the weights `λ` and factor matrices `U = (U[1],...,U[N])`
can be obtained via `F.λ` and `F.U`,
such that `A = Σ_j λ[j] U[1][:,j] ∘ ⋯ ∘ U[N][:,j]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        require_one_based_indexing(λ, U...)
        for k in Base.OneTo(N)
            size(U[k], 2) == length(λ) || throw(
                DimensionMismatch("U[$k] has dimensions $(size(U[k])) but λ has length $(length(λ))")
            )
        end
        new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

"""
    ncomponents(M::CPD) -> Integer

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomponents(M::CPD) = length(M.λ)
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
    dimstring = ndims(M) == 0 ? "0-dimensional" :
                ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    ncomps = ncomponents(M)
    print(io, dimstring, " ", typeof(M),
        " with ", ncomps, ncomps == 1 ? " component" : " components")
end

function getindex(M::CPD{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    return sum(M.λ[j] * prod(M.U[k][I[k], j] for k in Base.OneTo(ndims(M)))
               for j in Base.OneTo(ncomponents(M)); init=zero(eltype(T)))
end
getindex(M::CPD{T,N}, I::CartesianIndex{N}) where {T,N} = getindex(M, Tuple(I)...)
