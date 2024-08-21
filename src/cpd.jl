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

AbstractArray(A::CPD) = reshape(TensorKernels.khatrirao(reverse(A.U)...) * A.λ, size(A))
Array(A::CPD) = Array(AbstractArray(A))

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

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)

See also: `normalizecomps!`.
"""
normalizecomps(M::CPD, p::Real = 2; dims = [:λ; 1:ndims(M)], distribute_to = :λ) =
    normalizecomps!(deepcopy(M), p; dims, distribute_to)

"""
    normalizecomps!(M::CPD, p::Real = 2)

Normalize the components of `M` in-place so that the columns of all its factor matrices
all have `p`-norm equal to unity, i.e., `norm(M.U[k][:, j], p) == 1` for all
`k ∈ 1:ndims(M)` and `j ∈ 1:ncomps(M)`. The excess weight is absorbed into `M.λ`.

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)

See also: `normalizecomps`.
"""
function normalizecomps!(
    M::CPD{T,N},
    p::Real = 2;
    dims = [:λ; 1:N],
    distribute_to = :λ,
) where {T,N}
    # Put keyword arguments into standard form
    dims_λ, dims_U = _dims_list(dims, N)
    dist_λ, dist_U = _dims_list(distribute_to, N)

    # Normalize components and collect excess weight
    excess = ones(T, 1, ncomps(M))
    if dims_λ
        norms = map(abs, M.λ)
        M.λ ./= norms
        excess .*= reshape(norms, 1, ncomps(M))
    end
    for k in dims_U
        norms = mapslices(Base.Fix2(norm, p), M.U[k]; dims = 1)
        M.U[k] ./= norms
        excess .*= norms
    end

    # Distribute excess weight (uniformly across specified parts)
    excess .= excess .^ (1 / (length(dist_U) + (dist_λ ? 1 : 0)))
    if dist_λ
        M.λ .*= dropdims(excess; dims = 1)
    end
    for k in dist_U
        M.U[k] .*= excess
    end

    # Return normalized CPD
    return M
end

"""
    _dims_list(dims, N)

Make sure `dims` specifies the weights `:λ` or one of the modes (or a list of them)
and return them in a standardized form.
"""
_dims_list(dims::Symbol, N) = _dims_list([dims], N)
_dims_list(dims::Integer, N) = _dims_list([dims], N)
function _dims_list(dims, N)
    # Check dims
    for d in dims
        (d === :λ || (d isa Integer && d in 1:N)) || throw(
            ArgumentError(
                "dimension must be either :λ or an integer specifying a mode, got $d",
            ),
        )
    end

    # Return standardized forms
    return (:λ in dims, filter(in(dims), 1:N))
end
