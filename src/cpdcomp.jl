## CPD component type

"""
    CPDComp

Type for a single component of a canonical polyadic decompositions (CPD).

If `M::CPDComp` is the component object,
the scalar weight `λ` and the factor vectors `u = (u[1],...,u[N])`
can be obtained via `M.λ` and `M.u`.
"""
struct CPDComp{T,N,Tu<:AbstractVector{T}}
    λ::T
    u::NTuple{N,Tu}
    function CPDComp{T,N,Tu}(λ, u) where {T,N,Tu<:AbstractVector{T}}
        Base.require_one_based_indexing(u...)
        return new{T,N,Tu}(λ, u)
    end
end
CPDComp(λ::T, u::NTuple{N,Tu}) where {T,N,Tu<:AbstractVector{T}} = CPDComp{T,N,Tu}(λ, u)

ndims(::CPDComp{T,N}) where {T,N} = N
size(M::CPDComp{T,N}, dim::Integer) where {T,N} = dim <= N ? length(M.u[dim]) : 1
size(M::CPDComp{T,N}) where {T,N} = ntuple(d -> size(M, d), N)

function show(io::IO, mime::MIME{Symbol("text/plain")}, M::CPDComp{T,N}) where {T,N}
    # Compute displaysize for showing fields
    LINES, COLUMNS = displaysize(io)
    LINES_FIELD = max(LINES - 2 - N, 0) ÷ (1 + N)
    io_field = IOContext(io, :displaysize => (LINES_FIELD, COLUMNS))

    # Show summary and fields
    summary(io, M)
    println(io)
    println(io, "λ weight:")
    show(io_field, mime, M.λ)
    for k in Base.OneTo(N)
        println(io, "\nu[$k] factor vector:")
        show(io_field, mime, M.u[k])
    end
end

function summary(io::IO, M::CPDComp)
    dimstring =
        ndims(M) == 0 ? "0-dimensional" :
        ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    return print(io, dimstring, " ", typeof(M))
end

function getindex(M::CPDComp{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    return M.λ * prod(M.u[k][I[k]] for k in Base.OneTo(ndims(M)))
end
getindex(M::CPDComp{T,N}, I::CartesianIndex{N}) where {T,N} = getindex(M, Tuple(I)...)

AbstractArray(A::CPDComp) =
    reshape(TensorKernels.khatrirao(reverse(reshape.(A.u, :, 1))...) * A.λ, size(A))
Array(A::CPDComp) = Array(AbstractArray(A))

norm(M::CPDComp, p::Real = 2) =
    p == 2 ? norm2(M) : norm((M[I] for I in CartesianIndices(size(M))), p)
norm2(M::CPDComp{T,N}) where {T,N} = sqrt(abs2(M.λ) * prod(sum(abs2, M.u[i]) for i in 1:N))

"""
    normalizecomps(M::CPDComp, p::Real = 2)

Normalize `M` so that all its factor vectors have `p`-norm equal to unity,
i.e., `norm(M.u[k], p) == 1` for all `k ∈ 1:ndims(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:ndims(M)`,
or a collection of these.

See also: `normalizecomps!`, `norm`.
"""
normalizecomps(M::CPDComp, p::Real = 2; dims = [:λ; 1:ndims(M)], distribute_to = :λ) =
    normalizecomps!(deepcopy(M), p; dims, distribute_to)

"""
    normalizecomps!(M::CPDComp, p::Real = 2)

Normalize `M` in-place so that all its factor vectors have `p`-norm equal to unity,
i.e., `norm(M.u[k], p) == 1` for all `k ∈ 1:ndims(M)`. The excess weight is absorbed into `M.λ`.
Norms equal to zero are ignored (i.e., treated as though they were equal to one).

The following keyword arguments can be used to modify this behavior:
- `dims` specifies what to normalize (default: `[:λ; 1:ndims(M)]`)
- `distribute_to` specifies where to distribute the excess weight (default: `:λ`)
Valid options for these arguments are the symbol `:λ`, an integer in `1:ndims(M)`,
or a collection of these.

See also: `normalizecomps`, `norm`.
"""
function normalizecomps!(
    M::CPDComp{T,N},
    p::Real = 2;
    dims = [:λ; 1:N],
    distribute_to = :λ,
) where {T,N}
    # Check dims and put into standard (mask) form
    dims_iterable = dims isa Symbol ? (dims,) : dims
    all(d -> d === :λ || (d isa Integer && d in 1:N), dims_iterable) || throw(
        ArgumentError(
            "`dims` must be `:λ`, an integer specifying a mode, or a collection, got $dims",
        ),
    )
    dims_λ = :λ in dims_iterable
    dims_u = ntuple(in(dims_iterable), N)

    # Check distribute_to and put into standard (mask) form
    dist_iterable = distribute_to isa Symbol ? (distribute_to,) : distribute_to
    all(d -> d === :λ || (d isa Integer && d in 1:N), dist_iterable) || throw(
        ArgumentError(
            "`distribute_to` must be `:λ`, an integer specifying a mode, or a collection, got $distribute_to",
        ),
    )
    dist_λ = :λ in dist_iterable
    dist_u = ntuple(in(dist_iterable), N)

    # Call inner function
    return _normalizecomps!(M, p, dims_λ, dims_u, dist_λ, dist_u)
end

function _normalizecomps!(
    M::CPDComp{T,N},
    p::Real,
    dims_λ::Bool,
    dims_u::NTuple{N,Bool},
    dist_λ::Bool,
    dist_u::NTuple{N,Bool},
) where {T,N}
    # Utility function to handle zero weights and norms
    zero_to_one(x) = iszero(x) ? oneunit(x) : x

    # Normalize components and collect excess weight
    excess = oneunit(T)
    if dims_λ
        _norm = zero_to_one(abs(M.λ))
        M.λ /= _norm
        excess *= _norm
    end
    for k in Base.OneTo(N)
        if dims_u[k]
            _norm = zero_to_one(norm(M.u[k], p))
            M.u[k] ./= _norm
            excess *= _norm
        end
    end

    # Distribute excess weight (uniformly across specified parts)
    excess = excess^(1 / count((dist_λ, dist_u...)))
    if dist_λ
        M.λ *= excess
    end
    for k in Base.OneTo(N)
        if dist_u[k]
            M.u[k] .*= excess
        end
    end

    # Return normalized CPDComp
    return M
end
