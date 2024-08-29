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
