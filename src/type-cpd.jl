## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `A`.

If `F::CPD` is the decomposition object,
the weights `λ` and factor matrices `U = (U[1],...,U[N])`
can be obtained via `F.λ` and `F.U`,
such that `A = Σ_k λ[k] U[1][:,k] ∘ ⋯ ∘ U[N][:,k]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        require_one_based_indexing(λ, U...)
        for i in Base.OneTo(N)
            size(U[i], 2) == length(λ) || throw(
                DimensionMismatch("U[$i] has dimensions $(size(U[i])) but λ has length $(length(λ))")
            )
        end
        new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

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
    for i in Base.OneTo(N)
        println(io, "\nU[$i] factor matrix:")
        show(io_field, mime, M.U[i])
    end
end

function summary(io::IO, M::CPD)
    dimstring = ndims(M) == 0 ? "0-dimensional" :
                ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    ncomps = ncomponents(M)
    print(io, dimstring, " ", typeof(M),
        " with ", ncomps, ncomps == 1 ? " component" : " components")
end
