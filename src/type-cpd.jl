## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `T`.

If `F::CPD` is the decomposition object,
the weights `λ` and factor matrices `U = (U[1],...,U[N])`
can be obtained via `F.λ` and `F.U`,
such that `T = Σ_k λ[k] U[1][:,k] ∘ ⋯ ∘ U[N][:,k]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        require_one_based_indexing(λ, U...)
        for i in 1:N
            size(U[i], 2) == length(λ) || throw(
                DimensionMismatch("U[$i] has dimensions $(size(U[i])) but λ has length $(length(λ))")
            )
        end
        new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

ncomponents(X::CPD) = length(X.λ)
ndims(::CPD{T,N}) where {T,N} = N

size(X::CPD{T,N}, dim::Integer) where {T,N} = dim <= N ? size(X.U[dim], 1) : 1
size(X::CPD{T,N}) where {T,N} = ntuple(d -> size(X, d), N)

function show(io::IO, mime::MIME{Symbol("text/plain")}, X::CPD{T,N}) where {T,N}
    # Compute displaysize for showing fields
    LINES, COLUMNS = displaysize(io)
    LINES_FIELD = max(LINES - 2 - N, 0) ÷ (1 + N)
    io_field = IOContext(io, :displaysize => (LINES_FIELD, COLUMNS))

    # Show summary and fields
    summary(io, X)
    println(io)
    println(io, "λ weights:")
    show(io_field, mime, X.λ)
    for i in 1:N
        println(io, "\nU[$i] factor matrix:")
        show(io_field, mime, X.U[i])
    end
end

function summary(io::IO, X::CPD)
    dimstring = ndims(X) == 0 ? "0-dimensional" :
                ndims(X) == 1 ? "$(size(X,1))-element" : join(map(string, size(X)), '×')
    ncomps = ncomponents(X)
    print(io, dimstring, " ", typeof(X),
        " with ", ncomps, ncomps == 1 ? " component" : " components")
end
