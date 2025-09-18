## Sparse array type

module SparseArrayCOOs

# Imports: AbstractArray interface
import Base: size, getindex, setindex!
import Base: IndexStyle

# Imports: Overloads for specializing outputs
import Base: similar, show, summary

# Exports
export SparseArrayCOO

"""
    SparseArrayCOO{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N}

`N`-dimensional sparse array stored in the **COO**rdinate format.
Elements are stored as a vector of indices (using type `Ti`)
and a vector of values (of type `Tv`).
Values for duplicate indices are summed.

Fields:
+ `dims::Dims{N}`              : tuple of dimensions
+ `inds::Vector{NTuple{N,Ti}}` : vector of indices
+ `vals::Vector{Tv}`           : vector of values
"""
struct SparseArrayCOO{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N}
    dims::Dims{N}                   # Dimensions
    inds::Vector{NTuple{N,Ti}}      # Stored indices
    vals::Vector{Tv}                # Stored values

    function SparseArrayCOO{Tv,Ti,N}(
        dims::Dims{N},
        inds::Vector{NTuple{N,Ti}},
        vals::Vector{Tv},
    ) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        check_coo_buffers(inds, vals)
        check_coo_inds(dims, inds)
        return new(dims, inds, vals)
    end
end
SparseArrayCOO(
    dims::Dims{N},
    inds::Vector{NTuple{N,Ti}},
    vals::Vector{Tv},
) where {Tv,Ti<:Integer,N} = SparseArrayCOO{Tv,Ti,N}(dims, inds, vals)

"""
    SparseArrayCOO{Tv,Ti<:Integer}(undef, dims)
    SparseArrayCOO{Tv,Ti<:Integer,N}(undef, dims)

Construct an uninitialized `N`-dimensional `SparseArrayCOO`
with indices using type `Ti` and elements of type `Tv`.
Here uninitialized means it has no stored entries.

Here `undef` is the `UndefInitializer`. If `N` is supplied,
then it must match the length of `dims`.

# Examples
```julia-repl
julia> A = SparseArrayCOO{Float64, Int8, 3}(undef, (2, 3, 4)) # N given explicitly
2×3×4 SparseArrayCOO{Float64, Int8, 3} with 0 stored entries

julia> B = SparseArrayCOO{Float64, Int8}(undef, (4,)) # N determined by the input
4-element SparseArrayCOO{Float64, Int8, 1} with 0 stored entries

julia> similar(B, 2, 4, 1) # use typeof(B), and the given size
2×4×1 SparseArrayCOO{Float64, Int8, 3} with 0 stored entries
```
"""
SparseArrayCOO{Tv,Ti,N}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO(dims, Vector{NTuple{N,Ti}}(), Vector{Tv}())
SparseArrayCOO{Tv,Ti}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO{Tv,Ti,N}(undef, dims)

"""
    SparseArrayCOO([Ti=Int], A::AbstractArray)

Convert an AbstractArray `A` into a `SparseArrayCOO`
with indices using type `Ti`.

# Examples
```julia-repl
julia> A = SparseArrayCOO(Int8, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseArrayCOO{Float16, Int8, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3

julia> B = SparseArrayCOO(Int16, Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseArrayCOO{Float16, Int16, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3

julia> C = SparseArrayCOO(Float16[1.1 0.0 0.0; 2.1 0.0 2.3])
2×3 SparseArrayCOO{Float16, Int64, 2} with 3 stored entries:
  [1, 1]  =  1.1
  [2, 1]  =  2.1
  [2, 3]  =  2.3
```
"""
function SparseArrayCOO(Ti::Type{<:Integer}, A::AbstractArray)
    Tv, N = eltype(A), ndims(A)
    dims = size(A)
    nzidx = findall(!iszero, A)
    inds = convert(Vector{NTuple{N,Ti}}, CartesianIndices(A)[nzidx])
    vals = convert(Vector{Tv}, A[nzidx])
    return SparseArrayCOO{Tv,Ti,N}(dims, inds, vals)
end
SparseArrayCOO(A::AbstractArray) = SparseArrayCOO(Int, A)

## Minimal AbstractArray interface

size(A::SparseArrayCOO) = A.dims

function getindex(A::SparseArrayCOO{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    out = zero(Tv)
    for (ind, val) in zip(A.inds, A.vals)
        if ind == I
            out += val
        end
    end
    return out
end

function setindex!(A::SparseArrayCOO{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    ind, val = convert(NTuple{N,Ti}, I), convert(Tv, v)
    done = false
    for i in eachindex(A.inds, A.vals)
        if A.inds[i] == I
            if !done
                A.vals[i] = val
                done = true
            else
                A.vals[i] = zero(Tv)
            end
        end
    end
    if !done && !iszero(val)
        push!(A.inds, ind)
        push!(A.vals, val)
    end
    return A
end

IndexStyle(::Type{<:SparseArrayCOO}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseArrayCOO{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseArrayCOO{Tv,Ti,N}(undef, dims)

show(io::IO, A::SparseArrayCOO) = invoke(show, Tuple{IO,Any}, io, A)
function show(io::IO, ::MIME"text/plain", A::SparseArrayCOO)
    nstored, N = numstored(A), ndims(A)
    inds, vals = A.inds, A.vals

    # Print summary
    summary(io, A)
    iszero(nstored) && return
    print(io, ":")

    # Print stored entries
    entrylines = get(io, :limit, false) ? displaysize(io)[1] - 4 : typemax(Int)
    pad = map(ndigits, size(A))
    if entrylines >= nstored                  # Enough space to print all the stored entries
        perm = issorted(inds; by = reverse) ? (1:nstored) : sortperm(inds; by = reverse)
        sinds, svals = view(inds, perm), view(vals, perm)
        for ptr in 1:nstored
            if ptr == 1 || sinds[ptr] != sinds[ptr-1]
                _print_ln_entry(io, pad, sinds[ptr], svals[ptr])
            else
                _print_ln_dup_entry(io, pad, svals[ptr])
            end
        end
    elseif entrylines <= 0                    # No space to print any of the stored entries
        print(io, " \u2026")
    elseif entrylines == 1                    # Only space to print vertical dots
        print(io, '\n', " \u22ee")
    elseif entrylines == 2                    # Only space to print first stored entry
        ptr = findmin(reverse, inds)[2]
        ind, val = inds[ptr], vals[ptr]
        _print_ln_entry(io, pad, ind, val)
        !isnothing(findnext(==(ind), inds, ptr + 1)) && print(io, "  +  \u2026")
        print(io, '\n', ' '^(3 + sum(pad) + 2 * (N - 1) + 3), '\u22ee')
    else                                      # Print the stored entries in two chunks
        # Check if indices are already sorted
        indsorted = issorted(inds; by = reverse)

        # First chunk
        prechunk = div(entrylines - 1, 2, RoundUp)
        perm =
            indsorted ? (1:(prechunk+1)) :
            partialsortperm(inds, 1:(prechunk+1); by = reverse)
        sinds, svals = view(inds, perm), view(vals, perm)
        for ptr in 1:prechunk
            if ptr == 1 || sinds[ptr] != sinds[ptr-1]
                _print_ln_entry(io, pad, sinds[ptr], svals[ptr])
            else
                _print_ln_dup_entry(io, pad, svals[ptr])
            end
        end
        sinds[prechunk+1] == sinds[prechunk] && print(io, "  +  \u2026")

        # Dots
        print(io, '\n', ' '^(3 + sum(pad) + 2 * (N - 1) + 3), '\u22ee')

        # Second chunk
        postchunk = div(entrylines - 1, 2, RoundDown)
        perm =
            indsorted ? ((nstored-postchunk):nstored) :
            partialsortperm(inds, (nstored-postchunk):nstored; by = reverse)
        sinds, svals = view(inds, perm), view(vals, perm)
        for ptr in 2:(postchunk+1)
            if sinds[ptr] != sinds[ptr-1]
                _print_ln_entry(io, pad, sinds[ptr], svals[ptr])
            elseif ptr == 2
                _print_ln_entry(io, pad, sinds[ptr], svals[ptr], true)
            else
                _print_ln_dup_entry(io, pad, svals[ptr])
            end
        end
    end
end
function _print_ln_entry(
    io::IO,
    pad::NTuple{N,Int},
    ind::NTuple{N,<:Integer},
    val,
    predots = false,
) where {N}
    print(io, '\n', "  [")
    for k in 1:N
        print(io, lpad(Int(ind[k]), pad[k]))
        k == N || print(io, ", ")
    end
    return print(io, "]  =  ", predots ? "\u2026  +  " : "", val)
end
function _print_ln_dup_entry(io::IO, pad::NTuple{N,Int}, val) where {N}
    print(io, '\n', "   ")
    for k in 1:N
        print(io, ' '^(pad[k]))
        k == N || print(io, "  ")
    end
    return print(io, "   +  ", val)
end

function summary(io::IO, A::SparseArrayCOO)
    invoke(summary, Tuple{IO,AbstractArray}, io, A)
    nstored = numstored(A)
    return print(io, " with ", nstored, " stored ", nstored == 1 ? "entry" : "entries")
end

## Utilities

"""
    numstored(A::SparseArrayCOO{Tv,Ti,N})

Return the number of stored entries.
Includes any stored numerical zeros and duplicates;
use `count(!iszero,A)` to count the number of nonzeros.
"""
numstored(A::SparseArrayCOO) = length(A.vals)

"""
    check_coo_buffers(inds, vals)

Check that the `inds` and `vals` buffers are valid:
+ their lengths match (`length(inds) == length(vals)`)
If not, throw an `ArgumentError`.
"""
function check_coo_buffers(
    inds::Vector{NTuple{N,Ti}},
    vals::Vector{Tv},
) where {Tv,Ti<:Integer,N}
    length(inds) == length(vals) ||
        throw(ArgumentError("the buffer lengths (length(inds) = $(length(inds)), \
                             length(vals) = $(length(vals))) do not match"))
    return nothing
end

"""
    check_coo_inds(dims, inds)

Check that the indices in `inds` are valid:
+ each index is in bounds (`1 ≤ inds[ptr][k] ≤ dims[k]`)
If not, throw an `ArgumentError`.
"""
function check_coo_inds(dims::Dims{N}, inds::Vector{NTuple{N,Ti}}) where {Ti<:Integer,N}
    for ind in inds
        checkbounds_dims(dims, ind...)
    end
    return nothing
end

"""
    check_Ti(dims, Ti)

Check that the `dims` tuple and `Ti` index type are valid:
+ `dims` are nonnegative and fit in `Ti` (`0 ≤ dims[k] ≤ typemax(Ti)`)
+ corresponding length fits in `Int` (`prod(dims) ≤ typemax(Int)`)
If not, throw an `ArgumentError`.
"""
function check_Ti(dims::Dims{N}, Ti::Type) where {N}
    # Check that dims are nonnegative and fit in Ti
    maxTi = typemax(Ti)
    for k in 1:N
        dim = dims[k]
        dim >= 0 || throw(
            ArgumentError("the size along dimension $k (dims[$k] = $dim) is negative"),
        )
        dim <= maxTi ||
            throw(ArgumentError("the size along dimension $k (dims[$k] = $dim) does not \
                                 fit in Ti = $(Ti) (typemax($Ti) = $(typemax(Ti)))"))
    end

    # Check that corresponding length fits in Int
    len = reduce(widemul, dims)
    len <= typemax(Int) || throw(ArgumentError("number of elements (length = $len) does \
                                                not fit in Int (prevents linear indexing)"))
    # do not need to check that dims[k] <= typemax(Int)
    # for CartesianIndex since eltype(dims) == Int

    return nothing
end

"""
    checkbounds_dims(Bool, dims, I...)

Return `true` if the specified indices `I` are in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(::Type{Bool}, dims::Dims{N}, I::Vararg{Integer,N}) where {N}
    for k in 1:N
        (1 <= I[k] <= dims[k]) || return false
    end
    return true
end

"""
    checkbounds_dims(dims, I...)

Throw an error if the specified indices `I` are not in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(dims::Dims{N}, I::Vararg{Integer,N}) where {N}
    checkbounds_dims(Bool, dims, I...) ||
        throw(ArgumentError("index (= $I) out of bounds (dims = $dims)"))
    return nothing
end

end
