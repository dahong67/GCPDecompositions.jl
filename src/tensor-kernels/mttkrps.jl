## Tensor Kernel: mttkrps

"""
    mttkrps(X, (U1, U2, ..., UN))

Compute the Matricized Tensor Times Khatri-Rao Product Sequence (MTTKRPS)
of an N-way tensor X with the matrices U1, U2, ..., UN.

See also: `mttkrps!`
"""
function mttkrps(X::AbstractArray{T,N}, U::NTuple{N,TM}) where {TM<:AbstractMatrix,T,N}
    _checked_mttkrps_dims(X, U)
    return mttkrps!(similar.(U), X, U)
end

"""
    mttkrps!(G, X, (U1, U2, ..., UN))

Compute the Matricized Tensor Times Khatri-Rao Product Sequence (MTTKRPS)
of an N-way tensor X with the matrices U1, U2, ..., UN and store the result in G.

See also: `mttkrps`
"""
function mttkrps!(
    G::NTuple{N,TM},
    X::AbstractArray{T,N},
    U::NTuple{N,TM},
) where {TM<:AbstractMatrix,T,N}
    I, r = _checked_mttkrps_dims(X, U)

    # Check output dimensions
    Base.require_one_based_indexing(G...)
    size.(G) == size.(U) ||
        throw(DimensionMismatch("Output `G` must have the same size as `U`"))

    # Inefficient but simple algorithm
    return ntuple(Val(N)) do n
        Xn = reshape(PermutedDimsArray(X, [n; setdiff(1:N, n)]), I[n], :)
        Zn = similar(Xn, prod(I[setdiff(1:N, n)]), r)
        for j in Base.OneTo(r)
            Zn[:, j] = reduce(kron, [view(U[i], :, j) for i in reverse(setdiff(1:N, n))])
        end
        return mul!(G[n], Xn, Zn)
    end
end

"""
    _checked_mttkrps_dims(X, (U1, U2, ..., UN))

Check that `X` and `U` have compatible dimensions for the mode-`n` MTTKRP.
If so, return a tuple of the number of rows and the shared number of columns
for the Khatri-Rao product. If not, throw an error.
"""
function _checked_mttkrps_dims(
    X::AbstractArray{T,N},
    U::NTuple{N,TM},
) where {TM<:AbstractMatrix,T,N}
    # Check Khatri-Rao product
    I, r = _checked_khatrirao_dims(U...)

    # Check tensor
    Base.require_one_based_indexing(X)
    (I == size(X)) ||
        throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))

    return I, r
end
