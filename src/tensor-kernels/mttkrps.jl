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
    _checked_mttkrps_dims(X, U)

    # Check output dimensions
    Base.require_one_based_indexing(G...)
    size.(G) == size.(U) ||
        throw(DimensionMismatch("Output `G` must have the same size as `U`"))

    # Compute individual MTTKRP's
    for n in 1:N
        mttkrp!(G[n], X, U, n)
    end
    return G
end

"""
    sparse_mttkrps!(G, X, (U1, U2, ..., UN))

Compute the sparse Matricized Tensor Times Khatri-Rao Product Sequence (MTTKRPS)
of a sparse N-way tensor X with the matrices U1, U2, ..., UN
and store the result in G.

"""
function sparse_mttkrps!(
    G::NTuple{N,TM},
    X::SparseArray{TX,N},
    U::NTuple{N,TU},
) where {TM<:AbstractMatrix,TX,N,TU<:AbstractMatrix}
    _checked_mttkrps_dims(X, U)

    # Check output dimensions
    Base.require_one_based_indexing(G)
    for n in 1:N
        size(G[n]) == size(U[n]) ||
            throw(DimensionMismatch("Output `G[n]` must have the same size as `U[n]`"))
    end

    #inds, vals, s = storedindices(X), storedvalues(X), numstored(X)
    inds, vals, s = nonzero_keys(X), nonzero_values(X), nonzero_length(X)
    r = size(U[1])[2]
    mode_inds = ntuple(k -> getindex.(inds, k), N)
    
    Zh = ones.(eltype(G[1]), ntuple(_ -> (s, r), N))
    Zh[2] .= U[1][mode_inds[1], :]
    for k in 3:N
        Zh[k] .= Zh[k-1] .* U[k-1][mode_inds[k-1], :]
    end
    Zh[1] .= U[N][mode_inds[N], :]
    for k in N-1:-1:2
        Zh[k] .*= Zh[1]
        Zh[1] .*= U[k][mode_inds[k], :]
    end

    for n in 1:N
        In = size(X, n)
        Yh = sparse(mode_inds[n], 1:s, collect(vals), In, s)
        G[n] .= Yh*Zh[n]
    end
    return G

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
