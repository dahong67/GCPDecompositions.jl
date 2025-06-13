## Tensor Kernel: sparse_mttkrp

"""
    sparse mttkrp!(G, X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of a sparse N-way tensor X using the exploded factor matrices U1, U2, ..., UN along mode n, 
using the value in Y_samples which are samples of Y at the indices given 
by sample_idxs, and store the result in G.

Not the most efficient approach for now.

See also: `mttkrp`, `create_mttkrp_buffer`


function sparse_mttkrp!(
    G::TM,
    Y_hat::AbstractSparseMatrix,
    U_exp::NTuple{N,TU},
    n::Integer,
) where {TM<:AbstractMatrix,T,N,TU<:AbstractMatrix}
"""
function sparse_mttkrp!(
    G::TM,
    Y_hat::AbstractSparseMatrix,
    U_exp::NTuple{N,TU},
    n::Integer,
) where {TM<:AbstractMatrix,N,TU<:AbstractMatrix}
    #I, r = _checked_mttkrp_dims(X, U, n)

    # Check output dimensions
    Base.require_one_based_indexing(G)
    size(G)[1] == size(Y_hat)[1] ||
        throw(DimensionMismatch("First dim of output `G` must have the same size as first dim of `Y_hat`"))
    size(G)[2] == size(U_exp[n])[2] ||
        throw(DimensionMismatch("Second dim of output `G` must have the same size as second dim of `U[n]`"))

    # Compute hadamard product of exploded factor matrices
    if n == 1
        Z_exp = deepcopy(U_exp[2])
        for i in 3:N
            Z_exp = Z_exp .* U_exp[i]
        end
    else
        Z_exp = deepcopy(U_exp[1])
        for i in 2:N
            if i == n
                continue
            else
                Z_exp = Z_exp .* U_exp[i]
            end
        end
    end

    mul!(G, Y_hat, Z_exp)

    return G
end

