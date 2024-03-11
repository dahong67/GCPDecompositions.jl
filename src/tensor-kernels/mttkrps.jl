## Tensor Kernel: mttkrps

"""
    faster_mttkrps!(GU, M, X) 
    
    Algorithm for computing MTTKRP sequences is from "Fast Alternating LS Algorithms
    for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al., specifically
    section III-C.
"""
function faster_mttkrps!(GU, M, X)

    N = ndims(X)
    R = size(M.U[1])[2]

    # Determine order of modes of MTTKRP to compute
    Jns = [prod(size(X)[1:n]) for n in 1:N]
    Kns = [prod(size(X)[n+1:end]) for n in 1:N]
    Kn_minus_ones = [prod(size(X)[n:end]) for n in 1:N]
    comp = Jns .<= Kn_minus_ones
    n_star = maximum(map(x -> comp[x] ? x : 0, 1:N))
    order = vcat([i for i in n_star:-1:1], [i for i in n_star+1:N])

    # Compute MTTKRPs recursively
    saved = similar(M.U[1], Jns[n_star], R)
    for n in order
        if n == n_star
            if n == 1
                mul!(GU[n], reshape(X, (Jns[n], Kns[n])), khatrirao(M.U[reverse(n+1:N)]...))
            else
                saved = reshape(X, (Jns[n], Kns[n])) * khatrirao(M.U[reverse(n+1:N)]...)
                mttkrps_helper!(GU, saved, M, n, "right", N, Jns, Kns)
            end
        elseif n == n_star + 1
            if n == N
                mul!(GU[n], reshape(X, (Jns[n-1], Kns[n-1]))', khatrirao(M.U[reverse(1:n-1)]...))
            else
                saved = (khatrirao(M.U[reverse(1:n-1)]...)' * reshape(X, (Jns[n-1], Kns[n-1])))'
                mttkrps_helper!(GU, saved, M, n, "left", N, Jns, Kns)
            end  
        elseif n < n_star
            if n == 1
                for r in 1:R
                    mul!(view(GU[n], :, r), reshape(view(saved, :, r), (Jns[n], size(X)[n+1])), view(M.U[n+1], :, r))
                    #GU[n][:, r] = reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(M.U[n+1], :, r)
                end
            else
                saved = stack(reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(M.U[n+1], :, r) for r in 1:R)
                mttkrps_helper!(GU, saved, M, n, "right", N, Jns, Kns)
            end
        else
            if n == N
                for r in 1:R
                    mul!(view(GU[n], :, r), reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))', view(M.U[n-1], :, r))
                end
                #GU[n] = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(M.U[n-1], :, r) for r in 1:R)
            else
                saved = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(M.U[n-1], :, r) for r in 1:R)
                mttkrps_helper!(GU, saved, M, n, "left", N, Jns, Kns)
            end
        end
    end
end 


function mttkrps_helper!(GU, Zn, M, n, side, N, Jns, Kns)
    if side == "right"
        kr = khatrirao(M.U[reverse(1:n-1)]...)
        for r in 1:size(M.U[n])[2]
            mul!(view(GU[n], :, r), reshape(view(Zn, :, r), (Jns[n-1], size(M.U[n])[1]))', kr[:, r])
            #GU[n][:,r] = reshape(view(Zn, :, r), (Jns[n-1], size(M.U[n])[1]))' * kr[:, r]
        end
    elseif side == "left"
        kr = khatrirao(M.U[reverse(n+1:N)]...)
        for r in 1:size(M.U[n])[2]
            mul!(view(GU[n], :, r), reshape(view(Zn, :, r), (size(M.U[n])[1], Kns[n])), kr[:, r])
            #GU[n][:,r] = reshape(view(Zn, :, r), (size(M.U[n])[1], Kns[n])) * kr[:, r]
        end
    end
end

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
