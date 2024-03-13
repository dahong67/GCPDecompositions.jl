## Algorithm: FastALS

"""
    FastALS

Fast Alternating Least Squares.
Faster and more memory-efficient implementation of ALS from "Fast Alternating LS Algorithms
for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al.

Algorithm parameters:

- `maxiters::Int` : max number of iterations (default: `200`)

"""
Base.@kwdef struct FastALS <: AbstractAlgorithm
    maxiters::Int = 200
end

function _gcp(
    X::Array{TX,N},
    r,
    loss::GCPLosses.LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.FastALS,
    init,
) where {TX<:Real,N}

    # Initialization
    M = deepcopy(init)

    for _ in 1:algorithm.maxiters
        FastALS_iter!(X, M)
    end

    return M
end

"""
    FastALS_iter!(X, U, λ) 
    
    Algorithm for computing MTTKRP sequences is from "Fast Alternating LS Algorithms
    for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al., specifically
    section III-C.
"""
function FastALS_iter!(X, M)

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
                mul!(M.U[n], reshape(X, (Jns[n], Kns[n])), khatrirao(M.U[reverse(n+1:N)]...))  
            else
                saved = reshape(X, (Jns[n], Kns[n])) * khatrirao(M.U[reverse(n+1:N)]...)
                FastALS_mttkrps_helper!(saved, M.U, n, "right", N, Jns, Kns)
            end
        elseif n == n_star + 1
            if n == N
                mul!(M.U[n], reshape(X, (Jns[n-1], Kns[n-1]))', khatrirao(M.U[reverse(1:n-1)]...))
            else
                saved = (khatrirao(M.U[reverse(1:n-1)]...)' * reshape(X, (Jns[n-1], Kns[n-1])))'
                FastALS_mttkrps_helper!(saved, M.U, n, "left", N, Jns, Kns)
            end  
        elseif n < n_star
            if n == 1
                for r in 1:R
                    mul!(view(M.U[n], :, r), reshape(view(saved, :, r), (Jns[n], size(X)[n+1])), view(M.U[n+1], :, r))
                end
            else
                saved = stack(reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(M.U[n+1], :, r) for r in 1:R)
                FastALS_mttkrps_helper!(saved, M.U, n, "right", N, Jns, Kns)
            end
        else
            if n == N
                for r in 1:R
                    mul!(view(M.U[n], :, r), reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))', view(M.U[n-1], :, r))
                end
            else
                saved = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(M.U[n-1], :, r) for r in 1:R)
                FastALS_mttkrps_helper!(saved. M.U, n, "left", N, Jns, Kns)
            end
        end
        # Normalization, update weights
        V = reduce(.*, M.U[i]'M.U[i] for i in setdiff(1:N, n))
        #M.U[n] = M.U[n] / V
        rdiv!(M.U[n], lu!(V))
        M.λ .= norm.(eachcol(M.U[n]))
        M.U[n] ./= permutedims(M.λ)
        #M.λ .= norm.(eachcol(M.U[n]))
        #M.U[n] = M.U[n] ./ permutedims(M.λ)
    end
end 


function FastALS_mttkrps_helper!(Zn, U, n, side, N, Jns, Kns)
    if side == "right"
        kr = khatrirao(U[reverse(1:n-1)]...)
        for r in 1:size(U[n])[2]
            U[n][:, r] = reshape(view(Zn, :, r), (Jns[n-1], size(U[n])[1]))' * kr[:, r]
        end
    elseif side == "left"
        kr = khatrirao(U[reverse(n+1:N)]...)
        for r in 1:size(U[n])[2]
            U[n][:, r] = reshape(view(Zn, :, r), (size(U[n])[1], Kns[n])) * kr[:, r]
        end
    end
end
