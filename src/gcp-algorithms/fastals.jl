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
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.FastALS,
) where {TX<:Real,N}
    T = promote_type(TX, Float64)

    # Random initialization
    M0 = CPD(ones(T, r), rand.(T, size(X), r))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    λ, U = M0.λ, collect(M0.U)

    for _ in 1:algorithm.maxiters
        FastALS_iter!(X, U, λ)
    end

    return CPD(λ, Tuple(U))
end

"""
    FastALS_iter!(X, U, λ) 
    
    Algorithm for computing MTTKRP sequences is from "Fast Alternating LS Algorithms
    for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al., specifically
    section III-C.
"""
function FastALS_iter!(X, U, λ)

    N = ndims(X)
    R = size(U[1])[2]

    # Determine order of modes of MTTKRP to compute
    Jns = [prod(size(X)[1:n]) for n in 1:N]
    Kns = [prod(size(X)[n+1:end]) for n in 1:N]
    Kn_minus_ones = [prod(size(X)[n:end]) for n in 1:N]
    comp = Jns .<= Kn_minus_ones
    n_star = maximum(map(x -> comp[x] ? x : 0, 1:N))
    order = vcat([i for i in n_star:-1:1], [i for i in n_star+1:N])

    # Compute MTTKRPs recursively
    saved = similar(U[1], Jns[n_star], R)
    for n in order
        if n == n_star
            saved = reshape(X, (Jns[n], Kns[n])) * khatrirao(U[reverse(n+1:N)]...)
            mttkrps_helper!(saved, U, n, "right", N, Jns, Kns)
        elseif n == n_star + 1
            saved = (khatrirao(U[reverse(1:n-1)]...)' * reshape(X, (Jns[n-1], Kns[n-1])))'
            if n == N
                U[n] = saved
            else
                mttkrps_helper!(saved, U, n, "left", N, Jns, Kns)
            end  
        elseif n < n_star
            # Try stack
            saved = stack(reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(U[n+1], :, r) for r in 1:R)
            if n == 1
                U[n] = saved
            else
                mttkrps_helper!(saved, U, n, "right", N, Jns, Kns)
            end
        else
            saved = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(U[n-1], :, r) for r in 1:R)
            if n == N
                U[n] = saved
            else
                mttkrps_helper!(saved, U, n, "left", N, Jns, Kns)
            end
        end
        # Normalization, update weights
        V = reduce(.*, U[i]'U[i] for i in setdiff(1:N, n))
        U[n] = U[n] / V
        λ .= norm.(eachcol(U[n]))
        U[n] = U[n] ./ permutedims(λ)
    end
end 


function mttkrps_helper!(Zn, U, n, side, N, Jns, Kns)
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
