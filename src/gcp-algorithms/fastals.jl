## Algorithm: FastALS

"""
    FastALS

**Fast** **A**lternating **L**east **S**quares.

Efficient ALS algorithm proposed in:
> **Fast Alternating LS Algorithms for High Order
>   CANDECOMP/PARAFAC Tensor Factorizations**.
> Anh-Huy Phan, Petr Tichavský, Andrzej Cichocki.
> *IEEE Transactions on Signal Processing*, 2013.
> DOI: 10.1109/TSP.2013.2269903

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

    # Determine order of modes of MTTKRP to compute
    Jns = [prod(size(X)[1:n]) for n in 1:N]
    Kns = [prod(size(X)[n+1:end]) for n in 1:N]
    Kn_minus_ones = [prod(size(X)[n:end]) for n in 1:N]
    n_star = findlast(n -> Jns[n] <= Kn_minus_ones[n], 1:N)
    order = vcat([i for i in n_star:-1:1], [i for i in n_star+1:N])

    buffers = create_FastALS_buffers(M.U, order, Jns, Kns)

    for _ in 1:algorithm.maxiters
        FastALS_iter!(X, M, order, Jns, Kns, buffers)
    end

    return M
end

"""
    FastALS_iter!(X, U, λ) 
    
    Algorithm for computing MTTKRP sequences is from "Fast Alternating LS Algorithms
    for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al., specifically
    section III-C.
"""
function FastALS_iter!(X, M, order, Jns, Kns, buffers)
    N = ndims(X)
    R = size(M.U[1])[2]

    # Compute MTTKRPs recursively
    n_star = order[1]

    for n in order
        if n == n_star
            kr_right = khatrirao!(buffers.kr_buffer_descending, M.U[reverse(n_star+1:N)]...)
            if n_star == 1
                mul!(M.U[n], reshape(X, (Jns[n], Kns[n])), kr_right)
            else
                mul!(buffers.descending_buffers[1], reshape(X, (Jns[n], Kns[n])), kr_right)
                _rl_outer_multiplication!(
                    buffers.descending_buffers[1],
                    M.U,
                    buffers.helper_buffers_descending[n_star-n+1],
                    n,
                )
            end
        elseif n == n_star + 1
            kr_left = khatrirao!(buffers.kr_buffer_ascending, M.U[reverse(1:n-1)]...)
            if n == N
                mul!(M.U[n], reshape(X, (Jns[n-1], Kns[n-1]))', kr_left)
            else
                mul!(
                    buffers.ascending_buffers[1],
                    (reshape(X, (Jns[n-1], Kns[n-1])))',
                    kr_left,
                )
                _lr_outer_multiplication!(
                    buffers.ascending_buffers[1],
                    M.U,
                    buffers.helper_buffers_ascending[n-n_star],
                    n,
                )
            end
        elseif n < n_star
            if n == 1
                for r in 1:R
                    mul!(
                        view(M.U[n], :, r),
                        reshape(
                            view(buffers.descending_buffers[n_star-n], :, r),
                            (Jns[n], size(X)[n+1]),
                        ),
                        view(M.U[n+1], :, r),
                    )
                end
            else
                for r in 1:R
                    mul!(
                        view(buffers.descending_buffers[n_star-n+1], :, r),
                        reshape(
                            view(buffers.descending_buffers[n_star-n], :, r),
                            (Jns[n], size(X)[n+1]),
                        ),
                        view(M.U[n+1], :, r),
                    )
                end
                _rl_outer_multiplication!(
                    buffers.descending_buffers[n_star-n+1],
                    M.U,
                    buffers.helper_buffers_descending[n_star-n+1],
                    n,
                )
            end
        else
            if n == N
                for r in 1:R
                    mul!(
                        view(M.U[n], :, r),
                        reshape(
                            view(buffers.ascending_buffers[N-n_star-1], :, r),
                            (size(X)[n-1], Kns[n-1]),
                        )',
                        view(M.U[n-1], :, r),
                    )
                end
            else
                for r in 1:R
                    mul!(
                        view(buffers.ascending_buffers[n-n_star], :, r),
                        reshape(
                            view(buffers.ascending_buffers[n-n_star-1], :, r),
                            (size(X)[n-1], Kns[n-1]),
                        )',
                        view(M.U[n-1], :, r),
                    )
                end
                _lr_outer_multiplication!(
                    buffers.ascending_buffers[n-n_star],
                    M.U,
                    buffers.helper_buffers_ascending[n-n_star],
                    n,
                )
            end
        end
        # Normalization, update weights
        V = reduce(.*, M.U[i]'M.U[i] for i in setdiff(1:N, n))
        rdiv!(M.U[n], lu!(V))
        M.λ .= norm.(eachcol(M.U[n]))
        M.U[n] ./= permutedims(M.λ)
    end
end

# Helper function for right-to-left outer multiplications
function _rl_outer_multiplication!(Zn, U, kr_buffer, n)
    khatrirao!(kr_buffer, U[reverse(1:n-1)]...)
    for r in 1:size(U[n])[2]
        mul!(
            view(U[n], :, r),
            reshape(view(Zn, :, r), (prod(size(U[i])[1] for i in 1:n-1), size(U[n])[1]))',
            view(kr_buffer, :, r),
        )
    end
end

# Helper function for left-to-right outer multiplications
function _lr_outer_multiplication!(Zn, U, kr_buffer, n)
    khatrirao!(kr_buffer, U[reverse(n+1:length(U))]...)
    for r in 1:size(U[n])[2]
        mul!(
            view(U[n], :, r),
            reshape(
                view(Zn, :, r),
                (size(U[n])[1], prod(size(U[i])[1] for i in n+1:length(U))),
            ),
            view(kr_buffer, :, r),
        )
    end
end

function create_FastALS_buffers(
    U::NTuple{N,TM},
    order,
    Jns,
    Kns,
) where {TM<:AbstractMatrix,N}
    n_star = order[1]
    r = size(U[1])[2]
    dims = [size(U[u])[1] for u in 1:length(U)]

    # Allocate buffers 
    # Buffer for saved products between modes
    descending_buffers =
        n_star < 2 ? nothing : [similar(U[1], (Jns[n], r)) for n in n_star:-1:2]
    ascending_buffers =
        N - n_star - 1 < 1 ? nothing : [similar(U[1], (Kns[n], r)) for n in n_star:N]
    # Buffers for khatri-rao products
    kr_buffer_descending = similar(U[1], (Kns[n_star], r))
    kr_buffer_ascending = similar(U[1], (Jns[n_star], r))
    # Buffers for khatri-rao product in helper function
    helper_buffers_descending =
        n_star < 2 ? nothing : [similar(U[1], (prod(dims[1:n-1]), r)) for n in n_star:-1:2]
    helper_buffers_ascending =
        n_star >= N - 1 ? nothing :
        [similar(U[1], (prod(dims[n+1:N]), r)) for n in n_star+1:N-1]
    return (;
        descending_buffers,
        ascending_buffers,
        kr_buffer_descending,
        kr_buffer_ascending,
        helper_buffers_descending,
        helper_buffers_ascending,
    )
end
