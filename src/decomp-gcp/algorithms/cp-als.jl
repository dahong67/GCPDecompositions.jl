## Algorithm: CP_ALS

"""
    CP_ALS

**A**lternating **L**east **S**quares.
Workhorse algorithm for `LeastSquares` loss with no constraints.

Algorithm parameters:

+ `maxiters::Int` : max number of iterations (default: `200`)
"""
Base.@kwdef struct CP_ALS <: AbstractGCPAlgorithm
    maxiters::Int = 200
end

function _gcp!(
    rng::AbstractRNG,
    M::CPD{Float64,N},
    X::Array{<:Real,N},
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::CP_ALS,
) where {N}
    # Pre-allocate MTTKRP buffers
    mttkrp_buffers = ntuple(n -> create_mttkrp_buffer(X, M.U, n), N)

    # Alternating Least Squares (ALS) iterations
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, M.U[i]'M.U[i] for i in setdiff(1:N, n))
            mttkrp!(M.U[n], X, M.U, n, mttkrp_buffers[n])
            rdiv!(M.U[n], lu!(V))
            M.λ .= norm.(eachcol(M.U[n]))
            M.U[n] ./= permutedims(M.λ)
        end
    end

    return M
end
