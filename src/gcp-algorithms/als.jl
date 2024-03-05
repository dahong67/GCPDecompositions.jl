## Algorithm: LBFGSB

"""
    ALS

**A**lternating **L**east **S**quares.
Workhorse algorithm for `LeastSquaresLoss` with no constraints.

Algorithm parameters:

- `maxiters::Int` : max number of iterations (default: `200`)
"""
Base.@kwdef struct ALS <: AbstractAlgorithm
    maxiters::Int = 200
end

function _gcp(
    X::Array{TX,N},
    r,
    loss::GCPLosses.LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.ALS,
) where {TX<:Real,N}
    T = promote_type(TX, Float64)

    # Random initialization
    M0 = CPD(ones(T, r), rand.(T, size(X), r))
    M0norm = norm(M0)
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    M = deepcopy(M0)

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
