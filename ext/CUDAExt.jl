module CUDAExt

using GCPDecompositions, CUDA

GCPDecompositions.gcp(
    X::CuArray,
    r,
    loss = LeastSquaresLoss();
    constraints = (),
    algorithm = GCPAlgorithms.ALS(),
) = _gcp(X, r, loss, constraints, algorithm)
function _gcp(
    X::CuArray{TX,N},
    r,
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.ALS,
) where {TX<:Real,N}
    T = promote_type(TX, Float32)

    # Random initialization
    M0 = CPD(ones(T, r), rand.(T, size(X), r))
    #M0norm = sqrt(mapreduce(abs2, +, M0[I] for I in CartesianIndices(size(M0))))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(mapreduce(x -> isnan(x) ? 0 : abs2(x), +, X, init=0f0))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    λ, U = M0.λ, collect(M0.U)

    # Move λ, U to gpu
    λ = CuArray(λ)
    U = [CuArray(U[i]) for i in 1:N]

    # Inefficient but simple implementation
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, U[i]'U[i] for i in setdiff(1:N, n))
            U[n] = GCPDecompositions.mttkrp(X, U, n) / V
            λ = CuArray(CUDA.norm.(eachcol(U[n])))
            U[n] = U[n] ./ permutedims(λ)
        end
    end

    return CPD(Array(λ), Tuple([Array(U[i]) for i in 1:N]))
end

end