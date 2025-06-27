module BenchmarkSymGCP

using BenchmarkTools, GCPDecompositions
using Random, Distributions

const SUITE = BenchmarkGroup()

# Benchmark fully symmteric, least squares loss with lbfgsb (default)
for sz in [10, 40, 100], r in 1:2
    Random.seed!(0)
    U_star = rand.(sz, r)
    M = SymCPD(ones(r), (U_star,), (1,1,1))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["fully-symmetric-least-squares-lbfgsb-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable symgcp($X, $r, (1,1,1); loss = GCPLosses.LeastSquares())
end

# Benchmark fully symmteric, least squares loss with adam
for sz in [10, 40], r in 1:2
    Random.seed!(0)
    U_star = rand.(sz, r)
    M = SymCPD(ones(r), (U_star,), (1,1,1))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["fully-symmetric-least-squares-adam-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable symgcp($X, $r, (1,1,1); loss = GCPLosses.LeastSquares(), algorithm=GCPAlgorithms.Adam(α=0.01, τ=1000, κ=1, ν=0.1, sampling_strategy="uniform", κ_factor=1, s=100))
end

end