module BenchmarkLeastSquares

using BenchmarkTools, GCPDecompositions
using Random, Distributions

const SUITE = BenchmarkGroup()

# More thorough benchmarks for least squares than gcp benchmarks

# 3 modes
for sz in [(15, 20, 25), (30, 40, 50), (60, 70, 80)], r in [1, 10, 50]
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, loss = GCPLosses.LeastSquaresLoss())
end

# 4 modes
for sz in [(15, 20, 25, 30), (30, 40, 50, 60)], r in [1, 10, 50]
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["least-squares-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, loss = GCPLosses.LeastSquaresLoss())
end

# 5 modes
for sz in [(15, 20, 25, 30, 35), (30, 30, 30, 30, 30)], r in [1, 10, 50]
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["least-squares-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, loss = GCPLosses.LeastSquaresLoss())
end


end