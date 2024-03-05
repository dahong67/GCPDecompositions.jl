module BenchmarkGCP

using BenchmarkTools, GCPDecompositions
using Random, Distributions

const SUITE = BenchmarkGroup()

# Benchmark least squares loss 
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]
    SUITE["least-squares-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, GCPLosses.LeastSquaresLoss())
end

# Benchmark Poisson loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(fill(10.0, r), rand.(sz, r))
    X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]
    SUITE["poisson-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, GCPLosses.PoissonLoss())
end

# Benchmark Gamma loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    k = 1.5
    X = [rand(Gamma(k, M[I] / k)) for I in CartesianIndices(size(M))]
    SUITE["gamma-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, GCPLosses.GammaLoss())
end

# Benchmark BernoulliOdds Loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [rand(Bernoulli(M[I] / (M[I] + 1))) for I in CartesianIndices(size(M))]
    SUITE["bernoulliOdds-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, GCPLosses.BernoulliOddsLoss())
end

end
