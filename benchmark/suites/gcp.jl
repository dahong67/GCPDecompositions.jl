module BenchmarkGCP

using BenchmarkTools

using Random
using GCPDecompositions
using Distributions

const SUITE = BenchmarkGroup()

bench_gcp = SUITE

# Benchmark least squares loss 
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]
    bench_gcp["least-squares-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, LeastSquaresLoss()) 
end

# Benchmark Poisson loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(fill(10.0, r), rand.(sz, r))
    X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]
    bench_gcp["poisson-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, PoissonLoss())
end

# Benchmark Gamma loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    k = 1.5
    X = [rand(Gamma(k, M[I]/k)) for I in CartesianIndices(size(M))]
    bench_gcp["gamma-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, GammaLoss())
end

# Benchmark BernoulliOdds Loss
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [rand(Bernoulli(M[I]/(M[I] + 1))) for I in CartesianIndices(size(M))]
    bench_gcp["bernoulliOdds-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, BernoulliOddsLoss())
end

end