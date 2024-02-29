module BenchmarkGCPLarge

using BenchmarkTools, GCPDecompositions
using Random, Distributions

const SUITE = BenchmarkGroup()

# Benchmark Poisson loss
for sz in [(40,50,60), (80,90,100), (20,30,40,50), (15,20,25,30,35)], r in [1, 10, 100]
    Random.seed!(0)
    M = CPD(fill(10.0, r), rand.(sz, r))
    X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]
    SUITE["poisson-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, PoissonLoss())
end

# Benchmark Gamma loss
for sz in [(40,50,60), (80,90,100), (20,30,40,50), (15,20,25,30,35)], r in [1, 10, 100]
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    k = 1.5
    X = [rand(Gamma(k, M[I] / k)) for I in CartesianIndices(size(M))]
    SUITE["gamma-size(X)=$sz, rank(X)=$r"] = @benchmarkable gcp($X, $r, GammaLoss())
end

# Benchmark BernoulliOdds Loss
for sz in [(40,50,60), (80,90,100), (20,30,40,50), (15,20,25,30,35)], r in [1, 10, 100]
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [rand(Bernoulli(M[I] / (M[I] + 1))) for I in CartesianIndices(size(M))]
    SUITE["bernoulliOdds-size(X)=$sz, rank(X)=$r"] =
        @benchmarkable gcp($X, $r, BernoulliOddsLoss())
end

end
