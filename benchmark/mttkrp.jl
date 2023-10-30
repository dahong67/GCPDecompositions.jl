using BenchmarkTools

using Random
using GCPDecompositions

const SUITE = BenchmarkGroup()

bench = SUITE["mttkrp"] = BenchmarkGroup()

szs = [(15, 20, 25), (30, 40, 50), (60, 70, 80), (80, 90, 100), (100, 110, 120)]
for sz in szs, r in 1:7

    Random.seed!(0)
    X = randn(sz)
    U = [randn(Ik,r) for Ik in sz]
    n = 1

    bench["mttkrp-size(X)=$sz, rank(X)=$r"] = @benchmarkable GCPDecompositions.mttkrp($X, $U, $n)
end