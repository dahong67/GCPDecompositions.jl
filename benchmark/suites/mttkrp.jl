module BenchmarkMTTKRP

using BenchmarkTools

using Random
using GCPDecompositions

const SUITE = BenchmarkGroup()

bench_mttkrp = SUITE

szs = [10,30,50,80,120,200]
shapes = [(sz, sz, sz) for sz in szs]
n = 1
rs = 20:20:200

for sz in shapes, r in rs
    Random.seed!(0)
    X = randn(sz)
    U = [randn(Ik,r) for Ik in sz]
    bench_mttkrp["mttkrp-size(X)=$sz, rank(X)=$r"] = @benchmarkable GCPDecompositions.mttkrp($X, $U, $n)
end

end