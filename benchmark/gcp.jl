using BenchmarkTools
#const SUITE = BenchmarkGroup()

using Random
using GCPDecompositions

const SUITE = BenchmarkGroup()

bench = SUITE["gcp"] = BenchmarkGroup()

# Benchmark least squares loss 
for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]

    bench["least-squares"] = @benchmarkable gcp(X, r)
    #bench["least-squares"] = @benchmarkable gcp(X, r, LeastSquaresLoss()) 
end