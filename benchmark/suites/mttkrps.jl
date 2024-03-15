module BenchmarkMTTKRPS

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

# 3 modes, balanced
append!(
    SETUPS,
    [
        (; modes = 3, size = sz, rank = r) for
        sz in [ntuple(n -> In, 3) for In in [20, 50, 100, 200, 400]], r in [10, 100]
    ],
)

# 4 modes, balanced
append!(
    SETUPS,
    [
        (; modes = 4, size = sz, rank = r) for
        sz in [ntuple(n -> In, 4) for In in [20, 50, 100]], r in [10, 100]
    ],
)

# 5 modes, balanced
append!(
    SETUPS,
    [
        (; modes = 5, size = sz, rank = r) for
        sz in [ntuple(n -> In, 5) for In in [10, 30, 60]], r in [10, 100]
    ],
)

# Generate random benchmarks
for SETUP in SETUPS

    # Setup for ALS, do one iteration of MTTKRPs
    Random.seed!(0)
    X = randn(SETUP.size)
    T = Float64
    r = SETUP.rank
    N = SETUP.modes
    M0 = GCPDecompositions.CPD(ones(T, r), rand.(T, size(X), r))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    λ, U = M0.λ, collect(M0.U)

    SUITE["modes=$(SETUP.modes), size=$(SETUP.size), rank=$(SETUP.rank)"] =
        @benchmarkable(GCPDecompositions.mttkrps!($X, $U, $λ), seconds = 5, samples = 5,)
end

end
