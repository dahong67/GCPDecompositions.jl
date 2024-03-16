module BenchmarkMTTKRPS

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

# Balanced order-3 tensors
append!(
    SETUPS,
    [
        (; modes = 3, size = sz, rank = r) for
        sz in [ntuple(n -> In, 3) for In in [20, 50, 100, 200, 400]], r in [10, 100]
    ],
)

# Balanced order-4 tensors
append!(
    SETUPS,
    [
        (; modes = 4, size = sz, rank = r) for
        sz in [ntuple(n -> In, 4) for In in [20, 50, 100]], r in [10, 100]
    ],
)

# Balanced order-5 tensors
append!(
    SETUPS,
    [
        (; modes = 5, size = sz, rank = r) for
        sz in [ntuple(n -> In, 5) for In in [10, 30, 60]], r in [10, 100]
    ],
)

# Generate random benchmarks
for SETUP in SETUPS
    Random.seed!(0)
    X = randn(SETUP.size)
    U = Tuple([randn(In, SETUP.rank) for In in SETUP.size])

    SUITE["modes=$(SETUP.modes), size=$(SETUP.size), rank=$(SETUP.rank)"] =
        @benchmarkable(GCPDecompositions.mttkrps($X, $U), seconds = 5, samples = 5)
end

end
