module BenchmarkMTTKRPLarge

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

## Balanced order-4 tensors
append!(
    SETUPS,
    [
        (; size = sz, rank = r, mode = n) for
        sz in [ntuple(n -> In, 4) for In in 30:30:120], r in 30:30:180, n in 1:4
    ],
)

## Imbalanced order-4 tensors
append!(
    SETUPS,
    [
        (; size = sz, rank = r, mode = n) for sz in [(30, 60, 100, 1000), (1000, 100, 60, 30)],
        r in 100:100:300, n in 1:4
    ]
)

# Generate random benchmarks
for SETUP in SETUPS
    Random.seed!(0)
    X = randn(SETUP.size)
    U = [randn(In, SETUP.rank) for In in SETUP.size]
    SUITE["size=$(SETUP.size), rank=$(SETUP.rank), mode=$(SETUP.mode)"] = @benchmarkable(
        GCPDecompositions.mttkrp($X, $U, $(SETUP.mode)),
        seconds = 2,
        samples = 5,
    )
end

end
