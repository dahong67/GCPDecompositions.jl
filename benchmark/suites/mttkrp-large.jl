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
        sz in [ntuple(n -> In, 4) for In in 20:20:80], r in 20:20:120, n in 1:4
    ],
)

## Imbalanced order-4 tensors
append!(
    SETUPS,
    [
        (; size = sz, rank = r, mode = n) for sz in [(20, 40, 80, 500), (500, 80, 40, 20)],
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
