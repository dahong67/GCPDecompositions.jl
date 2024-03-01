module BenchmarkKhatriRao

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

## N=1 matrix
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for sz in [ntuple(n -> In, 1) for In in 30:30:90],
        r in [5; 30:30:90]
    ],
)

## N=2 matrices (balanced)
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for sz in [ntuple(n -> In, 2) for In in 30:30:90],
        r in [5; 30:30:90]
    ],
)

## N=3 matrices (balanced)
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for sz in [ntuple(n -> In, 3) for In in 30:30:90],
        r in [5; 30:30:90]
    ],
)

## N=3 matrices (imbalanced)
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for
        sz in [Tuple(circshift([30, 100, 1000], c)) for c in 0:2], r in [5; 30:30:90]
    ],
)

## N=4 matrices (balanced)
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for sz in [ntuple(n -> In, 4) for In in 30:30:90],
        r in [5; 30:30:90]
    ],
)

## N=4 matrices (imbalanced)
append!(
    SETUPS,
    [
        (; size = sz, rank = r) for
        sz in [Tuple(circshift([20, 40, 80, 500], c)) for c in 0:3], r in [5; 30:30:90]
    ],
)

# Generate random benchmarks
for SETUP in SETUPS
    Random.seed!(0)
    U = [randn(In, SETUP.rank) for In in SETUP.size]
    SUITE["size=$(SETUP.size), rank=$(SETUP.rank)"] =
        @benchmarkable(GCPDecompositions.khatrirao($U...), seconds = 2, samples = 5,)
end

end
