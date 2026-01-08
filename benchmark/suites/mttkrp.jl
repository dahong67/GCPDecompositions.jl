module BenchmarkMTTKRP

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

## Balanced order-3 tensors
append!(
    SETUPS,
    [
        (; size = sz, rank = r, mode = n) for
        sz in [ntuple(n -> In, 3) for In in 50:50:200], r in [10; 50:50:300], n in 1:3
    ],
)

# ## Balanced order-4 tensors
# append!(
#     SETUPS,
#     [
#         (; size = sz, rank = r, mode = n) for
#         sz in [ntuple(n -> In, 4) for In in 30:30:120], r in 30:30:180, n in 1:4
#     ],
# )

## Imbalanced tensors
append!(
    SETUPS,
    [
        (; size = sz, rank = r, mode = n) for sz in [(30, 100, 1000), (1000, 100, 30)],
        r in [10; 100:100:300], n in 1:3
    ],
)

# Generate random benchmarks
for SETUP in SETUPS
    Random.seed!(0)
    X = randn(SETUP.size)
    U = Tuple([randn(In, SETUP.rank) for In in SETUP.size])
    SUITE["size=$(SETUP.size), rank=$(SETUP.rank), mode=$(SETUP.mode)"] = @benchmarkable(
        GCPDecompositions.mttkrp($X, $U, $(SETUP.mode)),
        seconds = 2,
        samples = 5,
    )
end

end
