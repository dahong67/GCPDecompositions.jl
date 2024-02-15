module MTTKRPS

using BenchmarkTools, GCPDecompositions
using Random

const SUITE = BenchmarkGroup()

# Collect setups
const SETUPS = []

# 3 modes, balanced
append!(
    SETUPS,
    [
        (; modes=3, size = sz, rank = r) for
        sz in [ntuple(n -> In, 3) for In in 50:50:200], r in 50:50:300
    ],
)

# 4 modes, balanced
append!(
    SETUPS,
    [
        (; modes=4, size = sz, rank = r) for
        sz in [ntuple(n -> In, 4) for In in 50:50:200], r in 50:50:300
    ],
)

# 5 modes, balanced
append!(
    SETUPS,
    [
        (; modes=5, size = sz, rank = r) for
        sz in [ntuple(n -> In, 5) for In in 50:50:100], r in 50:50:300
    ],
)

# Generate random benchmarks
for SETUP in SETUPS
    Random.seed!(0)
    X = randn(SETUP.size)
    U = [randn(In, SETUP.rank) for In in SETUP.size]
    SUITE["modes=$(SETUP.modes), size=$(SETUP.size), rank=$(SETUP.rank)"] = @benchmarkable(
        GCPDecompositions.mttkrp($X, $U, $(SETUP.mode)),
        seconds = 5,
        samples = 5,
    )
end



end
