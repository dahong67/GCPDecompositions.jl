using GCPDecompositions
using Test
using OffsetArrays

TESTLIST = [
    "CPD type" => "type-cpd.jl",
]

@testset verbose = true "$name" for (name, path) in TESTLIST
    include(path)
end
