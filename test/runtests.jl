using GCPDecompositions
using Test
using OffsetArrays, Random

TESTLIST = [
    "CPD type" => "type-cpd.jl",
    "GCP full optimization" => "gcp-opt.jl",
]

@testset verbose = true "$name" for (name, path) in TESTLIST
    include(path)
end
