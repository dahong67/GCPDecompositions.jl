using BenchmarkTools

const SUITE = BenchmarkGroup()

module BenchmarkGCP
    include("gcp.jl")
end

module BenchmarkMTTKRP
    include("mttkrp.jl")
end

SUITE["gcp"] = BenchmarkGCP.SUITE
SUITE["mttkrp"] = BenchmarkMTTKRP.SUITE