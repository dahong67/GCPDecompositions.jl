using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

const SUITE = BenchmarkGroup()

module BenchmarkGCP
    include("gcp.jl")
end

module BenchmarkMTTKRP
    include("mttkrp.jl")
end

SUITE["gcp"] = BenchmarkGCP.SUITE
SUITE["mttkrp"] = BenchmarkMTTKRP.SUITE