using BenchmarkTools

const SUITE = BenchmarkGroup()

module BenchmarkGCP
    include("gcp.jl")
end

SUITE["gcp"] = BenchmarkGCP.SUITE