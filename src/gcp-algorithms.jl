## Algorithm types

module GCPAlgorithms

"""
    AbstractAlgorithm

Abstract type for GCP algorithms.
"""
abstract type AbstractAlgorithm end

include("gcp-algorithms/lbfgsb.jl")
include("gcp-algorithms/als.jl")

end
