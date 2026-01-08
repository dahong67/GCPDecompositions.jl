"""
Generalized CP Decomposition module.
Provides approximate CP tensor decompositions with respect to general losses.
"""
module GCPDecompositions

# Imports
using Base.Order: Ordering, Reverse
using ForwardDiff: ForwardDiff
using IntervalSets: Interval
using LBFGSB: lbfgsb
using LinearAlgebra: Diagonal, LinearAlgebra, lu!, mul!, norm, rdiv!, rmul!
using Random: AbstractRNG, default_rng
using SparseArrays: sparse
using StatsBase: sample, sample!

# Tensor Kernels
include("base-kernels/khatrirao.jl")
include("base-kernels/mttkrp.jl")
include("base-kernels/mttkrps.jl")
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!

# Tensor Types
include("base-types/sparsearraycoo.jl")
include("base-types/cpd.jl")
export SparseArrayCOO, numstored
export CPD,
    ncomps,
    normalizecomps,
    normalizecomps!,
    permutecomps,
    permutecomps!,
    sortcomps,
    sortcomps!

# Losses
include("losses.jl")
export AbstractLoss, deriv, domain, value
export LeastSquaresLoss,
    NonnegativeLeastSquaresLoss,
    PoissonLoss,
    PoissonLogLoss,
    GammaLoss,
    RayleighLoss,
    BernoulliOddsLoss,
    BernoulliLogitLoss,
    NegativeBinomialOddsLoss,
    HuberLoss,
    BetaDivergenceLoss,
    UserDefinedLoss,
    WrappedLoss

# Constraints
include("constraints.jl")
export AbstractConstraint, project!, satisfies
export LowerBoundConstraint

## GCP decomposition
include("decomp-gcp/main.jl")
export gcp, default_gcp_constraints, default_gcp_algorithm, default_gcp_init

# Objective function and gradients
include("decomp-gcp/obj-grad.jl")
export gcp_objective, gcp_grad_U!

# Stochastic objective function and gradients
include("decomp-gcp/stoch-obj-grad/abstract.jl")
include("decomp-gcp/stoch-obj-grad/uniform.jl")
include("decomp-gcp/stoch-obj-grad/stratified.jl")
include("decomp-gcp/stoch-obj-grad/semistratified.jl")
export AbstractGCPSampler, GCPSampleOnce, gcp_stoch_objective, gcp_stoch_grad_U!
export UniformGCPSampler, StratifiedGCPSampler, SemistratifiedGCPSampler

# Algorithms
include("decomp-gcp/algorithms/abstract.jl")
include("decomp-gcp/algorithms/cp-als.jl")
include("decomp-gcp/algorithms/cp-fastals.jl")
include("decomp-gcp/algorithms/gcp-lbfgsb.jl")
include("decomp-gcp/algorithms/gcp-adam.jl")
export AbstractGCPAlgorithm, CP_ALS, CP_FastALS, GCP_Adam, GCP_LBFGSB

end
