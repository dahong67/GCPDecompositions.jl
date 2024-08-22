"""
Generalized CP Decomposition module. Provides approximate CP tensor decomposition with respect to general losses.
"""
module GCPDecompositions

# Imports
import Base: ndims, size, show, summary
import Base: getindex
import Base: AbstractArray, Array
import LinearAlgebra: norm
using Base.Order: Ordering, Reverse
using IntervalSets: Interval
using Random: default_rng

# Exports
export CPD
export ncomps,
    normalizecomps, normalizecomps!, permutecomps, permutecomps!, sortcomps, sortcomps!
export gcp
export GCPLosses, GCPConstraints, GCPAlgorithms

include("tensor-kernels.jl")
include("cpd.jl")
include("gcp-losses.jl")
include("gcp-constraints.jl")
include("gcp-algorithms.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LossFunctionsExt.jl")
end

# Main fitting function

"""
    gcp(X::Array, r;
        loss = GCPLosses.LeastSquares(),
        constraints = default_constraints(loss),
        algorithm = default_algorithm(X, r, loss, constraints),
        init = default_init(X, r, loss, constraints, algorithm))

Compute an approximate rank-`r` CP decomposition of the tensor `X`
with respect to the loss function `loss` and return a `CPD` object.

Keyword arguments:
+ `constraints` : a `Tuple` of constraints on the factor matrices `U = (U[1],...,U[N])`.
+ `algorithm`   : algorithm to use

Conventional CP corresponds to the default `GCPLosses.LeastSquares()` loss
with the default of no constraints (i.e., `constraints = ()`).

If the LossFunctions.jl package is also loaded,
`loss` can also be a loss function from that package.
Check `GCPDecompositions.LossFunctionsExt.SupportedLosses`
to see what losses are supported.

See also: `CPD`, `GCPLosses`, `GCPConstraints`, `GCPAlgorithms`.
"""
gcp(
    X::Array,
    r;
    loss = GCPLosses.LeastSquares(),
    constraints = default_constraints(loss),
    algorithm = default_algorithm(X, r, loss, constraints),
    init = default_init(X, r, loss, constraints, algorithm),
) = GCPAlgorithms._gcp(X, r, loss, constraints, algorithm, init)

# Defaults

"""
    default_constraints(loss)

Return a default tuple of constraints for the loss function `loss`.

See also: `gcp`.
"""
function default_constraints(loss)
    dom = GCPLosses.domain(loss)
    if dom == Interval(-Inf, +Inf)
        return ()
    elseif dom == Interval(0.0, +Inf)
        return (GCPConstraints.LowerBound(0.0),)
    else
        error(
            "only loss functions with a domain of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        )
    end
end

"""
    default_algorithm(X, r, loss, constraints)

Return a default algorithm for the data tensor `X`, rank `r`,
loss function `loss`, and tuple of constraints `constraints`.

See also: `gcp`.
"""
default_algorithm(X::Array{<:Real}, r, loss::GCPLosses.LeastSquares, constraints::Tuple{}) =
    GCPAlgorithms.FastALS()
default_algorithm(X, r, loss, constraints) = GCPAlgorithms.LBFGSB()

"""
    default_init([rng=default_rng()], X, r, loss, constraints, algorithm)

Return a default initialization for the data tensor `X`, rank `r`,
loss function `loss`, tuple of constraints `constraints`, and
algorithm `algorithm`, using the random number generator `rng` if needed.

See also: `gcp`.
"""
default_init(X, r, loss, constraints, algorithm) =
    default_init(default_rng(), X, r, loss, constraints, algorithm)
function default_init(rng, X, r, loss, constraints, algorithm)
    # Generate CPD with random factors
    T, N = nonmissingtype(eltype(X)), ndims(X)
    T = promote_type(T, Float64)
    M = CPD(ones(T, r), rand.(rng, T, size(X), r))

    # Normalize
    Mnorm = norm(M)
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M.U[k] .*= (Xnorm / Mnorm)^(1 / N)
    end

    return M
end

end
