## GCP decomposition - full optimization

# Main fitting function
"""
    gcp(X::Array, r, loss = GCPLosses.LeastSquaresLoss();
        constraints = default_constraints(loss),
        algorithm = default_algorithm(X, r, loss, constraints)) -> CPD

Compute an approximate rank-`r` CP decomposition of the tensor `X`
with respect to the loss function `loss` and return a `CPD` object.

Keyword arguments:
+ `constraints` : a `Tuple` of constraints on the factor matrices `U = (U[1],...,U[N])`.
+ `algorithm`   : algorithm to use

Conventional CP corresponds to the default `GCPLosses.LeastSquaresLoss()` loss
with the default of no constraints (i.e., `constraints = ()`).

If the LossFunctions.jl package is also loaded,
`loss` can also be a loss function from that package.
Check `GCPDecompositions.LossFunctionsExt.SupportedLosses`
to see what losses are supported.

See also: `CPD`, `GCPLosses`.
"""
gcp(
    X::Array,
    r,
    loss = GCPLosses.LeastSquaresLoss();
    constraints = default_constraints(loss),
    algorithm = default_algorithm(X, r, loss, constraints),
) = GCPAlgorithms._gcp(X, r, loss, constraints, algorithm)

# Choose constraints based on the domain of the loss function
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

# Choose default algorithm
default_algorithm(
    X::Array{<:Real},
    r,
    loss::GCPLosses.LeastSquaresLoss,
    constraints::Tuple{},
) = GCPAlgorithms.ALS()
default_algorithm(X, r, loss, constraints) = GCPAlgorithms.LBFGSB()
