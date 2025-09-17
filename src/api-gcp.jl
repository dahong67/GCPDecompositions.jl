## Main GCP functions

# Main fitting function

"""
    gcp(X, r;
        rng = default_rng(),
        loss = GCPLosses.LeastSquares(),
        constraints = default_gcp_constraints(X, r, loss),
        algorithm = default_gcp_algorithm(X, r, loss, constraints),
        init = default_gcp_init(rng, X, r, loss, constraints, algorithm))

Compute an approximate rank-`r` CP decomposition of the data tensor `X`
with respect to the loss function `loss` and return a `CPD` object.

Keyword arguments:
+ `loss`        : loss function of type `GCPLosses.AbstractLoss`
+ `constraints` : a `Tuple` of constraints of type `GCPConstraints.AbstractConstraint`
+ `algorithm`   : algorithm of type `GCPAlgorithms.AbstractAlgorithm`
+ `rng`         : random number generator (used by `default_gcp_init` and some algorithms)

Conventional CP corresponds to the default `GCPLosses.LeastSquares()` loss
with the default of no constraints (i.e., `constraints = ()`).

If the LossFunctions.jl package is also loaded,
`loss` can also be a `DistanceLoss` or `MarginLoss` from that package;
`gcp` will automatically wrap it into a `GCPLosses.Wrapped` loss.

See also: `CPD`, `GCPLosses`, `GCPConstraints`, `GCPAlgorithms`.
"""
function gcp(
    X,
    r;
    rng = default_rng(),
    loss = GCPLosses.LeastSquares(),
    constraints = default_gcp_constraints(X, r, loss),
    algorithm = default_gcp_algorithm(X, r, loss, constraints),
    init = default_gcp_init(rng, X, r, loss, constraints, algorithm),
)
    # Normalize loss and constraints
    _loss = convert(GCPLosses.AbstractLoss, loss)
    _constraints =
        constraints isa GCPConstraints.AbstractConstraint ? tuple(constraints) :
        convert(Tuple{Vararg{GCPConstraints.AbstractConstraint}}, Tuple(constraints))

    # Check and copy init
    init isa CPD || throw(ArgumentError("`init` must be a `CPD`"))
    size(init) == size(X) || throw(ArgumentError("`init` must have the same size as `X`"))
    ncomps(init) == r || throw(ArgumentError("`init` must have `r` components"))
    _M = deepcopy(init)

    # Check if algorithm supports those inputs
    if !applicable(GCPAlgorithms._gcp!, rng, _M, X, _loss, _constraints, algorithm)
        error_str = """
        Algorithm `$(Base.nameof(typeof(algorithm)))` does not currently have an implementation supporting the provided types:
        + `X` is of type `$(typeof(X))`
        + `loss` is of type `$(typeof(_loss))`$(loss === _loss ? "" : " (converted from `$(typeof(loss))`)")
        + `constraints` is of type `$(typeof(_constraints))`$(constraints === _constraints ? "" : " (converted from `$(typeof(constraints))`)")
        + `init` is of type `$(typeof(_M))`
        + `rng` is of type `$(typeof(rng))`
        Please get in touch and let us know if you think it should - we are adding more methods over time!

        The currently implemented methods for `$(Base.nameof(typeof(algorithm)))` are:
        $(methods(GCPAlgorithms._gcp!, (Any, Any, Any, Any, Any, typeof(algorithm))))
        """
        throw(ErrorException(error_str))
    end

    # Call internal function with normalized inputs
    return GCPAlgorithms._gcp!(rng, _M, X, _loss, _constraints, algorithm)
end

# Default constraints

"""
    default_gcp_constraints(X, r, loss)

Return a default tuple of constraints for the data tensor `X`,
rank `r`, and loss function `loss`.

See also: `gcp`.
"""
default_gcp_constraints(X, r, loss) =
    default_gcp_constraints(X, r, convert(GCPLosses.AbstractLoss, loss))
function default_gcp_constraints(X, r, loss::GCPLosses.AbstractLoss)
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

# Default algorithm

"""
    default_gcp_algorithm(X, r, loss, constraints)

Return a default algorithm for the data tensor `X`, rank `r`,
loss function `loss`, and tuple of constraints `constraints`.

See also: `gcp`.
"""
default_gcp_algorithm(X, r, loss, constraints) =
    default_gcp_algorithm(X, r, convert(GCPLosses.AbstractLoss, loss), constraints)
default_gcp_algorithm(
    X::Array{<:Real},
    r,
    loss::GCPLosses.LeastSquares,
    constraints::Tuple{},
) = GCPAlgorithms.FastALS()
default_gcp_algorithm(X, r, loss::GCPLosses.AbstractLoss, constraints) =
    GCPAlgorithms.LBFGSB()

# Default initialization

"""
    default_gcp_init([rng=default_rng()], X, r, loss, constraints, algorithm)

Return a default initialization for the data tensor `X`, rank `r`,
loss function `loss`, tuple of constraints `constraints`, and
algorithm `algorithm`, using the random number generator `rng` if needed.

See also: `gcp`.
"""
default_gcp_init(X, r, loss, constraints, algorithm) =
    default_gcp_init(default_rng(), X, r, loss, constraints, algorithm)
default_gcp_init(rng, X, r, loss, constraints, algorithm) = default_gcp_init(
    rng,
    X,
    r,
    convert(GCPLosses.AbstractLoss, loss),
    constraints,
    algorithm,
)
function default_gcp_init(rng, X, r, loss::GCPLosses.AbstractLoss, constraints, algorithm)
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
