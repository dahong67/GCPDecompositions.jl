## Algorithm: GCP_LBFGSB

"""
    GCP_LBFGSB

**L**imited-memory **BFGS** with **B**ox constraints.

Algorithm parameters:

+ `m::Int`         : max number of variable metric corrections (default: `10`)
+ `factr::Float64` : function tolerance in units of machine epsilon (default: `1e7`)
+ `pgtol::Float64` : (projected) gradient tolerance (default: `1e-5`)
+ `maxfun::Int`    : max number of function evaluations (default: `15000`)
+ `maxiter::Int`   : max number of iterations (default: `15000`)
+ `iprint::Int`    : verbosity (default: `-1`)
    + `iprint < 0` means no output
    + `iprint = 0` prints only one line at the last iteration
    + `0 < iprint < 99` prints `f` and `|proj g|` every `iprint` iterations
    + `iprint = 99` prints details of every iteration except n-vectors
    + `iprint = 100` also prints the changes of active set and final `x`
    + `iprint > 100` prints details of every iteration including `x` and `g`

Notes:
+ this algorithm only supports `Float64` numbers

See documentation of [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl) for more details.
"""
Base.@kwdef struct GCP_LBFGSB <: AbstractGCPAlgorithm
    m::Int         = 10
    factr::Float64 = 1e7
    pgtol::Float64 = 1e-5
    maxfun::Int    = 15000
    maxiter::Int   = 15000
    iprint::Int    = -1
end

function _gcp!(
    rng::AbstractRNG,
    M::CPD{Float64,N},
    X::Array{<:Union{Real,Missing},N},
    loss::AbstractLoss,
    constraints::Tuple{Vararg{LowerBoundConstraint}},
    algorithm::GCP_LBFGSB,
) where {N}
    r = ncomps(M)
    T = Float64    # LBFGSB.jl seems to only support Float64

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = domain(loss)
    if dom == Interval(-Inf, +Inf)
        lower in (-Inf, 0.0) ||
            error("only lower bound constraints of `-Inf` or `0` are (currently) \
                   supported for loss functions with a domain of `-Inf .. Inf`")
    elseif dom == Interval(0.0, +Inf)
        lower == 0.0 || error("only lower bound constraints of `0` are (currently) \
                               supported for loss functions with a domain of `0 .. Inf`")
    else
        error("only loss functions with a domain of `-Inf .. Inf` \
               or `0 .. Inf` are (currently) supported")
    end

    # Initialization
    normalizecomps!(M; dims = :λ, distribute_to = 1:ndims(M))
    M.U[1] .*= permutedims(sign.(M.λ))
    M.λ .= oneunit(T)
    project!(M, LowerBoundConstraint(lower))
    U0 = M.U
    u0 = vcat(vec.(U0)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* size(X))...)
    vec_ranges = ntuple(k -> (vec_cutoffs[k]+1):vec_cutoffs[k+1], Val(N))
    function f(u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        return gcp_objective(CPD(ones(T, r), U), X, loss)
    end
    function g!(gu, u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        GU = map(range -> reshape(view(gu, range), :, r), vec_ranges)
        gcp_grad_U!(GU, CPD(ones(T, r), U), X, loss)
        return gu
    end

    # Run LBFGSB
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    u = lbfgsb(f, g!, u0; lb = fill(lower, length(u0)), lbfgsopts...)[2]
    for k in 1:N
        M.U[k] .= reshape(u[vec_ranges[k]], :, r)
    end
    return M
end
