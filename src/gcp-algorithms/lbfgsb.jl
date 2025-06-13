## Algorithm: LBFGSB

"""
    LBFGSB

**L**imited-memory **BFGS** with **B**ox constraints.

Brief description of algorithm parameters:

  - `m::Int`         : max number of variable metric corrections (default: `10`)
  - `factr::Float64` : function tolerance in units of machine epsilon (default: `1e7`)
  - `pgtol::Float64` : (projected) gradient tolerance (default: `1e-5`)
  - `maxfun::Int`    : max number of function evaluations (default: `15000`)
  - `maxiter::Int`   : max number of iterations (default: `15000`)
  - `iprint::Int`    : verbosity (default: `-1`)
      + `iprint < 0` means no output
      + `iprint = 0` prints only one line at the last iteration
      + `0 < iprint < 99` prints `f` and `|proj g|` every `iprint` iterations
      + `iprint = 99` prints details of every iteration except n-vectors
      + `iprint = 100` also prints the changes of active set and final `x`
      + `iprint > 100` prints details of every iteration including `x` and `g`

See documentation of [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl) for more details.
"""
Base.@kwdef struct LBFGSB <: AbstractAlgorithm
    m::Int         = 10
    factr::Float64 = 1e7
    pgtol::Float64 = 1e-5
    maxfun::Int    = 15000
    maxiter::Int   = 15000
    iprint::Int    = -1
end

function _gcp(
    X::Array{TX,N},
    r,
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.LBFGSB,
    init,
) where {TX,N}
    # T = promote_type(nonmissingtype(TX), Float64)
    T = Float64    # LBFGSB.jl seems to only support Float64

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = GCPLosses.domain(loss)
    if dom == Interval(-Inf, +Inf)
        lower in (-Inf, 0.0) || error(
            "only lower bound constraints of `-Inf` or `0` are (currently) supported for loss functions with a domain of `-Inf .. Inf`",
        )
    elseif dom == Interval(0.0, +Inf)
        lower == 0.0 || error(
            "only lower bound constraints of `0` are (currently) supported for loss functions with a domain of `0 .. Inf`",
        )
    else
        error(
            "only loss functions with a domain of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        )
    end

    # Initialization
    M0 = deepcopy(init)
    u0 = vcat(vec.(M0.U)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* size(X))...)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(N))
    function f(u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        return GCPLosses.objective(CPD(ones(T, r), U), X, loss)
    end
    function g!(gu, u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        GU = map(range -> reshape(view(gu, range), :, r), vec_ranges)
        GCPLosses.grad_U!(GU, CPD(ones(T, r), U), X, loss)
        return gu
    end

    # Run LBFGSB
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    u = lbfgsb(f, g!, u0; lb = fill(lower, length(u0)), lbfgsopts...)[2]
    U = map(range -> reshape(u[range], :, r), vec_ranges)
    return CPD(ones(T, r), U)
end

function _symgcp(
    X::Array{TX,N},
    r,
    S::NTuple{N,Int},
    sym_data_eps,
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.LBFGSB,
    init,
    γ,
) where {TX,N}
    # T = promote_type(nonmissingtype(TX), Float64)
    T = Float64    # LBFGSB.jl seems to only support Float64

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = GCPLosses.domain(loss)
    if dom == Interval(-Inf, +Inf)
        lower in (-Inf, 0.0) || error(
            "only lower bound constraints of `-Inf` or `0` are (currently) supported for loss functions with a domain of `-Inf .. Inf`",
        )
    elseif dom == Interval(0.0, +Inf)
        lower == 0.0 || error(
            "only lower bound constraints of `0` are (currently) supported for loss functions with a domain of `0 .. Inf`",
        )
    else
        error(
            "only loss functions with a domain of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        )
    end

    # Initialization
    M0 = deepcopy(init)
    u_λ_0 = vcat(vec.(M0.U)..., M0.λ)
    K = ngroups(M0)

    # Check if data is symmetric (if it is, gradients can be simplified)
    #sym_data = checksym(X, S, sym_data_eps)
    sym_data = false

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, (cumsum(r .* tuple([size(M0.U[k])[1] for k in 1:K]...))...), sum((length(M0.U[k]) for k in 1:K)) + r)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(K+1))

    # losses = []
    # reg_term_losses = []
    # times = []
    # t0 = time()

    function f(u_λ)
        U = map(range -> reshape(view(u_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        λ = view(u_λ, vec_ranges[length(vec_ranges)])
        push!(losses, GCPLosses.objective(SymCPD(λ, U, S), X, loss, 0)) 
        push!(reg_term_losses, γ * sum(sum((norm(U[k][:, r])^2 - 1)^2 for r in 1:size(U[1])[2]) for k in 1:maximum(S)))
        push!(times, time() - t0)
        return GCPLosses.objective(SymCPD(λ, U, S), X, loss, γ)
    end

    function g!(gu_λ, u_λ)
        U = map(range -> reshape(view(u_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        λ = view(u_λ, vec_ranges[length(vec_ranges)])
        GU = map(range -> reshape(view(gu_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        Gλ = view(gu_λ, vec_ranges[length(vec_ranges)])
        GCPLosses.grad_U_λ!((GU..., Gλ), SymCPD(λ, U, S), X, loss, sym_data, γ)
        return gu_λ
    end

    # Run LBFGSB
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    u_λ = lbfgsb(f, g!, u_λ_0; lb = fill(lower, length(u_λ_0)), lbfgsopts...)[2]

    U = map(range -> reshape(u_λ[range], :, r), vec_ranges[1:length(vec_ranges)-1])
    λ = u_λ[vec_ranges[length(vec_ranges)]]

    return SymCPD(λ, U, S), losses, reg_term_losses, times
end