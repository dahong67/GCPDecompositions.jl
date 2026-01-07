## Algorithm: Adam

"""
    Adam

Stochastic gradient-based optimization with
**ada**ptive **m**oment estimation.

Brief description of standard Adam parameters:

  - `α::Float64`  : step size (default: `0.001`)
  - `β1::Float64` : exponential decay rate for first moment estimate (default: `0.9`)
  - `β2::Float64` : exponential decay rate for second moment estimate (default: `0.999`)
  - `ϵ::Float64`  : offset added to denominator for numerical stability (default: `1e-8`)

We employ a few common modifications with the following parameters:

  - `epochiters::Int`    : iterations per epoch (default: `1000`)
  - `maxepochs::Int`     : max number of epochs (default: `1000`)
  - `faildecay::Float64` : step size decay rate for failure to decrease (default: `0.1`)
  - `maxfails::Int`      : max number of failures (default: `1`)

And the final parameter defines what sampler to use:

  - `fsampler::AbstractSampler` : sampler to use for function value
  - `gsampler::AbstractSampler` : sampler to use for gradients
"""
Base.@kwdef struct Adam{FS<:AbstractSampler,GS<:AbstractSampler} <: AbstractAlgorithm
    α::Float64 = 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 1e-8
    epochiters::Int = 1000
    maxepochs::Int = 1000
    faildecay::Float64 = 0.1
    maxfails::Int = 1
    fsampler::FS
    gsampler::GS
end

function _gcp!(
    rng::AbstractRNG,
    M::CPD{Float64,N},
    X::Union{Array{<:Union{Real,Missing},N},SparseArrayCOO{<:Real,<:Integer,N}},
    loss::GCPLosses.AbstractLoss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.Adam,
) where {N}
    r = ncomps(M)
    T = Float64    # Simpler for now

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

    # Normalize / project the provided initialization
    normalizecomps!(M; dims = :λ, distribute_to = 1:ndims(M))
    M.U[1] .*= permutedims(sign.(M.λ))
    M.λ .= oneunit(T)
    project!(M, GCPConstraints.LowerBound(lower))

    # Setup fsampler
    fsampler = SampleOnce(X, algorithm.fsampler)

    # Initialize
    A = M.U       # factor matrices
    B = zero.(A)  # first moment estimates
    C = zero.(A)  # second moment estimates
    F = gcp_stoch_objective(rng, CPD(ones(T, r), A), X, loss, fsampler)  # objective function value
    G = similar.(A)  # gradients

    # Main loop
    nfails = 0
    niters = 0
    Aprev, Bprev, Cprev = similar.(A), similar.(B), similar.(C)
    for _ in 1:algorithm.maxepochs
        # Save copies of variables in case of a bad epoch
        for k in 1:N
            copyto!(Aprev[k], A[k])
            copyto!(Bprev[k], B[k])
            copyto!(Cprev[k], C[k])
        end
        Fprev = F

        # Epoch loop
        for _ in 1:algorithm.epochiters
            niters += 1
            gcp_stoch_grad_U!(rng, G, CPD(ones(T, r), A), X, loss, algorithm.gsampler)
            for k in 1:N
                B[k] .= algorithm.β1 .* B[k] .+ (1 - algorithm.β1) .* G[k]
                C[k] .= algorithm.β2 .* C[k] .+ (1 - algorithm.β2) .* G[k] .^ 2
                A[k] .=
                    max.(
                        lower,
                        A[k] .-
                        (algorithm.faildecay^nfails * algorithm.α) .*
                        (B[k] ./ (1 - algorithm.β1^niters)) ./
                        sqrt.((C[k] ./ (1 - algorithm.β2^niters)) .+ algorithm.ϵ),
                    )
            end
        end

        # Check failure and termination criterion
        F = gcp_stoch_objective(rng, CPD(ones(T, r), A), X, loss, fsampler)
        if F > Fprev
            @info "Failed epoch, rewinding to previous epoch"
            for k in 1:N
                copyto!(Aprev[k], A[k])
                copyto!(Bprev[k], B[k])
                copyto!(Cprev[k], C[k])
            end
            F = Fprev
            niters -= algorithm.epochiters
            nfails += 1
            nfails > algorithm.maxfails && break
        end
    end

    # Return final value
    return CPD(ones(T, r), A)
end
