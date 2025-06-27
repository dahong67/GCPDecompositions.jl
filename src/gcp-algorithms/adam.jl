"""
    Adam

    Adam algorithm for stochastic optimization of GCP from https://epubs.siam.org/doi/epdf/10.1137/19M1266265 (Algorithm 5.1).
    Basing somewhat off of PyTorch implementation from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html.

Brief description of algorithm parameters:

  - `s::Int`                 : number of samples / batch size for uniform sampling. Ignored for stratified sampling. (default: `1`)
  - `α::Float`               : learning rate (default: `0.01`)
  - `betas::Tuple`           : exponential decay coefficients (default: `(0.9, 0.999)`)
  - `ϵ::Float`               : numerical stability constant (default: `1e-8`)
  - `τ::Int`                 : number of iterations per epoch (default:1000)
  - `κ::Int`                 : max number of bad epochs (default: `1`)
  - `ν::Float`               : learning rate decay after bad epoch (deafult: `0.1`)
  - `sampling_strategy::String`    : how to sample elements, options are `uniform` or ... (default: `uniform`)
  - `p::Int`                 : number of nonzero samples for stratified sampling (default: `1`)
  - `q::Int`                 : number of zero samples for stratified sampling (default: `1`)
  - `κ_factor::Float`        : minimum relative decrease in objective function to not be a bad epoch, smaller is more stringent (default: `0.999`)

"""

Base.@kwdef struct Adam <: AbstractAlgorithm
    s::Int         = 1
    α::Float64       = 0.01
    betas::Tuple{Float64,Float64}   = (0.9, 0.999)
    ϵ::Float64       = 1e-8
    τ::Int         = 1000
    κ::Int         = 1
    ν::Float64       = 0.1
    sampling_strategy::String  = "uniform"
    p::Int          = 1
    q::Int          = 1
    κ_factor        = 1
end

function _symgcp(
    X::Array{TX,N},
    r,
    S::NTuple{N,Int},
    sym_data_eps,
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.Adam,
    init,
    γ,
) where {TX,N}
    T = promote_type(nonmissingtype(TX), Float64)

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

    # Check if data is symmetric (if it is, gradients can be simplified)
    #sym_data = checksym(X, S, sym_data_eps)
    sym_data = false

    # Initialization
    M0 = deepcopy(init)
    K = ngroups(M0)
    U  = deepcopy(M0.U)                               # Current factor matrices
    λ = deepcopy(M0.λ)                                # Current weights
    B_U = tuple([zeros(size(U[k])) for k in 1:K]...)  # First-order moment estimate for factor matrices
    B_λ = zeros(size(λ))                              # First-order moment estimate for weights
    C_U = tuple([zeros(size(U[k])) for k in 1:K]...)  # Second-order moment estimate for factor matrices
    C_λ = zeros(size(λ))                              # Second-order moment estimate for weights
    c = 0
    t = 1
    lr = algorithm.α

    # For stratified sampling, keep list of indices where data tensor is nonzero
    if algorithm.sampling_strategy == "stratified"
        nonzero_idxs = findall(!=(0), X)
        p = algorithm.p
        q = algorithm.q
    end

    GU = tuple([similar(U[k]) for k in 1:K]...)   # Gradient buffer for factor matrices
    Gλ = similar(λ)   # Gradient buffer for weights

    # TODO: Estimate loss from fixed set of samples
    # (For now, just use entire tensor)
    F_hat = GCPLosses.objective(SymCPD(λ, U, S), X, loss, γ)
    epoch_losses = [] # Record loss from objective function (no regularization)
    epoch_reg_term_losses = [] # Record loss from regularization term
    epoch_times = [] # Record total elapsed time after every epoch
    push!(epoch_losses, GCPLosses.objective(SymCPD(λ, U, S), X, loss, 0))  # Keep track of losses without regularization term
   
    push!(epoch_reg_term_losses, γ * sum(sum((norm(U[k][:, r])^2 - 1)^2 for r in 1:size(U[1])[2]) for k in 1:maximum(S)))
    
    epochs = 1
    
    time_start = time()
    # Temporarily add global stopping criterion for experiments
    total_iters = 0
    max_iters = 10000000000. # Temporary
    while c <= algorithm.κ && total_iters < max_iters
        
        #  Save copies of factor matrices and weights, first- and second-order moment estimates in case of failed epoch
        U_old = deepcopy(U)
        λ_old = deepcopy(λ)
        B_U_old = deepcopy(B_U)
        B_λ_old = deepcopy(B_λ)
        C_U_old = deepcopy(C_U)
        C_λ_old = deepcopy(C_λ)

        # Save old loss estimate
        F_hat_old = deepcopy(F_hat)
        
        for _ in 1:algorithm.τ

            # Sample elements 
            if algorithm.sampling_strategy == "uniform"
                B = [CartesianIndex([rand(1:I) for I in size(X)]...) for _ in 1:algorithm.s]
            elseif algorithm.sampling_strategy == "stratified"
                B = Vector{CartesianIndex}(undef, p+q)
                B[1:p] = rand(nonzero_idxs, p)            # Sample p nonzero entries (with replacement)
                # Sample q zero entries (with replacement)
                zero_count = 0
                while zero_count < q
                    candidate_idx = rand(CartesianIndices(X)) 
                    if X[candidate_idx] == 0
                        B[p+1+zero_count] = candidate_idx
                        zero_count += 1
                    end
                end
            else
                error(
                    "The only supported sampling strategies are uniform and stratified",
                )
            end

            # Compute stochastic gradient
            if algorithm.sampling_strategy == "stratified"
                GCPLosses.stochastic_grad_U_λ!((GU..., Gλ), SymCPD(λ, U, S), X, loss, sym_data, γ, B, "stratified"; η=length(nonzero_idxs), p=p, q=q)
            else
                GCPLosses.stochastic_grad_U_λ!((GU..., Gλ), SymCPD(λ, U, S), X, loss, sym_data, γ, B, "uniform")
            end

            # Adam updates
            for k in 1:K
                B_U[k] .= algorithm.betas[1] * B_U[k] + (1 - algorithm.betas[1]) * GU[k]
                C_U[k] .= algorithm.betas[2] * C_U[k] + (1 - algorithm.betas[2]) * (GU[k] .^ 2)
                B_U_k_hat = B_U[k] ./ (1 - algorithm.betas[1]^t)   # Bias-corrected
                C_U_k_hat = C_U[k] ./ (1 - algorithm.betas[2]^t)   # Bias-corrected
                U[k] .= U[k] - lr * (B_U_k_hat ./ sqrt.(C_U_k_hat .+ algorithm.ϵ))

                # Enforce lower bounds via projection
                U[k] .= max.(U[k], lower)

            end
            B_λ .= algorithm.betas[1] * B_λ + (1 - algorithm.betas[1]) * Gλ
            C_λ .= algorithm.betas[2] * C_λ + (1 - algorithm.betas[2]) * (Gλ .^ 2)
            B_λ_hat = B_λ ./ (1 - algorithm.betas[1]^t)    # Bias-corrected
            C_λ_hat = C_λ ./ (1 - algorithm.betas[2]^t)    # Bias-corrected
            λ .= λ - lr * (B_λ_hat ./ sqrt.(C_λ_hat .+ algorithm.ϵ))

            # Enforce lower bounds via projection
            λ .= max.(λ, lower)

            t += 1

        end
        total_iters += algorithm.τ

        # TODO: Estimate loss from fixed set of samples, make necessary adjustments
        F_hat = GCPLosses.objective(SymCPD(λ, U, S), X, loss, γ)  # Currently using entire tensor
        push!(epoch_losses, GCPLosses.objective(SymCPD(λ, U, S), X, loss, 0))  # Keep track of losses without regularization term
        push!(epoch_reg_term_losses, γ * sum(sum((norm(U[k][:, r])^2 - 1)^2 for r in 1:size(U[1])[2]) for k in 1:maximum(S)))
        #println("Epoch ", epochs, " objective function proportional decrease: ", 1 - F_hat / F_hat_old, "      ", F_hat)
        if F_hat / F_hat_old > algorithm.κ_factor
            U = deepcopy(U_old)
            λ = deepcopy(λ_old)
            B_U = deepcopy(B_U_old)
            B_λ = deepcopy(B_λ_old)
            C_U = deepcopy(C_U_old)
            C_λ = deepcopy(C_λ_old)
            F_hat = F_hat_old
            t -= algorithm.τ
            lr *= algorithm.ν
            c += 1
        end
        epochs += 1
        push!(epoch_times, time() - time_start)
    end

    # Return final model and loss after each epoch
    return SymCPD(λ, U, S), epoch_losses, epoch_reg_term_losses, epoch_times

end