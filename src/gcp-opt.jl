## GCP decomposition - full optimization

# Main fitting function
"""
    gcp(X::AbstractArray, r, loss = LeastSquaresLoss();
        constraints = default_constraints(loss),
        algorithm = default_algorithm(X, r, loss, constraints)) -> CPD

Compute an approximate rank-`r` CP decomposition of the tensor `X`
with respect to the loss function `loss` and return a `CPD` object.
The weights `λ` are constrained to all be one and `constraints` is a
`Tuple` of constraints on the factor matrices `U = (U[1],...,U[N])`.
Conventional CP corresponds to the default `LeastSquaresLoss()` loss
with no constraints (i.e., `constraints = ()`).

If the LossFunctions.jl package is also loaded,
`loss` can also be a loss function from that package.
Check `GCPDecompositions.LossFunctionsExt.SupportedLosses`
to see what losses are supported.

See also: `CPD`, `AbstractLoss`.
"""
gcp(
    X::AbstractArray,
    r,
    loss = LeastSquaresLoss();
    constraints = default_constraints(loss),
    algorithm = default_algorithm(X, r, loss, constraints),
) = _gcp(X, r, loss, constraints, algorithm)

# Choose constraints based on the domain of the loss function
function default_constraints(loss)
    dom = domain(loss)
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
default_algorithm(X::AbstractArray{<:Real}, r, loss::LeastSquaresLoss, constraints::Tuple{}) =
    GCPAlgorithms.ALS()
default_algorithm(X, r, loss, constraints) = GCPAlgorithms.LBFGSB()

# TODO: remove this `func, grad, lower` signature
# will require reworking how we do testing
_gcp(X::Array{TX,N}, r, func, grad, lower, lbfgsopts) where {TX,N} = _gcp(
    X,
    r,
    UserDefinedLoss(func; deriv = grad, domain = Interval(lower, +Inf)),
    (GCPConstraints.LowerBound(lower),),
    GCPAlgorithms.LBFGSB(; lbfgsopts...),
)
function _gcp(
    X::Array{TX,N},
    r,
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.LBFGSB,
) where {TX,N}
    # T = promote_type(nonmissingtype(TX), Float64)
    T = Float64    # LBFGSB.jl seems to only support Float64

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = domain(loss)
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

    # Random initialization
    M0 = CPD(ones(T, r), rand.(T, size(X), r))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    u0 = vcat(vec.(M0.U)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* size(X))...)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(N))
    function f(u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        return gcp_func(CPD(ones(T, r), U), X, loss)
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
    U = map(range -> reshape(u[range], :, r), vec_ranges)
    return CPD(ones(T, r), U)
end

# Objective function and gradient (w.r.t. `M.U`)
function gcp_func(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

function gcp_grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]

    # MTTKRPs (inefficient but simple)
    return ntuple(Val(N)) do k
        Yk = reshape(PermutedDimsArray(Y, [k; setdiff(1:N, k)]), size(X, k), :)
        Zk = similar(Yk, prod(size(X)[setdiff(1:N, k)]), ncomponents(M))
        for j in Base.OneTo(ncomponents(M))
            Zk[:, j] = reduce(kron, [view(M.U[i], :, j) for i in reverse(setdiff(1:N, k))])
        end
        mul!(GU[k], Yk, Zk)
        return rmul!(GU[k], Diagonal(M.λ))
    end
end

function _gcp(
    X::Array{TX,N},
    r,
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.ALS,
) where {TX<:Real,N}
    T = promote_type(TX, Float64)

    # Random initialization
    M0 = CPD(ones(T, r), rand.(T, size(X), r))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(sum(abs2, skipmissing(X)))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    λ, U = M0.λ, collect(M0.U)

    # Inefficient but simple implementation
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, U[i]'U[i] for i in setdiff(1:N, n))
            U[n] = mttkrp(X, U, n) / V
            λ = norm.(eachcol(U[n]))
            U[n] = U[n] ./ permutedims(λ)
        end
    end

    return CPD(λ, Tuple(U))
end

# GPU implementation
function _gcp(
    X::CuArray{TX,N},
    r,
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::GCPAlgorithms.ALS,
) where {TX<:Real,N}
    T = promote_type(TX, Float64)

    # Random initialization
    M0 = CPD(CUDA.ones(T, r), CUDA.rand.(T, size(X), r))
    M0norm = sqrt(sum(abs2, M0[I] for I in CartesianIndices(size(M0))))
    Xnorm = sqrt(mapreduce(x -> isnan(x) ? 0 : abs2(x), +, X))
    for k in Base.OneTo(N)
        M0.U[k] .*= (Xnorm / M0norm)^(1 / N)
    end
    λ, U = M0.λ, collect(M0.U)

    # Inefficient but simple implementation
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, U[i]'U[i] for i in setdiff(1:N, n))
            U[n] = mttkrp(X, U, n) / V
            λ = norm.(eachcol(U[n]))
            U[n] = U[n] ./ permutedims(λ)
        end
    end

    return CPD(λ, Tuple(U))
end

# inefficient but simple
function mttkrp(X, U, n)
    # Dimensions
    N, I, r = length(U), Tuple(size.(U, 1)), (only∘unique)(size.(U, 2))
    (N == ndims(X) && I == size(X)) || throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))

    # Matricized tensor (in mode n)
    Xn = reshape(permutedims(X, [n; setdiff(1:N, n)]), size(X, n), :)

    # Khatri-Rao product (in mode n)
    Zn = similar(U[1], prod(I[setdiff(1:N, n)]), r)
    for j in 1:r
        Zn[:, j] = reduce(kron, [view(U[i], :, j) for i in reverse(setdiff(1:N, n))])
    end

    # MTTKRP (in mode n)
    return Xn * Zn
end
