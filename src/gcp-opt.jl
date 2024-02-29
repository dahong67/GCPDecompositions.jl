## GCP decomposition - full optimization

# Main fitting function
"""
    gcp(X::Array, r, loss = LeastSquaresLoss();
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
    X::Array,
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
default_algorithm(X::Array{<:Real}, r, loss::LeastSquaresLoss, constraints::Tuple{}) =
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
    # Use faster MTTKRPs algorithm
    faster_mttkrps!(GU, M, Y)
    return ntuple(Val(N)) do k
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

"""
    faster_mttkrps!(GU, M, X) 
    
    Algorithm for computing MTTKRP sequences is from "Fast Alternating LS Algorithms
    for High Order CANDECOMP/PARAFAC Tensor Factorizations" by Phan et al., specifically
    section III-C.
"""
function faster_mttkrps!(GU, M, X)

    N = ndims(X)
    R = size(M.U[1])[2]

    # Determine order of modes of MTTKRP to compute
    Jns = [prod(size(X)[1:n]) for n in 1:N]
    Kns = [prod(size(X)[n+1:end]) for n in 1:N]
    Kn_minus_ones = [prod(size(X)[n:end]) for n in 1:N]
    comp = Jns .<= Kn_minus_ones
    n_star = maximum(map(x -> comp[x] ? x : 0, 1:N))
    order = vcat([i for i in n_star:-1:1], [i for i in n_star+1:N])

    # Compute MTTKRPs recursively
    saved = similar(M.U[1], Jns[n_star], R)
    for n in order
        if n == n_star
            saved = reshape(X, (Jns[n], Kns[n])) * khatrirao(M.U[reverse(n+1:N)]...)
            mttkrps_helper!(GU, saved, M, n, "right", N, Jns, Kns)
        elseif n == n_star + 1
            if n == N
                mul!(GU[n], reshape(X, (Jns[n-1], Kns[n-1]))', khatrirao(M.U[reverse(1:n-1)]...))
            else
                saved = (khatrirao(M.U[reverse(1:n-1)]...)' * reshape(X, (Jns[n-1], Kns[n-1])))'
                mttkrps_helper!(GU, saved, M, n, "left", N, Jns, Kns)
            end  
        elseif n < n_star
            if n == 1
                for r in 1:R
                    mul!(view(GU[n], :, r), reshape(view(saved, :, r), (Jns[n], size(X)[n+1])), view(M.U[n+1], :, r))
                    #GU[n][:, r] = reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(M.U[n+1], :, r)
                end
            else
                saved = stack(reshape(view(saved, :, r), (Jns[n], size(X)[n+1])) * view(M.U[n+1], :, r) for r in 1:R)
                mttkrps_helper!(GU, saved, M, n, "right", N, Jns, Kns)
            end
        else
            if n == N
                GU[n] = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(M.U[n-1], :, r) for r in 1:R)
            else
                saved = stack(reshape(view(saved, :, r), (size(X)[n-1], Kns[n-1]))' * view(M.U[n-1], :, r) for r in 1:R)
                mttkrps_helper!(GU, saved, M, n, "left", N, Jns, Kns)
            end
        end
    end
end 


function mttkrps_helper!(GU, Zn, M, n, side, N, Jns, Kns)
    if side == "right"
        kr = khatrirao(M.U[reverse(1:n-1)]...)
        for r in 1:size(M.U[n])[2]
            mul!(view(GU[n], :, r), reshape(view(Zn, :, r), (Jns[n-1], size(M.U[n])[1]))', kr[:, r])
            #GU[n][:,r] = reshape(view(Zn, :, r), (Jns[n-1], size(M.U[n])[1]))' * kr[:, r]
        end
    elseif side == "left"
        kr = khatrirao(M.U[reverse(n+1:N)]...)
        for r in 1:size(M.U[n])[2]
            mul!(view(GU[n], :, r), reshape(view(Zn, :, r), (size(M.U[n])[1], Kns[n])), kr[:, r])
            #GU[n][:,r] = reshape(view(Zn, :, r), (size(M.U[n])[1], Kns[n])) * kr[:, r]
        end
    end
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

function khatrirao(A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    r = size(A[1],2)
    # @boundscheck all(==(r),size.(A,2)) || throw(DimensionMismatch())
    R = ntuple(Val(N)) do k
        dims = (ntuple(i->1,Val(N-k))..., :, ntuple(i->1,Val(k-1))..., r)
        return reshape(A[k],dims)
    end
    return reshape(broadcast(*, R...),:,r)
end
