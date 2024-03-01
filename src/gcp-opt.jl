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

"""
    mttkrp(X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n.

Algorithm is based on Section III-B of the paper:
> **Fast Alternating LS Algorithms for High Order CANDECOMP/PARAFAC Tensor Factorizations**.
> Anh-Huy Phan, Petr Tichavský, Andrzej Cichocki.
> *IEEE Transactions on Signal Processing*, 2013.
> DOI: 10.1109/TSP.2013.2269903
"""
function mttkrp(X, U, n)
    # Dimensions
    N, I, r = length(U), Tuple(size.(U, 1)), (only∘unique)(size.(U, 2))
    (N == ndims(X) && I == size(X)) || throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))

    # Allocate output array G
    G = similar(U[n])

    # Choose appropriate multiplication order:
    # + n == 1: no splitting required
    # + n == N: no splitting required
    # + 1 < n < N: better to multiply "bigger" side out first
    #   + prod(I[1:n]) > prod(I[n:N]): better to multiply left-to-right
    #   + prod(I[1:n]) < prod(I[n:N]): better to multiply right-to-left
    if n == 1
        mul!(G, reshape(X, I[1], :), khatrirao(U[reverse(2:N)]...))
    elseif n == N
        mul!(G, transpose(reshape(X, :, I[N])), khatrirao(U[reverse(1:N-1)]...))
    elseif prod(I[1:n]) > prod(I[n:N])
        # Inner multiplication: left side
        kr_left = khatrirao(U[reverse(1:n-1)]...)
        L = reshape(transpose(reshape(X, :, prod(I[n:N]))) * kr_left, (I[n:N]..., r))

        # Outer multiplication: right side
        kr_right = khatrirao(U[reverse(n+1:N)]...)
        for j in 1:r
            mul!(
                view(G, :, j),
                reshape(selectdim(L, ndims(L), j), I[n], :),
                view(kr_right, :, j),
            )
        end
    else
        # Inner multiplication: right side
        kr_right = khatrirao(U[reverse(n+1:N)]...)
        R = reshape(reshape(X, prod(I[1:n]), :) * kr_right, (I[1:n]..., r))

        # Outer multiplication: left side
        kr_left = khatrirao(U[reverse(1:n-1)]...)
        for j in 1:r
            mul!(
                view(G, :, j),
                transpose(reshape(selectdim(R, ndims(R), j), :, I[n])),
                view(kr_left, :, j),
            )
        end
    end
    return G
end

"""
    khatrirao(A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc.
"""
function khatrirao(A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    # Special case: N = 1
    if N == 1
        return A[1]
    end

    # General case: N > 1
    r = size(A[1], 2)
    all(==(r),size.(A,2)) || throw(DimensionMismatch())
    R = ntuple(Val(N)) do k
        dims = (ntuple(i -> 1, Val(N - k))..., :, ntuple(i -> 1, Val(k - 1))..., r)
        return reshape(A[k], dims)
    end
    return reshape(broadcast(*, R...), :, r)
end
