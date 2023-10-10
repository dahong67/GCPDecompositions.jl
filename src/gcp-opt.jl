## GCP decomposition - full optimization

# Main fitting function
"""
    gcp(X::Array, r, loss = LeastSquaresLoss()) -> CPD

Compute an approximate rank-`r` CP decomposition of the tensor `X`
with respect to the loss function `loss` and return a `CPD` object.
Conventional CP corresponds to the default `LeastSquaresLoss()`.

See also: `CPD`, `AbstractLoss`.
"""
gcp(X::Array, r, loss::AbstractLoss = LeastSquaresLoss()) = _gcp(
    X,
    r,
    (x, m) -> value(loss, x, m),
    (x, m) -> deriv(loss, x, m),
    _factor_matrix_lower_bound(loss),
    (;),
)

# Choose lower bound on factor matrix entries based on the domain of the loss
function _factor_matrix_lower_bound(loss)
    # Get domain for the loss function
    dom = domain(loss)
    min, max = extrema(dom)

    # Throw errors for domains that are not supported
    dom isa Interval ||
        throw(DomainError(dom, "only domains of type `Interval` are (currently) supported"))
    max === +Inf || throw(
        DomainError(
            dom,
            "only domains of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        ),
    )
    min === -Inf ||
        iszero(min) ||
        throw(
            DomainError(
                dom,
                "only domains of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
            ),
        )

    # Return value
    return min
end

function _gcp(X::Array{TX,N}, r, func, grad, lower, lbfgsopts) where {TX,N}
    T = nonmissingtype(TX)

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
        return gcp_func(CPD(ones(T, r), U), X, func)
    end
    function g!(gu, u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        GU = map(range -> reshape(view(gu, range), :, r), vec_ranges)
        gcp_grad_U!(GU, CPD(ones(T, r), U), X, grad)
        return gu
    end

    # Run LBFGSB
    (lower === -Inf) || (lbfgsopts = (; lb = fill(lower, length(u0)), lbfgsopts...))
    u = lbfgsb(f, g!, u0; lbfgsopts...)[2]
    U = map(range -> reshape(u[range], :, r), vec_ranges)
    return CPD(ones(T, r), U)
end

# Objective function and gradient (w.r.t. `M.U`)
function gcp_func(M::CPD{T,N}, X::Array{TX,N}, func) where {T,TX,N}
    return sum(func(X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

function gcp_grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    grad,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : grad(X[I], M[I]) for
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
