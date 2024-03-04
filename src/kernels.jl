## Tensor Kernels

"""
    mttkrp(X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n.

See also: `mttkrp!`
"""
mttkrp(X, U, n) = mttkrp!(similar(U[n]), X, U, n)

"""
    mttkrp!(G, X, (U1, U2, ..., UN), n, buffer=create_mttkrp_buffer(X, U, n))

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n
and store the result in G.

Optionally, provide a `buffer` for intermediate calculations.
Always use `create_mttkrp_buffer` to make the `buffer`;
the internal details of `buffer` may change in the future
and should not be relied upon.

Algorithm is based on Section III-B of the paper:
> **Fast Alternating LS Algorithms for High Order
>   CANDECOMP/PARAFAC Tensor Factorizations**.
> Anh-Huy Phan, Petr Tichavský, Andrzej Cichocki.
> *IEEE Transactions on Signal Processing*, 2013.
> DOI: 10.1109/TSP.2013.2269903

See also: `mttkrp`, `create_mttkrp_buffer`
"""
function mttkrp!(G, X, U, n, buffer = create_mttkrp_buffer(X, U, n))
    # Dimensions
    Base.require_one_based_indexing(G, X, U...)
    N, I, r = length(U), Tuple(size.(U, 1)), (only ∘ unique)(size.(U, 2))
    (N == ndims(X) && I == size(X)) ||
        throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))
    n in 1:N || throw(DimensionMismatch("`n` must be in `1:ndims(X)`"))
    size(G) == size(U[n]) ||
        throw(DimensionMismatch("Output `G` must have the same size as `U[n]`"))

    # Choose appropriate multiplication order:
    # + n == 1: no splitting required
    # + n == N: no splitting required
    # + 1 < n < N: better to multiply "bigger" side out first
    #   + prod(I[1:n]) > prod(I[n:N]): better to multiply left-to-right
    #   + prod(I[1:n]) < prod(I[n:N]): better to multiply right-to-left
    if n == 1
        kr_right = n + 1 == N ? U[N] : khatrirao!(buffer.kr_right, U[reverse(n+1:N)]...)
        mul!(G, reshape(X, I[1], :), kr_right)
    elseif n == N
        kr_left = n == 2 ? U[1] : khatrirao!(buffer.kr_left, U[reverse(1:n-1)]...)
        mul!(G, transpose(reshape(X, :, I[N])), kr_left)
    else
        # Compute left and right Khatri-Rao products
        kr_left = n == 2 ? U[1] : khatrirao!(buffer.kr_left, U[reverse(1:n-1)]...)
        kr_right = n + 1 == N ? U[N] : khatrirao!(buffer.kr_right, U[reverse(n+1:N)]...)

        if prod(I[1:n]) > prod(I[n:N])
            # Inner multiplication: left side
            mul!(
                reshape(buffer.inner, :, r),
                transpose(reshape(X, :, prod(I[n:N]))),
                kr_left,
            )

            # Outer multiplication: right side
            for j in 1:r
                mul!(
                    view(G, :, j),
                    reshape(selectdim(buffer.inner, ndims(buffer.inner), j), I[n], :),
                    view(kr_right, :, j),
                )
            end
        else
            # Inner multiplication: right side
            mul!(reshape(buffer.inner, :, r), reshape(X, prod(I[1:n]), :), kr_right)

            # Outer multiplication: left side
            for j in 1:r
                mul!(
                    view(G, :, j),
                    transpose(
                        reshape(selectdim(buffer.inner, ndims(buffer.inner), j), :, I[n]),
                    ),
                    view(kr_left, :, j),
                )
            end
        end
    end
    return G
end

"""
    create_mttkrp_buffer(X, U, n)

Create buffer to hold intermediate calculations in `mttkrp!`.

Always use `create_mttkrp_buffer` to make a `buffer` for `mttkrp!`;
the internal details of `buffer` may change in the future
and should not be relied upon.

See also: `mttkrp!`
"""
function create_mttkrp_buffer(X, U, n)
    # Dimensions
    Base.require_one_based_indexing(X, U...)
    N, I, r = length(U), Tuple(size.(U, 1)), (only ∘ unique)(size.(U, 2))
    (N == ndims(X) && I == size(X)) ||
        throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))
    n in 1:N || throw(DimensionMismatch("`n` must be in `1:ndims(X)`"))

    # Allocate buffers
    return (;
        kr_left = n in 1:2 ? nothing : similar(U[1], prod(I[1:n-1]), r),
        kr_right = n in N-1:N ? nothing : similar(U[n+1], prod(I[n+1:N]), r),
        inner = n in [1, N] ? nothing :
                prod(I[1:n]) > prod(I[n:N]) ? similar(U[n], I[n:N]..., r) :
                similar(U[n], I[1:n]..., r),
    )
end

"""
    khatrirao(A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc.
"""
function khatrirao(A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    I, r = _checked_khatrirao_dims(A...)
    return khatrirao!(similar(A[1], prod(I), r), A...)
end

"""
    khatrirao!(K, A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc. and store the result in `K`.
"""
function khatrirao!(K::T, A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    I, r = _checked_khatrirao_dims(A...)

    # Check output dimensions
    Base.require_one_based_indexing(K)
    size(K) == (prod(I), r) || throw(
        DimensionMismatch(
            "Output `K` must have size equal to `(prod(size.(A,1)), size(A[1],2))",
        ),
    )

    # Special case: N = 1
    if N == 1
        K .= A[1]
        return K
    end

    # Base case: N = 2
    if N == 2
        reshape(K, I[2], I[1], r) .= reshape(A[2], :, 1, r) .* reshape(A[1], 1, :, r)
        return K
    end

    # Recursive case: N > 2
    n = argmin(n -> I[n] * I[n+1], 1:N-1)
    return khatrirao!(K, A[1:n-1]..., khatrirao(A[n], A[n+1]), A[n+2:end]...)
end

"""
    _checked_khatrirao_dims(A1, A2, ...)

Check that `A1`, `A2`, etc. have compatible dimensions for the Khatri-Rao product.
If so, return a tuple of the number of rows and the shared number of columns.
If not, throw an error.
"""
function _checked_khatrirao_dims(A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    Base.require_one_based_indexing(A...)
    allequal(size.(A, 2)) || throw(
        DimensionMismatch(
            "Matrices in a Khatri-Rao product must have the same number of columns.",
        ),
    )
    return size.(A, 1), size(A[1], 2)
end
