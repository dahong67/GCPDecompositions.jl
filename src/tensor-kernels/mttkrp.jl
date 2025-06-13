## Tensor Kernel: mttkrp

"""
    mttkrp(X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n.

See also: `mttkrp!`
"""
function mttkrp(
    X::AbstractArray{T,N},
    U::NTuple{N,TM},
    n::Integer,
) where {TM<:AbstractMatrix,T,N}
    _checked_mttkrp_dims(X, U, n)
    return mttkrp!(similar(U[n]), X, U, n)
end

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
function mttkrp!(
    G::TM,
    X::AbstractArray{T,N},
    U::NTuple{N,TU},
    n::Integer,
    buffer = create_mttkrp_buffer(X, U, n),
) where {TM<:AbstractMatrix,T,N,TU<:AbstractMatrix}
    I, r = _checked_mttkrp_dims(X, U, n)

    # Check output dimensions
    Base.require_one_based_indexing(G)
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
function create_mttkrp_buffer(
    X::AbstractArray{T,N},
    U::NTuple{N,TM},
    n::Integer,
) where {TM<:AbstractMatrix,T,N}
    I, r = _checked_mttkrp_dims(X, U, n)

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
    _checked_mttkrp_dims(X, (U1, U2, ..., UN), n)

Check that `X` and `U` have compatible dimensions for the mode-`n` MTTKRP.
If so, return a tuple of the number of rows and the shared number of columns
for the Khatri-Rao product. If not, throw an error.
"""
function _checked_mttkrp_dims(
    X::AbstractArray{T,N},
    U::NTuple{N,TM},
    n::Integer,
) where {TM<:AbstractMatrix,T,N}
    # Check mode
    n in 1:N || throw(DimensionMismatch("`n` must be in `1:ndims(X)`"))

    # Check Khatri-Rao product
    I, r = _checked_khatrirao_dims(U...)

    # Check tensor
    Base.require_one_based_indexing(X)
    (I == size(X)) ||
        throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))

    return I, r
end
