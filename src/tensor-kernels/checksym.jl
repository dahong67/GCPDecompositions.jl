## Tensor Kernel: checksym

"""
    checksym(X, S)

Determine whether tensor X has the symmetry given by S.
X may have stronger symmetry than S
(i.e. if X is 3-way and fully symmetric, checksym will return true for S=(1,1,1), S=(1,2,3), S=(1,1,2), ...)
"""
function checksym(X::AbstractArray, S::NTuple, eps::T = 1e-10) where {T<:Real}
    sym = true
    ndims(X) == length(S) || throw(
        DimensionMismatch(
            "Number of modes in X must be same as number of modes given by S",
        ),
    )
    minimum([S...]) == 1 && maximum([S...]) <= ndims(X) ||
        throw(DimensionMismatch("Symmetric Groups must be numbered 1,2,... (max N)"))
    eps >= zero(eps) || throw(DomainError("`eps` must be nonnegative"))

    if !(Tuple(unique(S)) == S)  # Trivial case where number of symmetric groups = number of modes
        for I in CartesianIndices(X)
            for group in unique(S)
                sym_dims = findall(S .== group)
                for perm in index_permutations(tuple([Tuple(I)...]...), tuple(sym_dims...))
                    if abs(X[I] - X[perm...]) > eps
                        sym = false
                        return sym
                    end
                end
            end
        end
    end

    return sym
end

function index_permutations(indices::Tuple, sym_dims::Tuple)
    values_to_permute = getindex.(Ref(indices), sym_dims)
    permuted_values = permutations(values_to_permute)

    result = []
    for perm in permuted_values
        permuted_indices = [i for i in indices]
        for (dim, val) in zip(sym_dims, perm)
            permuted_indices[dim] = val
        end
        push!(result, tuple(permuted_indices...))
    end

    return result
end

function permutations(p::Tuple)
    if length(p) <= 1
        return (p,)
    end

    result = []
    for i in eachindex(p)
        fixed_element = p[i]
        remaining_elements = (p[1:i-1]..., p[i+1:end]...)
        for perm in permutations(remaining_elements)
            push!(result, (fixed_element, perm...))
        end
    end

    return result
end
