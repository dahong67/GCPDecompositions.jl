## CP decomposition type

"""
    CPD

Tensor decomposition type for the canonical polyadic decompositions (CPD)
of a tensor (i.e., a multi-dimensional array) `A`.
This is the return type of `gcp(_)`,
the corresponding tensor decomposition function.

If `M::CPD` is the decomposition object,
the weights `λ` and the factor matrices `U = (U[1],...,U[N])`
can be obtained via `M.λ` and `M.U`,
such that `A = Σ_j λ[j] U[1][:,j] ∘ ⋯ ∘ U[N][:,j]`.
"""
struct CPD{T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
    λ::Tλ
    U::NTuple{N,TU}
    function CPD{T,N,Tλ,TU}(λ, U) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}}
        Base.require_one_based_indexing(λ, U...)
        for k in Base.OneTo(N)
            size(U[k], 2) == length(λ) || throw(
                DimensionMismatch(
                    "U[$k] has dimensions $(size(U[k])) but λ has length $(length(λ))",
                ),
            )
        end
        return new{T,N,Tλ,TU}(λ, U)
    end
end
CPD(λ::Tλ, U::NTuple{N,TU}) where {T,N,Tλ<:AbstractVector{T},TU<:AbstractMatrix{T}} =
    CPD{T,N,Tλ,TU}(λ, U)

"""
    ncomponents(M::CPD)

Return the number of components in `M`.

See also: `ndims`, `size`.
"""
ncomponents(M::CPD) = length(M.λ)
ndims(::CPD{T,N}) where {T,N} = N

size(M::CPD{T,N}, dim::Integer) where {T,N} = dim <= N ? size(M.U[dim], 1) : 1
size(M::CPD{T,N}) where {T,N} = ntuple(d -> size(M, d), N)

function show(io::IO, mime::MIME{Symbol("text/plain")}, M::CPD{T,N}) where {T,N}
    # Compute displaysize for showing fields
    LINES, COLUMNS = displaysize(io)
    LINES_FIELD = max(LINES - 2 - N, 0) ÷ (1 + N)
    io_field = IOContext(io, :displaysize => (LINES_FIELD, COLUMNS))

    # Show summary and fields
    summary(io, M)
    println(io)
    println(io, "λ weights:")
    show(io_field, mime, M.λ)
    for k in Base.OneTo(N)
        println(io, "\nU[$k] factor matrix:")
        show(io_field, mime, M.U[k])
    end
end

function summary(io::IO, M::CPD)
    dimstring =
        ndims(M) == 0 ? "0-dimensional" :
        ndims(M) == 1 ? "$(size(M,1))-element" : join(map(string, size(M)), '×')
    ncomps = ncomponents(M)
    return print(
        io,
        dimstring,
        " ",
        typeof(M),
        " with ",
        ncomps,
        ncomps == 1 ? " component" : " components",
    )
end

function getindex(M::CPD{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck Base.checkbounds_indices(Bool, axes(M), I) || Base.throw_boundserror(M, I)
    val = zero(eltype(T))
    for j in Base.OneTo(ncomponents(M))
        val += M.λ[j] * prod(M.U[k][I[k], j] for k in Base.OneTo(ndims(M)))
    end
    return val
end
getindex(M::CPD{T,N}, I::CartesianIndex{N}) where {T,N} = getindex(M, Tuple(I)...)

norm(M::CPD, p::Real = 2) =
    p == 2 ? norm2(M) : norm((M[I] for I in CartesianIndices(size(M))), p)
function norm2(M::CPD{T,N}) where {T,N}
    V = reduce(.*, M.U[i]'M.U[i] for i in 1:N)
    return sqrt(abs(M.λ' * V * M.λ))
end

function plot_factors(M::CPD, X::Array{}, plot_types::Vector{Symbol}; graphsize::Tuple{Int, Int}=(800,600), titlesize::Int64=30, labelsize::Int64=20, colors::Vector{Symbol}=[:steelblue], title::String="GCP Tensor Decomposition", factor_names::Vector{String}=["Mode"])
	
    fig = Figure(size = graphsize)

	if length(colors) < ndims(M)
        colors = vcat(colors, fill(colors[end], length(plot_types) - length(colors)))
    end
    # Set up axes and plot each factor matrix
    for row in 1:ncomponents(M)
		
		for matrix in 1:ndims(M)
			ax = Axis(fig[row+1, matrix])
			if plot_types[matrix] == :barplot
				barplot!(ax, 1:size(X, matrix), LinearAlgebra.normalize(M.U[matrix][:, row], Inf); color = colors[matrix])
			elseif plot_types[matrix] == :lines
				lines!(ax, 1:size(X, matrix), LinearAlgebra.normalize(M.U[matrix][:, row], Inf); color = colors[matrix])
			elseif plot_types[matrix] == :scatter
				scatter!(ax, 1:size(X, matrix), LinearAlgebra.normalize(M.U[matrix][:, row], Inf); color = colors[matrix])
			end	
		end
	end
    # Link and hide axes
    for axis in 1:ndims(M)
		
        linkxaxes!(contents(fig[:, axis])...)
        linkyaxes!(contents(fig[:, axis])...)
    end

    # Add labels and super title
	for i in 1:ndims(M)
		if factor_names == ["Mode"]
			Label(fig[1, i], "$(factor_names[1]) $i"; tellwidth = false, fontsize = labelsize, font = "Bold Arial")
		else	
			Label(fig[1, i], "$(factor_names[i])"; tellwidth = false, fontsize = labelsize, font = "Bold Arial")
		end
	end
				
	

	fig[0, 1:ndims(M)] = Label(fig, title, fontsize = titlesize, halign = :center, valign = :bottom, tellwidth = false, font = "Bold Arial")
	fig
    
end
