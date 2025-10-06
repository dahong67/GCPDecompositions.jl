using GCPDecompositions, LinearAlgebra
using GCPDecompositions.TensorKernels

function gmlm(X, Y, r; loss=GCPLosses.LeastSquares())
	# Extract dimensions
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	N = only(unique(size.(X)))
	P, Q = length(M), length(N)

	# Initialization
	B0 = CPD(ones(r), rand.((N..., M...), r))
	vu0 = vcat(vec.(B0.U)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* (N..., M...))...)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(P+Q))
    function f(vu)
        VU = map(range -> reshape(view(vu, range), :, r), vec_ranges)
        return gmlm_objective(CPD(ones(r), VU), X, Y, loss)
    end
    function g!(gvu, vu)
        VU = map(range -> reshape(view(vu, range), :, r), vec_ranges)
        GVU = map(range -> reshape(view(gvu, range), :, r), vec_ranges)
        gmlm_grad!(GVU, CPD(ones(r), VU), X, Y, loss)
        return gvu
    end

	# Run LBFGSB
	algorithm = GCPAlgorithms.LBFGSB(iprint=0)
	lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    vu = GCPDecompositions.GCPAlgorithms.lbfgsb(f, g!, vu0; lbfgsopts...)[2]
    VU = map(range -> reshape(vu[range], :, r), vec_ranges)
	return CPD(ones(r), VU)
end

function contract!(result, Xi, B::Array)
    q = ndims(Xi)
    m_dims = size(B)[q+1:end]
    @assert size(result) == m_dims "Result array must have shape $(m_dims)"

    # Flatten Xi as a 1×P row vector (no copy)
    Xi_vec = reshape(Xi, 1, :)
    # Reshape B as a P×Q matrix
    B_mat  = reshape(B, size(Xi_vec, 2), :)
    # Flatten result as 1×Q row vector
    result_vec = reshape(result, 1, :)

    # In-place multiply 
    mul!(result_vec, Xi_vec, B_mat)

    return result
end

function contract!(result, Xi, B::CPD)
	temporal_B = Array(B)
    contract!(result, Xi, temporal_B)
end

function gmlm_objective(B, X, Y, loss)
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	η = zeros(M)  

	total = 0.0
    for i in 1:n
        contract!(η, X[i], B)
        @inbounds for j in CartesianIndices(M)
            total += GCPLosses.value(loss, Y[i][j], η[j])
        end
    end
    return total
end

function gmlm_grad!(GVU, B, X, Y, loss)
	n = only(unique([length(X), length(Y)]))
	M = only(unique(size.(Y)))
	N = only(unique(size.(X)))
	P, Q = length(M), length(N)

	V, U = collect.(B.U[1:Q]), collect.(B.U[Q+1:end])
	GV, GU = GVU[1:Q], GVU[Q+1:end]

	η  = zeros(M)    
    Gi = zeros(M)    

    _GU = [zero(GU[k]) for k in 1:P]
    _GV = [zero(GV[k]) for k in 1:Q]

    for i in 1:n
        contract!(η, X[i], B)

        # ---- loss derivatives ----
        @inbounds for j in CartesianIndices(M)
            Gi[j] = GCPLosses.deriv(loss, Y[i][j], η[j])
        end

        # ---- update U-grad ----
        wi = khatrirao(reverse(V)...)' * vec(X[i])
        tmpU = mttkrps(Gi, U) .* Ref(Diagonal(wi))
        for k in 1:P
            _GU[k] .+= tmpU[k]
        end

        # ---- update V-grad ----
        zi = khatrirao(reverse(U)...)' * vec(Gi)
        tmpV = mttkrps(X[i], V) .* Ref(Diagonal(zi))
        for k in 1:Q
            _GV[k] .+= tmpV[k]
        end
    end

    # Write results into GU / GV
    for k in 1:P
        GU[k] .= _GU[k]
    end
    for k in 1:Q
        GV[k] .= _GV[k]
    end

    return GVU
end