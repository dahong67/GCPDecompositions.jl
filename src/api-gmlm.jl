using GCPDecompositions, LinearAlgebra
using GCPDecompositions.TensorKernels

function gmlm(X, Y, r; loss = GCPLosses.LeastSquares())
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
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(P + Q))
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
    algorithm = GCPAlgorithms.LBFGSB(; iprint = -1)
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    vu = GCPDecompositions.GCPAlgorithms.lbfgsb(f, g!, vu0; lbfgsopts...)[2]
    VU = map(range -> reshape(vu[range], :, r), vec_ranges)
    return CPD(ones(r), VU)
end

function gmlm_objective(B::CPD, X, Y, loss)
    n = only(unique([length(X), length(Y)]))
    M = only(unique(size.(Y)))
    η = zeros(M)
    total = 0.0

    # Split B into predictor and response factors
    Q = length(size(X[1]))
    V = B.U[1:Q]
    U = B.U[Q+1:end]

    KR_V = khatrirao(reverse(V)...)
    for i in 1:n
        ωi = KR_V' * vec(X[i])
        copy!(η, CPD(ωi, U))
        Y_i = Y[i]
        for k in eachindex(Y_i, η)
            total += GCPLosses.value(loss, Y_i[k], η[k])
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

    η = zeros(M)
    Gi = zeros(M)

    _GU = [zero(GU[k]) for k in 1:P]
    _GV = [zero(GV[k]) for k in 1:Q]

    KR_V = khatrirao(reverse(V)...)
    KR_U = khatrirao(reverse(U)...)
    for i in 1:n
        ωi = KR_V' * vec(X[i])
        copy!(η, CPD(ωi, U))

        Gi .= GCPLosses.deriv.(Ref(loss), Y[i], η)

        # ---- update U-grad ----
        wi = KR_V' * vec(X[i])
        tmpU = mttkrps(Gi, U) .* Ref(Diagonal(wi))
        for k in 1:P
            _GU[k] .+= tmpU[k]
        end

        # ---- update V-grad ----
        zi = KR_U' * vec(Gi)
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