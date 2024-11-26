## SymCP algorithms

@testitem "gradients-fullsym" begin
    using GCPDecompositions:
        GCPAlgorithms.LBFGSB,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U!,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    function form_fullsym_M(U::Matrix{T}) where {T}
        sz = size(U)[1]
        M = zeros(T, sz, sz, sz)
        for i1 in axes(U, 1), i2 in axes(U, 1), i3 in axes(U, 1)
            M[i1, i2, i3] = sum(U[i1, :] .* U[i2, :] .* U[i3, :])
        end
        return M
    end

    @testset "r=$r, sz=$sz" for r in [1, 2, 5], sz in [3, 10, 50]

        # Form fully symmetric rank-1 tensor
        U_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
            X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
        end

        # Check that computed and autodiff gradients at solution = 0
        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()
        S = (1, 1, 1)

        M_star = SymCPD(ones(r), (U_star,), (1, 1, 1))
        @test maximum([M_star[I] - X[I] for I in CartesianIndices(X)]) == 0.0

        GU = (similar(M_star.U[1]),)
        computed_grad_solution = grad_U!(GU, M_star, X, loss, false)[1]  # Without using simplified form for symmetric data
        computed_grad_solution_simplified = grad_U!(GU, M_star, X, loss, true)[1]  # With using simplified form for symmetric data
        @test computed_grad_solution ==
              zeros(eltype(computed_grad_solution), size(computed_grad_solution))
        @test computed_grad_solution_simplified == zeros(
            eltype(computed_grad_solution_simplified),
            size(computed_grad_solution_simplified),
        )

        objective(U) = norm(X - form_fullsym_M(U))^2
        auto_grad_solution = ForwardDiff.gradient(objective, M_star.U[1])
        @test auto_grad_solution ==
              zeros(eltype(auto_grad_solution), size(auto_grad_solution))

        # Check gradients at random init compared to autodiff
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU = (similar(M0.U[1]),)

        computed_grad = grad_U!(GU, M0, X, loss, false)[1]
        computed_grad_simplified = grad_U!(GU, M0, X, loss, true)[1]

        auto_grad = ForwardDiff.gradient(objective, M0.U[1])

        @test maximum([
            abs(computed_grad[I] - auto_grad[I]) for I in CartesianIndices(auto_grad)
        ]) <= 1e-6
        @test maximum([
            abs(computed_grad_simplified[I] - auto_grad[I]) for
            I in CartesianIndices(auto_grad)
        ]) <= 1e-6
    end
end

@testitem "gradients-partialsym" begin
    using GCPDecompositions:
        GCPAlgorithms.LBFGSB,
        symgcp,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        ngroups,
        GCPLosses.grad_U!,
        convertCPD,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    function form_partialsym_M(U1::Array{T}, U2::Array{T}) where {T}
        sz1 = size(U1)[1]
        sz2 = size(U2)[1]
        M = zeros(T, sz1, sz2, sz1)
        for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U1, 1)
            M[i1, i2, i3] = sum(U1[i1, :] .* U2[i2, :] .* U1[i3, :])
        end
        return M
    end

    @testset "r=$r, sz1=$sz1, sz2=$sz2" for r in [1, 2, 5], sz1 in [3, 10, 50], sz2 in [5, 15, 40]

        # Form fully symmetric rank-1 tensor
        U1_star = randn(sz1, r)
        U2_star = randn(sz2, r)
        X = zeros(sz1, sz2, sz1)
        for i1 in axes(U1_star, 1), i2 in axes(U2_star, 1), i3 in axes(U1_star, 1)
            X[i1, i2, i3] = sum(U1_star[i1, :] .* U2_star[i2, :] .* U1_star[i3, :])
        end

        # Check that computed and autodiff gradients at solution = 0
        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()
        S = (1, 2, 1)

        M_star = SymCPD(ones(r), (U1_star, U2_star), (1, 2, 1))
        @test maximum([M_star[I] - X[I] for I in CartesianIndices(X)]) == 0.0

        GU = (similar(M_star.U[1]), similar(M_star.U[2]))
        computed_grad_solution_U1, computed_grad_solution_U2 =
            grad_U!(GU, M_star, X, loss, false)
        computed_grad_solution_simplified_U1, computed_grad_solution_simplified_U2 =
            grad_U!(GU, M_star, X, loss, true)
        @test computed_grad_solution_U1 ==
              zeros(eltype(computed_grad_solution_U1), size(computed_grad_solution_U1))
        @test computed_grad_solution_simplified_U1 == zeros(
            eltype(computed_grad_solution_simplified_U1),
            size(computed_grad_solution_simplified_U1),
        )

        U1_sz = size(M_star.U[1])
        U2_sz = size(M_star.U[2])
        function vectorized_objective(U_vec::Vector)
            # Unflatten   
            U1 = reshape(U_vec[1:prod(U1_sz)], U1_sz)
            U2 = reshape(U_vec[prod(U1_sz)+1:end], U2_sz)
            return norm(X - form_partialsym_M(U1, U2))^2
        end
        vectorized_auto_grad_solution = ForwardDiff.gradient(
            vectorized_objective,
            vcat(vec(M_star.U[1]), vec(M_star.U[2])),
        )
        # Unflatten
        U1_auto_grad_solution = reshape(vectorized_auto_grad_solution[1:prod(U1_sz)], U1_sz)
        U2_auto_grad_solution =
            reshape(vectorized_auto_grad_solution[prod(U1_sz)+1:end], U2_sz)

        @test U1_auto_grad_solution ==
              zeros(eltype(U1_auto_grad_solution), size(U1_auto_grad_solution))
        @test U2_auto_grad_solution ==
              zeros(eltype(U2_auto_grad_solution), size(U2_auto_grad_solution))

        # Check gradients at random init compared to autodiff
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU = (similar(M0.U[1]), similar(M0.U[2]))

        computed_grad_U1, computed_grad_U2 = grad_U!(GU, M0, X, loss, false)
        computed_grad_simplified_U1, computed_grad_simplified_U2 = grad_U!(GU, M0, X, loss, true)

        vectorized_auto_grad =
            ForwardDiff.gradient(vectorized_objective, vcat(vec(M0.U[1]), vec(M0.U[2])))
        # Unflatten
        U1_auto_grad = reshape(vectorized_auto_grad[1:prod(U1_sz)], U1_sz)
        U2_auto_grad = reshape(vectorized_auto_grad[prod(U1_sz)+1:end], U2_sz)

        @test maximum([
            abs(computed_grad_U1[I] - U1_auto_grad[I]) for I in CartesianIndices(U1_auto_grad)
        ]) <= 1e-6
        @test maximum([
            abs(computed_grad_U2[I] - U2_auto_grad[I]) for I in CartesianIndices(U2_auto_grad)
        ]) <= 1e-6
        @test maximum([
            abs(computed_grad_simplified_U1[I] - U1_auto_grad[I]) for I in CartesianIndices(U1_auto_grad)
        ]) <= 1e-6
        @test maximum([
            abs(computed_grad_simplified_U2[I] - U2_auto_grad[I]) for I in CartesianIndices(U2_auto_grad)
        ]) <= 1e-6
    end
end