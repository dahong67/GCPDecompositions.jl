## SymCP algorithms

@testitem "gradients-fullsym" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.LBFGSB,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U_λ!,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    @testset "r=$r, sz=$sz" for r in [1, 2, 5], sz in [3, 10, 50]

        # Form tensor
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
        @test isapprox([M_star[I] - X[I] for I in CartesianIndices(X)], zeros(eltype(X), size(X)))

        GU = (similar(M_star.U[1]), similar(M_star.λ))
        computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M_star, X, loss, false)  # Without using simplified form for symmetric data
        computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M_star, X, loss, true)  # With using simplified form for symmetric data
        @test isapprox(computed_grad_solution_U, zeros(eltype(computed_grad_solution_U), size(computed_grad_solution_U)))
        @test isapprox(computed_grad_solution_λ, zeros(eltype(computed_grad_solution_λ), size(computed_grad_solution_λ)))
        @test isapprox(computed_grad_solution_simplified_U, zeros(eltype(computed_grad_solution_simplified_U), size(computed_grad_solution_simplified_U)))
        @test isapprox(computed_grad_solution_simplified_λ, zeros(eltype(computed_grad_solution_simplified_λ), size(computed_grad_solution_simplified_λ)))

        function form_fullsym_M(U_λ_vec::Vector{T}) where {T}
            U = reshape(U_λ_vec[1:sz*r], (sz, r))
            λ = U_λ_vec[sz*r+1:end]
            return reshape(khatrirao(U, U, U) * λ, (sz, sz, sz))
        end
        objective(Uλ_vec) = norm(X - form_fullsym_M(Uλ_vec))^2
        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M_star.U[1]), M_star.λ))
        auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M_star.U[1]))
        auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]
        @test isapprox(auto_grad_solution_U, zeros(eltype(auto_grad_solution_U), size(auto_grad_solution_U)), atol=1e-12)
        @test isapprox(auto_grad_solution_λ, zeros(eltype(auto_grad_solution_λ), size(auto_grad_solution_λ)), atol=1e-12)

        # Check gradients at random init compared to autodiff
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU = (similar(M0.U[1]), similar(M0.λ))

        computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false)
        computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true)

        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M0.U[1]), M0.λ))
        auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M0.U[1]))
        auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]

        @test isapprox(computed_grad_solution_U, auto_grad_solution_U, rtol=1e-6)
        @test isapprox(computed_grad_solution_λ, auto_grad_solution_λ, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_U, auto_grad_solution_U, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_λ, auto_grad_solution_λ, rtol=1e-6)

    end
end

@testitem "gradients-partialsym" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.LBFGSB,
        symgcp,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        ngroups,
        GCPLosses.grad_U_λ!,
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

    @testset "r=$r, sz1=$sz1, sz2=$sz2" for r in [1, 5], sz1 in [3, 50], sz2 in [5, 40]

        # Form tensor 
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
        @test isapprox([M_star[I] - X[I] for I in CartesianIndices(X)], zeros(eltype(X), size(X)))

        GU = (similar(M_star.U[1]), similar(M_star.U[2]), similar(M_star.λ))
        computed_grad_solution_U1, computed_grad_solution_U2, computed_grad_solution_λ = grad_U_λ!(GU, M_star, X, loss, false)  # Without using simplified form for symmetric data
        computed_grad_solution_simplified_U1, computed_grad_solution_simplified_U2, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M_star, X, loss, true)  # With using simplified form for symmetric data
        @test isapprox(computed_grad_solution_U1, zeros(eltype(computed_grad_solution_U1), size(computed_grad_solution_U1)))
        @test isapprox(computed_grad_solution_U2, zeros(eltype(computed_grad_solution_U2), size(computed_grad_solution_U2)))
        @test isapprox(computed_grad_solution_λ, zeros(eltype(computed_grad_solution_λ), size(computed_grad_solution_λ)))
        @test isapprox(computed_grad_solution_simplified_U1, zeros(eltype(computed_grad_solution_simplified_U1), size(computed_grad_solution_simplified_U1)))
        @test isapprox(computed_grad_solution_simplified_U2, zeros(eltype(computed_grad_solution_simplified_U2), size(computed_grad_solution_simplified_U2)))
        @test isapprox(computed_grad_solution_simplified_λ, zeros(eltype(computed_grad_solution_simplified_λ), size(computed_grad_solution_simplified_λ)))

        function form_partialsym_M(U_λ_vec::Vector{T}) where {T}
            U1 = reshape(U_λ_vec[1:sz1*r], (sz1, r))
            U2 = reshape(U_λ_vec[sz1*r+1:(sz1+sz2)*r], (sz2, r))
            λ = U_λ_vec[(sz1+sz2)*r+1:end]
            return reshape(khatrirao(U1, U2, U1) * λ, (sz1, sz2, sz1))
        end
        objective(Uλ_vec) = norm(X - form_partialsym_M(Uλ_vec))^2

        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M_star.U[1]), vec(M_star.U[2]), M_star.λ))
        auto_grad_solution_U1 = reshape(auto_grad_solution[1:sz1*r], size(M_star.U[1]))
        auto_grad_solution_U2 = reshape(auto_grad_solution[sz1*r+1:(sz1+sz2)*r], size(M_star.U[2]))
        auto_grad_solution_λ = auto_grad_solution[(sz1+sz2)*r+1:end]

        @test isapprox(auto_grad_solution_U1, zeros(eltype(auto_grad_solution_U1), size(auto_grad_solution_U1)), atol=1e-12)
        @test isapprox(auto_grad_solution_U2, zeros(eltype(auto_grad_solution_U2), size(auto_grad_solution_U2)), atol=1e-12)
        @test isapprox(auto_grad_solution_λ, zeros(eltype(auto_grad_solution_λ), size(auto_grad_solution_λ)), atol=1e-12)


        # Check gradients at random init compared to autodiff
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU = (similar(M0.U[1]), similar(M0.U[2]), similar(M0.λ))

        computed_grad_solution_U1, computed_grad_solution_U2, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false)
        computed_grad_solution_simplified_U1, computed_grad_solution_simplified_U2, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true)

        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M0.U[1]), vec(M0.U[2]), M0.λ))
        auto_grad_solution_U1 = reshape(auto_grad_solution[1:sz1*r], size(M_star.U[1]))
        auto_grad_solution_U2 = reshape(auto_grad_solution[sz1*r+1:(sz1+sz2)*r], size(M_star.U[2]))
        auto_grad_solution_λ = auto_grad_solution[(sz1+sz2)*r+1:end]

        @test isapprox(computed_grad_solution_U1, auto_grad_solution_U1, rtol=1e-6)
        @test isapprox(computed_grad_solution_U2, auto_grad_solution_U2, rtol=1e-6)
        @test isapprox(computed_grad_solution_λ, auto_grad_solution_λ, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_U1, auto_grad_solution_U1, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_U2, auto_grad_solution_U2, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_λ, auto_grad_solution_λ, rtol=1e-6)

    end
end