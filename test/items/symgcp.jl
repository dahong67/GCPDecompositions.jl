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
        computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M_star, X, loss, false, 0)  # Without using simplified form for symmetric data
        computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M_star, X, loss, true, 0)  # With using simplified form for symmetric data
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

        computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false, 0)
        computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true, 0)

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
        computed_grad_solution_U1, computed_grad_solution_U2, computed_grad_solution_λ = grad_U_λ!(GU, M_star, X, loss, false, 0)  # Without using simplified form for symmetric data
        computed_grad_solution_simplified_U1, computed_grad_solution_simplified_U2, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M_star, X, loss, true, 0)  # With using simplified form for symmetric data
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

        computed_grad_solution_U1, computed_grad_solution_U2, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false, 0)
        computed_grad_solution_simplified_U1, computed_grad_solution_simplified_U2, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true, 0)

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

@testitem "gradients-fullsym-reg" begin
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

        # Add regularization
        γ = 0.1

        # Form tensor
        U_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
            X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
        end

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()
        S = (1, 1, 1)
        M_star = SymCPD(ones(r), (U_star,), (1, 1, 1))

        function form_fullsym_M(U_λ_vec::Vector{T}) where {T}
            U = reshape(U_λ_vec[1:sz*r], (sz, r))
            λ = U_λ_vec[sz*r+1:end]
            return reshape(khatrirao(U, U, U) * λ, (sz, sz, sz))
        end
        function vec_to_col_norms(U_λ_vec::Vector{T}) where {T}
            # Get norms columns of factor matrices from vectorized form
            U = reshape(U_λ_vec[1:sz*r], (sz, r))
            return sum((norm.(eachcol(U)).^2 - ones(T, r)).^2)
        end
        objective(Uλ_vec) = norm(X - form_fullsym_M(Uλ_vec))^2 + γ * vec_to_col_norms(Uλ_vec)
        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M_star.U[1]), M_star.λ))
        auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M_star.U[1]))
        auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]

        # Check gradients at random init compared to autodiff
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU = (similar(M0.U[1]), similar(M0.λ))

        computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false, γ)
        computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true, γ)

        auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M0.U[1]), M0.λ))
        auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M0.U[1]))
        auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]

        @test isapprox(computed_grad_solution_U, auto_grad_solution_U, rtol=1e-6)
        @test isapprox(computed_grad_solution_λ, auto_grad_solution_λ, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_U, auto_grad_solution_U, rtol=1e-6)
        @test isapprox(computed_grad_solution_simplified_λ, auto_grad_solution_λ, rtol=1e-6)

    end
end

@testitem "stochastic-gradients-nonsym" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.Adam,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U_λ!,
        GCPLosses.stochastic_grad_U_λ!,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    @testset "r=$r, sz=$sz, γ=$γ" for r in [1, 5], sz in [3, 20], γ in [0, 0.1]

        # Form tensor
        U1_star = randn(sz, r)
        U2_star = randn(sz, r)
        U3_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U1_star, 1), i2 in axes(U2_star, 1), i3 in axes(U3_star, 1)
            X[i1, i2, i3] = sum(U1_star[i1, :] .* U2_star[i2, :] .* U3_star[i3, :])
        end

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = Adam()
        S = (1, 2, 3)

        # Check gradients at random init for stochastic with batch size equal to entire tensor and non-stochastic
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU_batch = (ntuple(i -> similar(M0.U[i]), length(M0.U))..., similar(M0.λ))
        grad_U_λ!(GU_batch, M0, X, loss, false, γ)
        batch_grad_U1 = GU_batch[1]
        batch_grad_U2 = GU_batch[2]
        batch_grad_U3 = GU_batch[3]
        batch_grad_λ = GU_batch[4]

        GU_stochastic = (ntuple(i -> similar(M0.U[i]), length(M0.U))..., similar(M0.λ))
        GU_stochastic_simplified = (ntuple(i -> similar(M0.U[i]), length(M0.U))..., similar(M0.λ))

        stochastic_grad_U_λ!(GU_stochastic, M0, X, loss, false, γ, CartesianIndices(X), "uniform")
        stochastic_grad_U1 = GU_stochastic[1]
        stochastic_grad_U2 = GU_stochastic[2]
        stochastic_grad_U3 = GU_stochastic[3]
        stochastic_grad_λ = GU_stochastic[4]
        stochastic_grad_U_λ!(GU_stochastic_simplified, M0, X, loss, true, γ, CartesianIndices(X), "uniform")
        stochastic_grad_U1_simplified = GU_stochastic_simplified[1]
        stochastic_grad_U2_simplified = GU_stochastic_simplified[2]
        stochastic_grad_U3_simplified = GU_stochastic_simplified[3]
        stochastic_grad_λ_simplified = GU_stochastic_simplified[4]

        @test isapprox(batch_grad_U1, stochastic_grad_U1, rtol=1e-6)
        @test isapprox(batch_grad_U2, stochastic_grad_U2, rtol=1e-6)
        @test isapprox(batch_grad_U3, stochastic_grad_U3, rtol=1e-6)
        @test isapprox(batch_grad_λ, stochastic_grad_λ, rtol=1e-6)
        @test isapprox(batch_grad_U1, stochastic_grad_U1_simplified, rtol=1e-6)
        @test isapprox(batch_grad_U2, stochastic_grad_U2_simplified, rtol=1e-6)
        @test isapprox(batch_grad_U3, stochastic_grad_U3_simplified, rtol=1e-6)
        @test isapprox(batch_grad_λ, stochastic_grad_λ_simplified, rtol=1e-6)

    end
end

@testitem "stochastic-gradients-fullsym" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.Adam,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U_λ!,
        GCPLosses.stochastic_grad_U_λ!,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    @testset "r=$r, sz=$sz, γ=$γ" for r in [1, 5], sz in [3, 20], γ in [0, 0.1]

        # Form tensor
        U_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
            X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
        end

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = Adam()
        S = (1, 1, 1)

        # Check gradients at random init for stochastic with batch size equal to entire tensor and non-stochastic
        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M0 = deepcopy(init)
        GU_batch = (similar(M0.U[1]), similar(M0.λ))
        grad_U_λ!(GU_batch, M0, X, loss, false, γ)
        batch_grad_U = GU_batch[1]
        batch_grad_λ = GU_batch[2]

        GU_stochastic = (similar(M0.U[1]), similar(M0.λ))
        GU_stochastic_simplified = (similar(M0.U[1]), similar(M0.λ))

        stochastic_grad_U_λ!(GU_stochastic, M0, X, loss, false, γ, CartesianIndices(X), "uniform")
        stochastic_grad_U = GU_stochastic[1]
        stochastic_grad_λ = GU_stochastic[2]
        stochastic_grad_U_λ!(GU_stochastic_simplified, M0, X, loss, true, γ, CartesianIndices(X), "uniform")
        stochastic_grad_U_simplified = GU_stochastic_simplified[1]
        stochastic_grad_λ_simplified = GU_stochastic_simplified[2]

        @test isapprox(batch_grad_U, stochastic_grad_U, rtol=1e-6)
        @test isapprox(batch_grad_λ, stochastic_grad_λ, rtol=1e-6)
        @test isapprox(batch_grad_U, stochastic_grad_U_simplified, rtol=1e-6)
        @test isapprox(batch_grad_λ, stochastic_grad_λ_simplified, rtol=1e-6)

    end
end

@testitem "stochastic-gradients-bias-uniform-nonsymmetric" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.Adam,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U_λ!,
        GCPLosses.stochastic_grad_U_λ!,
        SymCPD
    using LinearAlgebra: norm

    @testset "γ=$γ" for γ in [0, 0.1]

        r = 3
        sz = 5      # 125 total entries
        s = 50      # Sample size
        N = 100000   # Number of stochastic realizations

        # Form tensor
        U1_star = randn(sz, r)
        U2_star = randn(sz, r)
        U3_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U1_star, 1), i2 in axes(U2_star, 1), i3 in axes(U3_star, 1)
            X[i1, i2, i3] = sum(U1_star[i1, :] .* U2_star[i2, :] .* U3_star[i3, :])
        end

        U1_init = randn(sz, r)
        U2_init = randn(sz, r)
        U3_init = randn(sz, r)
        loss_func = LeastSquares()
        M = SymCPD(ones(r), (U1_init, U2_init, U3_init), (1,2,3))

        # Allocate for results
        GU_λ_batch = (ntuple(i -> similar(M.U[i]), length(M.U))..., similar(M.λ))
        GU_λ_stochastic = (ntuple(i -> similar(M.U[i]), length(M.U))..., similar(M.λ))
        stochastic_grads_vec = []

        # Compute batch gradient, concatenate and vectorize
        grad_U_λ!(GU_λ_batch, M, X, loss_func, false, γ)
        batch_grad_vec = vcat(vec.(GU_λ_batch)...)
        batch_grad_norm = norm(batch_grad_vec)

        # n stochastic realizations
        for _ in 1:N

            # Sample elements
            B = [CartesianIndex([rand(1:I) for I in size(X)]...) for _ in 1:s]
            
            # Compute stochastic gradients
            stochastic_grad_U_λ!(GU_λ_stochastic, M, X, loss_func, false, γ, B, "uniform")

            # Concatenate and vectorize, save
            push!(stochastic_grads_vec, vcat(vec.(GU_λ_stochastic)...))

        end

        # Compute empirical bias
        mean_stochastic_grad_vec = (1 / N) * reduce(+, stochastic_grads_vec)
        empirical_bias = norm(mean_stochastic_grad_vec - batch_grad_vec)

        @test isless(empirical_bias / batch_grad_norm, 1e-2)

        end

end

@testitem "stochastic-gradients-bias-uniform-symmetric" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.Adam,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        GCPLosses.grad_U_λ!,
        GCPLosses.stochastic_grad_U_λ!,
        SymCPD
    using LinearAlgebra: norm

    @testset "γ=$γ" for γ in [0, 0.1]

        r = 3
        sz = 5      # 125 total entries
        s = 50      # Sample size
        N = 50000   # Number of stochastic realizations

        # Form tensor
        U_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
            X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
        end

        U_init = randn(sz, r)
        loss_func = LeastSquares()
        M = SymCPD(ones(r), (U_init,), (1,1,1))

        # Allocate for results
        GU_λ_batch = (ntuple(i -> similar(M.U[i]), length(M.U))..., similar(M.λ))
        GU_λ_stochastic = (ntuple(i -> similar(M.U[i]), length(M.U))..., similar(M.λ))
        stochastic_grads_vec = []

        # Compute batch gradient, concatenate and vectorize
        grad_U_λ!(GU_λ_batch, M, X, loss_func, false, γ)
        batch_grad_vec = vcat(vec.(GU_λ_batch)...)
        batch_grad_norm = norm(batch_grad_vec)

        # n stochastic realizations
        for _ in 1:N

            # Sample elements
            B = [CartesianIndex([rand(1:I) for I in size(X)]...) for _ in 1:s]
            
            # Compute stochastic gradients
            stochastic_grad_U_λ!(GU_λ_stochastic, M, X, loss_func, false, γ, B, "uniform")

            # Concatenate and vectorize, save
            push!(stochastic_grads_vec, vcat(vec.(GU_λ_stochastic)...))

        end

        # Compute empirical bias
        mean_stochastic_grad_vec = (1 / N) * reduce(+, stochastic_grads_vec)
        empirical_bias = norm(mean_stochastic_grad_vec - batch_grad_vec)

        @test isless(empirical_bias / batch_grad_norm, 1e-2)

        end

end