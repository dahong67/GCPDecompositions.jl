## SymCP algorithms

@testitem "gradients-fullsym" begin
    using GCPDecompositions:
        TensorKernels.khatrirao,
        GCPAlgorithms.LBFGSB,
        default_constraints,
        default_init_sym,
        GCPLosses.LeastSquares,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    @testset "N=$N, r=$r, sz=$sz" for N in [4], r in [1, 5], sz in [3, 10]

        # Form tensor
        S = ntuple(_ -> 1, N)
        S_reduced = ntuple(_ -> 1, N-1)
        U_star = randn(sz, r)
        X = Array(SymCPD(ones(r), (U_star,), S))

        # First, check that computed gradients at solution = 0

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()

        M_star = SymCPD(ones(r), (U_star,), S)
        GU_unsimplified = (ones(sz, r), ones(r))
        GU_krp_mult = (ones(sz, r), ones(r))
        GU_ttv = (ones(sz, r), ones(r))
        GU_krp_ttv = (ones(sz, r), ones(r))

        # Form coefficient vectors, mappings to reduced linear indexes, mappings to KRP row,
        # buffers for KRPs and reduced matricization
        multinomial_coefs = GCPLosses.collect_multinomial_coefficients(S, (sz,), Val(N))
        multinomial_coefs_minus1 = GCPLosses.collect_multinomial_coefficients(S_reduced, (sz,), Val(N-1))
        idx_map_mats = (form_reduced_linear_mapping_matrix(M_star, 1, Val(N)),)
        vec_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S .== k))÷factorial(count(S .== k)), unique(S))
        mat_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S_reduced .== k))÷factorial(count(S_reduced .== k)), unique(S_reduced))
        krp_row_maps = GCPLosses.make_krp_row_mappings(M_star, vec_size, Val(S), Val(1))

        kr_buffer_weight = similar(M_star.U[1], vec_size, r)
        kr_buffer_factor = similar(M_star.U[1], mat_size, r)
        Y_mat_buffer = similar(M_star.U[1], sz, mat_size)

        # Form vector of unique values of derivative tensor, and buffer for scaled version of KRP mult
        Y_vec = zeros(vec_size)
        Y_vec_scaled_buffer = zeros(vec_size)
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_star, loss, Val(N), Val(r))

        # Compute gradients with different methods
        GCPLosses.grad_U_λ!(GU_unsimplified, M_star, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_star, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat_buffer, kr_buffer_factor, M_star, idx_map_mats[1], (multinomial_coefs_minus1,), 1)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor, M_star, krp_row_maps, multinomial_coefs, 1)
        
        ref_U = zeros(eltype(M_star.U[1]), sz, r)
        ref_λ = zeros(eltype(M_star.U[1]), r)
        @test isapprox(GU_unsimplified[1], ref_U, atol=1e-10)
        @test isapprox(GU_unsimplified[2], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_mult[1], ref_U, atol=1e-10)
        @test isapprox(GU_krp_mult[2], ref_λ, atol=1e-10)
        @test isapprox(GU_ttv[1], ref_U, atol=1e-10)
        @test isapprox(GU_ttv[2], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_ttv[1], ref_U, atol=1e-10)


        # Next, check gradients at random init compared to autodiff

        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M_init = deepcopy(init)
        M_init.λ .= 2*ones(r)   # Change from default ones to make sure we scale by weights correctly
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_init, loss, Val(N), Val(r))

        function form_fullsym_M(U_λ_vec::Vector{T}) where {T}
            U = reshape(U_λ_vec[1:sz*r], (sz, r))
            λ = U_λ_vec[sz*r+1:end]
            return reshape(khatrirao(ntuple(_ -> U, N)...) * λ, ntuple(_ -> sz, N))
        end
        objective(Uλ_vec) = norm(X - form_fullsym_M(Uλ_vec))^2

        auto_grad = ForwardDiff.gradient(objective, vcat(vec(M_init.U[1]), M_init.λ))
        auto_grad_U = reshape(auto_grad[1:sz*r], size(M_init.U[1]))
        auto_grad_λ = auto_grad[sz*r+1:end]

        GCPLosses.grad_U_λ!(GU_unsimplified, M_init, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_init, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat_buffer, kr_buffer_factor, M_init, idx_map_mats[1], (multinomial_coefs_minus1,), 1)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor, M_init, krp_row_maps, multinomial_coefs, 1)

        @test isapprox(GU_unsimplified[1], auto_grad_U, rtol=1e-6)
        @test isapprox(GU_unsimplified[2], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_mult[1], auto_grad_U, rtol=1e-6)
        @test isapprox(GU_krp_mult[2], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_ttv[1], auto_grad_U, rtol=1e-6)
        @test isapprox(GU_ttv[2], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_ttv[1], auto_grad_U, rtol=1e-6)

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
        GCPLosses.grad_U_λ_symmetric!,
        convertCPD,
        SymCPD
    using LinearAlgebra: norm
    import ForwardDiff

    @testset "S=(1,1,2), r=$r, sz1=$sz1, sz2=$sz2" for r in [1, 2], sz1 in [3, 10], sz2 in [5, 15]

        # Form tensor 
        N = 3
        S = (1, 1, 2)
        S_reduced1 = (1, 2)
        S_reduced2 = (1, 1)
        U1_star = randn(sz1, r)
        U2_star = randn(sz2, r)
        λ_star = ones(r)
        X = Array(SymCPD(ones(r), (U1_star, U2_star), S))

        # First, check that computed gradients at solution = 0

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()

        M_star = SymCPD(ones(r), (U1_star, U2_star), S)
        GU_unsimplified = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_krp_mult = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_ttv = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_krp_ttv = (ones(sz1, r), ones(sz2, r), ones(r))

        # Form coefficient vectors, mappings to reduced linear indexes, mappings to KRP row,
        # buffers for KRPs and reduced matricization
        multinomial_coefs = GCPLosses.collect_multinomial_coefficients(S, (sz1, sz2), Val(N))
        multinomial_coefs_minus_mode1 = (GCPLosses.collect_multinomial_coefficients((1,), (sz2,), Val(1)),
                                        GCPLosses.collect_multinomial_coefficients((1,), (sz1,), Val(1)))
        multinomial_coefs_minus_mode3 = (GCPLosses.collect_multinomial_coefficients(S_reduced2, (sz1,), Val(2)),)
        idx_map_mats = (form_reduced_linear_mapping_matrix(M_star, 1, Val(N)), form_reduced_linear_mapping_matrix(M_star, 2, Val(N)))
        vec_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S .== k))÷factorial(count(S .== k)), unique(S))
        mat1_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S_reduced1 .== k))÷factorial(count(S_reduced1 .== k)), unique(S_reduced1))
        mat2_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S_reduced2 .== k))÷factorial(count(S_reduced2 .== k)), unique(S_reduced2))
        krp_row_maps1 = GCPLosses.make_krp_row_mappings(M_star, vec_size, Val(S), Val(1))
        krp_row_maps2 = GCPLosses.make_krp_row_mappings(M_star, vec_size, Val(S), Val(2))

        kr_buffer_weight = similar(M_star.U[1], vec_size, r)
        kr_buffer_factor1 = similar(M_star.U[1], mat1_size, r)
        kr_buffer_factor2 = similar(M_star.U[2], mat2_size, r)
        Y_mat1_buffer = similar(M_star.U[1], sz1, mat1_size)
        Y_mat2_buffer = similar(M_star.U[2], sz2, mat2_size)

        # Form vector of unique values of derivative tensor
        Y_vec = zeros(vec_size)
        Y_vec_scaled_buffer = zeros(vec_size)
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_star, loss, Val(N), Val(r))

        # Compute gradients with different methods
        GCPLosses.grad_U_λ!(GU_unsimplified, M_star, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_star, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat1_buffer, kr_buffer_factor1, M_star, idx_map_mats[1], multinomial_coefs_minus_mode1, 1)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat2_buffer, kr_buffer_factor2, M_star, idx_map_mats[2], multinomial_coefs_minus_mode3, 2)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs, 1)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs, 2)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor1, M_star, krp_row_maps1, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor2, M_star, krp_row_maps2, multinomial_coefs, 2)

        ref_U1 = zeros(eltype(M_star.U[1]), sz1, r)
        ref_U2 = zeros(eltype(M_star.U[2]), sz2, r)
        ref_λ = zeros(eltype(M_star.U[2]), r)
        @test isapprox(GU_unsimplified[1], ref_U1, atol=1e-10)
        @test isapprox(GU_unsimplified[2], ref_U2, atol=1e-10)
        @test isapprox(GU_unsimplified[3], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_mult[1], ref_U1, atol=1e-10)
        @test isapprox(GU_krp_mult[2], ref_U2, atol=1e-10)
        @test isapprox(GU_krp_mult[3], ref_λ, atol=1e-10)
        @test isapprox(GU_ttv[1], ref_U1, atol=1e-10)
        @test isapprox(GU_ttv[2], ref_U2, atol=1e-10)
        @test isapprox(GU_ttv[3], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_ttv[1], ref_U1, atol=1e-10)
        @test isapprox(GU_krp_ttv[2], ref_U2, atol=1e-10)

        # Next, check gradients at random init compared to autodiff

        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M_init = deepcopy(init)
        M_init.λ .= 2*ones(r)   # Change from default ones to make sure we scale by weights correctly
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_init, loss, Val(N), Val(r))

        function form_partialsym_M(U_λ_vec)
            U1 = reshape(U_λ_vec[1:sz1*r], (sz1, r))
            U2 = reshape(U_λ_vec[sz1*r+1:(sz1+sz2)*r], (sz2, r))
            λ = U_λ_vec[(sz1+sz2)*r+1:end]
            return reshape(khatrirao(U2, U1, U1) * λ, (sz1, sz1, sz2))
        end
        objective(Uλ_vec) = norm(X - form_partialsym_M(Uλ_vec))^2

        auto_grad = ForwardDiff.gradient(objective, vcat(vec(M_init.U[1]), vec(M_init.U[2]), M_init.λ))
        auto_grad_U1 = reshape(auto_grad[1:sz1*r], size(M_star.U[1]))
        auto_grad_U2 = reshape(auto_grad[sz1*r+1:(sz1+sz2)*r], size(M_star.U[2]))
        auto_grad_λ = auto_grad[(sz1+sz2)*r+1:end]

        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_init, loss, Val(N), Val(r));

        # Compute gradients with different methods
        GCPLosses.grad_U_λ!(GU_unsimplified, M_init, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_init, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat1_buffer, kr_buffer_factor1, M_init, idx_map_mats[1], multinomial_coefs_minus_mode1, 1)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat2_buffer, kr_buffer_factor2, M_init, idx_map_mats[2], multinomial_coefs_minus_mode3, 2)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs, 1)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs, 2)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor1, M_init, krp_row_maps1, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor2, M_init, krp_row_maps2, multinomial_coefs, 2)

        @test isapprox(GU_unsimplified[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_unsimplified[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_unsimplified[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_mult[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_krp_mult[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_krp_mult[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_ttv[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_ttv[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_ttv[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_ttv[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_krp_ttv[2], auto_grad_U2, rtol=1e-6)
    end

    @testset "S=(1,1,2,2), r=$r, sz1=$sz1, sz2=$sz2" for r in [1, 2], sz1 in [3, 10], sz2 in [5, 15]

        # Form tensor 
        N = 4
        S = (1, 1, 2, 2)
        S_reduced1 = (1, 2, 2)
        S_reduced2 = (1, 1, 2)
        U1_star = randn(sz1, r)
        U2_star = randn(sz2, r)
        λ_star = ones(r)
        X = Array(SymCPD(ones(r), (U1_star, U2_star), S))

        # First, check that computed gradients at solution = 0

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = LBFGSB()

        M_star = SymCPD(ones(r), (U1_star, U2_star), S)
        GU_unsimplified = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_krp_mult = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_ttv = (ones(sz1, r), ones(sz2, r), ones(r))
        GU_krp_ttv = (ones(sz1, r), ones(sz2, r), ones(r))

        # Form coefficient vectors, mappings to reduced linear indexes, mappings to KRP row,
        # buffers for KRPs and reduced matricization
        multinomial_coefs = GCPLosses.collect_multinomial_coefficients(S, (sz1, sz2), Val(N))
        multinomial_coefs_minus_mode1 = (GCPLosses.collect_multinomial_coefficients((1,1), (sz2,), Val(2)),
                                        GCPLosses.collect_multinomial_coefficients((1,), (sz1,), Val(1)))
        multinomial_coefs_minus_mode3 = (GCPLosses.collect_multinomial_coefficients((1,), (sz2,), Val(1)),
                                        GCPLosses.collect_multinomial_coefficients((1,1), (sz1,), Val(2)))
        idx_map_mats = (form_reduced_linear_mapping_matrix(M_star, 1, Val(N)), form_reduced_linear_mapping_matrix(M_star, 2, Val(N)))
        vec_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S .== k))÷factorial(count(S .== k)), unique(S))
        mat1_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S_reduced1 .== k))÷factorial(count(S_reduced1 .== k)), unique(S_reduced1))
        mat2_size = prod(k -> prod(i -> size(M_star.U[k],1)+i-1, 1:count(S_reduced2 .== k))÷factorial(count(S_reduced2 .== k)), unique(S_reduced2))
        krp_row_maps1 = GCPLosses.make_krp_row_mappings(M_star, vec_size, Val(S), Val(1))
        krp_row_maps2 = GCPLosses.make_krp_row_mappings(M_star, vec_size, Val(S), Val(2))

        kr_buffer_weight = similar(M_star.U[1], vec_size, r)
        kr_buffer_factor1 = similar(M_star.U[1], mat1_size, r)
        kr_buffer_factor2 = similar(M_star.U[2], mat2_size, r)
        Y_mat1_buffer = similar(M_star.U[1], sz1, mat1_size)
        Y_mat2_buffer = similar(M_star.U[2], sz2, mat2_size)

        # Form vector of unique values of derivative tensor
        Y_vec = zeros(vec_size)
        Y_vec_scaled_buffer = zeros(vec_size)
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_star, loss, Val(N), Val(r))

        # Compute gradients with different methods
        GCPLosses.grad_U_λ!(GU_unsimplified, M_star, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_star, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat1_buffer, kr_buffer_factor1, M_star, idx_map_mats[1], multinomial_coefs_minus_mode1, 1)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat2_buffer, kr_buffer_factor2, M_star, idx_map_mats[2], multinomial_coefs_minus_mode3, 2)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs, 1)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_star, multinomial_coefs, 2)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor1, M_star, krp_row_maps1, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor2, M_star, krp_row_maps2, multinomial_coefs, 2)

        ref_U1 = zeros(eltype(M_star.U[1]), sz1, r)
        ref_U2 = zeros(eltype(M_star.U[2]), sz2, r)
        ref_λ = zeros(eltype(M_star.U[2]), r)
        @test isapprox(GU_unsimplified[1], ref_U1, atol=1e-10)
        @test isapprox(GU_unsimplified[2], ref_U2, atol=1e-10)
        @test isapprox(GU_unsimplified[3], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_mult[1], ref_U1, atol=1e-10)
        @test isapprox(GU_krp_mult[2], ref_U2, atol=1e-10)
        @test isapprox(GU_krp_mult[3], ref_λ, atol=1e-10)
        @test isapprox(GU_ttv[1], ref_U1, atol=1e-10)
        @test isapprox(GU_ttv[2], ref_U2, atol=1e-10)
        @test isapprox(GU_ttv[3], ref_λ, atol=1e-10)
        @test isapprox(GU_krp_ttv[1], ref_U1, atol=1e-10)
        @test isapprox(GU_krp_ttv[2], ref_U2, atol=1e-10)

        # Next, check gradients at random init compared to autodiff

        init = default_init_sym(X, r, loss, constraints, algorithm, S)
        M_init = deepcopy(init)
        M_init.λ .= 2*ones(r)   # Change from default ones to make sure we scale by weights correctly
        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_init, loss, Val(N), Val(r))

        function form_partialsym_M(U_λ_vec)
            U1 = reshape(U_λ_vec[1:sz1*r], (sz1, r))
            U2 = reshape(U_λ_vec[sz1*r+1:(sz1+sz2)*r], (sz2, r))
            λ = U_λ_vec[(sz1+sz2)*r+1:end]
            return reshape(khatrirao(U2, U2, U1, U1) * λ, (sz1, sz1, sz2, sz2))
        end
        objective(Uλ_vec) = norm(X - form_partialsym_M(Uλ_vec))^2

        auto_grad = ForwardDiff.gradient(objective, vcat(vec(M_init.U[1]), vec(M_init.U[2]), M_init.λ))
        auto_grad_U1 = reshape(auto_grad[1:sz1*r], size(M_star.U[1]))
        auto_grad_U2 = reshape(auto_grad[sz1*r+1:(sz1+sz2)*r], size(M_star.U[2]))
        auto_grad_λ = auto_grad[(sz1+sz2)*r+1:end]

        GCPLosses.fill_reduced_Y_vec!(Y_vec, X, M_init, loss, Val(N), Val(r))

        # Compute gradients with different methods
        GCPLosses.grad_U_λ!(GU_unsimplified, M_init, X, loss, true, 0)
        GCPLosses.weight_grad_krp_mult!(GU_krp_mult, Y_vec, Y_vec_scaled_buffer, kr_buffer_weight, M_init, multinomial_coefs)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat1_buffer, kr_buffer_factor1, M_init, idx_map_mats[1], multinomial_coefs_minus_mode1, 1)
        GCPLosses.factor_grad_krp_mult!(GU_krp_mult, Y_vec, Y_mat2_buffer, kr_buffer_factor2, M_init, idx_map_mats[2], multinomial_coefs_minus_mode3, 2)
        GCPLosses.weight_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs, 1)
        GCPLosses.factor_grad_ttv!(GU_ttv, Y_vec, M_init, multinomial_coefs, 2)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor1, M_init, krp_row_maps1, multinomial_coefs, 1)
        GCPLosses.factor_grad_krp_ttv!(GU_krp_ttv, Y_vec, kr_buffer_factor2, M_init, krp_row_maps2, multinomial_coefs, 2)

        @test isapprox(GU_unsimplified[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_unsimplified[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_unsimplified[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_mult[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_krp_mult[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_krp_mult[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_ttv[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_ttv[2], auto_grad_U2, rtol=1e-6)
        @test isapprox(GU_ttv[3], auto_grad_λ, rtol=1e-6)
        @test isapprox(GU_krp_ttv[1], auto_grad_U1, rtol=1e-6)
        @test isapprox(GU_krp_ttv[2], auto_grad_U2, rtol=1e-6)
    end
end

# @testitem "gradients-fullsym-reg" begin
#     using GCPDecompositions:
#         TensorKernels.khatrirao,
#         GCPAlgorithms.LBFGSB,
#         default_constraints,
#         default_init_sym,
#         GCPLosses.LeastSquares,
#         GCPLosses.grad_U_λ!,
#         SymCPD
#     using LinearAlgebra: norm
#     import ForwardDiff

#     @testset "r=$r, sz=$sz" for r in [1, 2, 5], sz in [3, 10, 50]

#         # Add regularization
#         γ = 0.1

#         # Form tensor
#         U_star = randn(sz, r)
#         X = zeros(sz, sz, sz)
#         for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
#             X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
#         end

#         loss = LeastSquares()
#         constraints = default_constraints(loss)
#         algorithm = LBFGSB()
#         S = (1, 1, 1)
#         M_star = SymCPD(ones(r), (U_star,), (1, 1, 1))

#         function form_fullsym_M(U_λ_vec::Vector{T}) where {T}
#             U = reshape(U_λ_vec[1:sz*r], (sz, r))
#             λ = U_λ_vec[sz*r+1:end]
#             return reshape(khatrirao(U, U, U) * λ, (sz, sz, sz))
#         end
#         function vec_to_col_norms(U_λ_vec::Vector{T}) where {T}
#             # Get norms columns of factor matrices from vectorized form
#             U = reshape(U_λ_vec[1:sz*r], (sz, r))
#             return sum((norm.(eachcol(U)).^2 - ones(T, r)).^2)
#         end
#         objective(Uλ_vec) = norm(X - form_fullsym_M(Uλ_vec))^2 + γ * vec_to_col_norms(Uλ_vec)
#         auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M_star.U[1]), M_star.λ))
#         auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M_star.U[1]))
#         auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]

#         # Check gradients at random init compared to autodiff
#         init = default_init_sym(X, r, loss, constraints, algorithm, S)
#         M0 = deepcopy(init)
#         GU = (similar(M0.U[1]), similar(M0.λ))

#         computed_grad_solution_U, computed_grad_solution_λ = grad_U_λ!(GU, M0, X, loss, false, γ)
#         computed_grad_solution_simplified_U, computed_grad_solution_simplified_λ = grad_U_λ!(GU, M0, X, loss, true, γ)

#         auto_grad_solution = ForwardDiff.gradient(objective, vcat(vec(M0.U[1]), M0.λ))
#         auto_grad_solution_U = reshape(auto_grad_solution[1:sz*r], size(M0.U[1]))
#         auto_grad_solution_λ = auto_grad_solution[sz*r+1:end]

#         @test isapprox(computed_grad_solution_U, auto_grad_solution_U, rtol=1e-6)
#         @test isapprox(computed_grad_solution_λ, auto_grad_solution_λ, rtol=1e-6)
#         @test isapprox(computed_grad_solution_simplified_U, auto_grad_solution_U, rtol=1e-6)
#         @test isapprox(computed_grad_solution_simplified_λ, auto_grad_solution_λ, rtol=1e-6)

#     end
# end

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

        # Check gradients at random init for stochastic with batch equal to entire tensor and non-stochastic
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

        # Check gradients at random init for stochastic with batch equal to entire tensor and non-stochastic
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
        N = 100000   # Number of stochastic realizations

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

@testitem "stochastic-gradients-stratified" begin
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
    using Random

    @testset "r=$r, sz=10, γ=$γ" for r in [1, 5], γ in [0, 0.1]

        sz = 10
        # Form tensor with 100 zeros and 900 nonzeros
        X = zeros(sz,sz,sz)
        nonzero_idxs = randperm(sz^3)[1:Int(sz^3/10)]
        X[nonzero_idxs] = randn(Int(sz^3/10))

        loss = LeastSquares()
        constraints = default_constraints(loss)
        algorithm = Adam()
        S = (1,2,3)

        # Check gradients at random init for stochastic with batch equal to entire tensor and non-stochastic
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

        stochastic_grad_U_λ!(GU_stochastic, M0, X, loss, false, γ, CartesianIndices(X), "stratified"; p=length(nonzero_idxs), q=length(X)-length(nonzero_idxs))
        stochastic_grad_U1 = GU_stochastic[1]
        stochastic_grad_U2 = GU_stochastic[2]
        stochastic_grad_U3 = GU_stochastic[3]
        stochastic_grad_λ = GU_stochastic[4]
        stochastic_grad_U_λ!(GU_stochastic_simplified, M0, X, loss, true, γ, CartesianIndices(X), "stratified"; p=length(nonzero_idxs), q=length(X)-length(nonzero_idxs))
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

@testitem "stochastic-gradients-unsupported" begin
    using GCPDecompositions:
        default_constraints,
        default_init_sym,
        GCPLosses,
        GCPAlgorithms

    @testset "unsupported sampling strategies" begin
        # Form tensor
        sz = 10
        r = 3
        U_star = randn(sz, r)
        X = zeros(sz, sz, sz)
        for i1 in axes(U_star, 1), i2 in axes(U_star, 1), i3 in axes(U_star, 1)
            X[i1, i2, i3] = sum(U_star[i1, :] .* U_star[i2, :] .* U_star[i3, :])
        end

        loss = GCPLosses.LeastSquares()
        constraints = default_constraints(loss)
        algorithm = GCPAlgorithms.Adam()
        S = (1, 1, 1)
        M0 = default_init_sym(X, r, loss, constraints, algorithm, S)

        GU_stochastic = (similar(U_star), ones(r))
        @test_throws ErrorException GCPLosses.stochastic_grad_U_λ!(GU_stochastic, M0, X, loss, false, 0.1, CartesianIndices(X), "unstratified")
    end
end

@testitem "symgcp-lbfgs" begin
    using Random, IntervalSets
    using Distributions

    @testset "nonsymmetric, size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = SymCPD(ones(r), rand.(sz, r), (1,2,3))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh, _, _, _ = symgcp(X, r, (1,2,3); loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    @testset "unsupported domain, constraints" begin
        @test_throws ErrorException symgcp(
            randn(5,5,5),
            2,
            (1,2,3);
            loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
        )
        @test_throws ErrorException symgcp(
            randn(5,5,5),
            2,
            (1,2,3);
            loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(0, Inf)),
            constraints = (GCPConstraints.LowerBound(1.0),),
        )
    end

    @testset "fully symmetric, size(X)=$sz, rank(X)=$r" for sz in [(15,15,15), (30,30,30)], r in 1:2
        Random.seed!(0)
        M = SymCPD(ones(r), (rand(sz[1],r), ), (1,1,1))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh, _, _, _ = symgcp(X, r, (1,1,1); loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    @testset "partially symmetric, size(X)=$sz, rank(X)=$r" for sz in [(15,15,20), (25,25,30)], r in 1:2
        Random.seed!(0)
        M = SymCPD(ones(r), (rand(sz[1],r), rand(sz[3],r)), (1,1,2))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh, _, _, _ = symgcp(X, r, (1,1,2); loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    @testset "nonsymmetric, nonnegative, size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
       Random.seed!(0)
        M = SymCPD(ones(r), rand.(sz, r), (1,2,3))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh, _, _, _ = symgcp(X, r, (1,2,3); loss = GCPLosses.NonnegativeLeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "symgcp-adam" begin
    using Random
    using SparseArrays

    @testset "nonsymmetric, size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25)], r in 1:2
        Random.seed!(0)
        M = SymCPD(ones(r), rand.(sz, r), (1,2,3))
        X = [M[I] for I in CartesianIndices(size(M))]
        # Run Adam with large batch size
        Mh, _, _, _ = symgcp(X, r, (1,2,3); loss = GCPLosses.LeastSquares(), algorithm = GCPAlgorithms.Adam(s=length(X), τ=100))
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
    @testset "nonsymmetric, stratified, size(X)=$sz, rank(X)=$r" for sz in [(10, 15, 20)], r in 1:2
        Random.seed!(0)
        U = sprand.(sz, r, 0.5)
        M = SymCPD(ones(r), U, (1,2,3))
        X = [M[I] for I in CartesianIndices(size(M))]
        nnz = count((!iszero).(X))
        # Run Adam with large batch size
        Uinit = map(Ui -> Ui .+ randn(size(Ui)) * 0.00001, U)
        Minit = SymCPD(ones(r), Uinit, (1,2,3))
        Mh, _, _, _ = symgcp(X, r, (1,2,3); loss = GCPLosses.LeastSquares(), 
                                algorithm = GCPAlgorithms.Adam(sampling_strategy="stratified", p=nnz, s=length(X)-nnz, τ=1000))
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-4
    end
end

@testitem "symgcp-adam-unsupported" begin
    using IntervalSets
 
    @testset "unsupported domain, constraints" begin
        @test_throws ErrorException symgcp(
            randn(5,5,5),
            2,
            (1,2,3);
            loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
            algorithm = GCPAlgorithms.Adam()
        )
        @test_throws ErrorException symgcp(
            randn(5,5,5),
            2,
            (1,2,3);
            loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(0, Inf)),
            algorithm = GCPAlgorithms.Adam(),
            constraints = (GCPConstraints.LowerBound(1.0),),
        )
    end
    @testset "unsupported sampling strategy" begin
        @test_throws ErrorException symgcp(
            randn(5,5,5),
            2,
            (1,2,3);
            loss = GCPLosses.LeastSquares(),
            algorithm = GCPAlgorithms.Adam(sampling_strategy="semistratified"),
        )
    end
end