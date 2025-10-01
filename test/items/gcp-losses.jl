## Loss types

@testitem "loss constructors" begin
    # LeastSquares loss
    @test GCPLosses.LeastSquares() isa GCPLosses.LeastSquares

    # Poisson loss
    @test GCPLosses.Poisson() isa GCPLosses.Poisson{Float64}
    @test GCPLosses.Poisson(1.0f-5) isa GCPLosses.Poisson{Float32}
    @test_throws DomainError GCPLosses.Poisson(-0.1)
end

@testitem "value/deriv/domain methods" begin
    using InteractiveUtils: subtypes
    using .GCPLosses: value, deriv, domain, AbstractLoss
    @testset "type=$type" for type in subtypes(AbstractLoss)
        @test hasmethod(value, Tuple{type,Real,Real})
        @test hasmethod(deriv, Tuple{type,Real,Real})
        @test hasmethod(domain, Tuple{type})
    end
end

# Compare manually computed derivatives to autodiff
@testitem "deriv methods correctness" begin
    using InteractiveUtils: subtypes
    using ForwardDiff
    using .GCPLosses: AbstractLoss
    @testset "loss_func=$loss_type" for loss_type in [GCPLosses.BernoulliLogit,
        GCPLosses.BernoulliOdds,
        GCPLosses.BetaDivergence,
        GCPLosses.Gamma,
        GCPLosses.Huber,
        GCPLosses.LeastSquares,
        GCPLosses.NegativeBinomialOdds,
        GCPLosses.NonnegativeLeastSquares,
        GCPLosses.Poisson,
        GCPLosses.PoissonLog,
        GCPLosses.Rayleigh]
        
        sz = 10
        r = 2
        X = rand(sz,sz,sz)
        M = CPD(ones(r), (rand(sz,r), rand(sz,r), rand(sz,r)))
        if loss_type == GCPLosses.BetaDivergence
            loss_func = loss_type(0.5)
        elseif loss_type == GCPLosses.Huber
            loss_func = loss_type(0.1)
        elseif loss_type == GCPLosses.NegativeBinomialOdds
            loss_func = loss_type(1)
        else
            loss_func = loss_type()
        end

        # For autodiff
        function form_M(U_λ_vec::Vector{T}) where {T}
            U1 = reshape(U_λ_vec[1:sz*r], (sz, r))
            U2 = reshape(U_λ_vec[sz*r+1:2*sz*r], (sz, r))
            U3 = reshape(U_λ_vec[2*sz*r+1:3*sz*r], (sz, r))
            λ = U_λ_vec[3*sz*r+1:end]
            return CPD(λ, (U1, U2, U3))
        end
        objective(Uλ_vec) = GCPAlgorithms.gcp_objective(form_M(Uλ_vec), X, loss_func)

        # Check gradients at random init compared to autodiff
        GU = similar.(M.U)
        computed_grad = GCPAlgorithms.gcp_grad_U!(GU, M, X, loss_func)
        computed_grad_U1, computed_grad_U2, computed_grad_U3 = computed_grad[1], computed_grad[2], computed_grad[3]
        
        Uλ_vec = vcat(vec(M.U[1]), vec(M.U[2]), vec(M.U[3]), M.λ)
        auto_grad = ForwardDiff.gradient(objective, Uλ_vec)
        auto_grad_U1 = reshape(auto_grad[1:sz*r], (sz,r))
        auto_grad_U2 = reshape(auto_grad[sz*r+1:2*sz*r], (sz,r))
        auto_grad_U3 = reshape(auto_grad[2*sz*r+1:3*sz*r], (sz,r))
        
        @test isapprox(computed_grad_U1, auto_grad_U1, rtol=1e-6)
        @test isapprox(computed_grad_U2, auto_grad_U2, rtol=1e-6)
        @test isapprox(computed_grad_U3, auto_grad_U3, rtol=1e-6)

    end
end