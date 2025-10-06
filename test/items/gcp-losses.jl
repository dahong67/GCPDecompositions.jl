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
    @testset "loss_func=$loss_type" for loss_type in [
        GCPLosses.BernoulliLogit,
        GCPLosses.BernoulliOdds,
        GCPLosses.BetaDivergence,
        GCPLosses.Gamma,
        GCPLosses.Huber,
        GCPLosses.LeastSquares,
        GCPLosses.NegativeBinomialOdds,
        GCPLosses.NonnegativeLeastSquares,
        GCPLosses.Poisson,
        GCPLosses.PoissonLog,
        GCPLosses.Rayleigh,
    ]
        # Setup
        sz = 10
        r = 2
        X = rand(sz, sz, sz)
        M = CPD(ones(r), (rand(sz, r), rand(sz, r), rand(sz, r)))
        # Losses with no default parameters
        loss_params = Dict(
            GCPLosses.BetaDivergence => (0.5,),
            GCPLosses.Huber => (0.1,),
            GCPLosses.NegativeBinomialOdds => (1,),
        )
        loss_func =
            haskey(loss_params, loss_type) ? loss_type(loss_params[loss_type]...) :
            loss_type()

        # ForwardDiff requires vectorized objective function
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
        computed_grads = [computed_grad[i] for i in 1:3]

        Uλ_vec = vcat(vec(M.U[1]), vec(M.U[2]), vec(M.U[3]), M.λ)
        auto_grad = ForwardDiff.gradient(objective, Uλ_vec)
        auto_grads = [reshape(auto_grad[i*sz*r+1:(i+1)*sz*r], (sz, r)) for i in 0:2]

        for (cg, ag) in zip(computed_grads, auto_grads)
            @test isapprox(cg, ag, rtol = 1e-6)
        end
    end
end