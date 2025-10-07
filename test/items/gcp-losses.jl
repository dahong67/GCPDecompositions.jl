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
        GCPLosses.Gamma,
        GCPLosses.Huber,
        GCPLosses.LeastSquares,
        GCPLosses.NegativeBinomialOdds,
        GCPLosses.NonnegativeLeastSquares,
        GCPLosses.Poisson,
        GCPLosses.PoissonLog,
        GCPLosses.Rayleigh,
    ]

        # Losses with no default parameters
        loss_params = Dict(GCPLosses.Huber => (1,), GCPLosses.NegativeBinomialOdds => (1,))
        # Define range of values of (x,m) to test
        test_vals = Dict(
            GCPLosses.BetaDivergence => (0.5:0.5:2, 0.5:0.5:2),
            GCPLosses.BernoulliLogit => (-2:0.5:2, -2:0.5:2),
            GCPLosses.BernoulliOdds => (0:0.5:2, 0:0.5:2),
            GCPLosses.Gamma => (0:0.5:2, 0.0:0.5:2),
            GCPLosses.Huber => (-2:0.5:2, -2:0.5:2),
            GCPLosses.LeastSquares => (-2:0.5:2, -2:0.5:2),
            GCPLosses.NegativeBinomialOdds => (0:0.5:2, 0:0.5:2),
            GCPLosses.NonnegativeLeastSquares => (0:0.5:2, 0:0.5:2),
            GCPLosses.Poisson => (0:0.5:3, 0:0.5:3),
            GCPLosses.PoissonLog => (-2:0.5:2, -2:0.5:2),
            GCPLosses.Rayleigh => (0:0.5:2, 0:0.5:2),
        )
        loss_func =
            haskey(loss_params, loss_type) ? loss_type(loss_params[loss_type]...) :
            loss_type()

        for x in test_vals[loss_type][1]
            for m in test_vals[loss_type][2]
                auto_diff_func(m) = GCPLosses.value(loss_func, x, m)
                computed_diff = GCPLosses.deriv(loss_func, x, m)
                auto_diff = ForwardDiff.derivative(auto_diff_func, m)
                @test isapprox(computed_diff, auto_diff, rtol = 1e-6)
            end
        end
    end

    @testset "loss_func=GCPDecompositions.GCPLosses.BetaDivergence, beta=$β" for β in
                                                                                 [0, 0.5, 1]
        loss_func = GCPLosses.BetaDivergence(β)
        test_vals = (0:0.5:3, 0.1:0.5:3)
        for x in test_vals[1]
            for m in test_vals[2]
                auto_diff_func(m) = GCPLosses.value(loss_func, x, m)
                computed_diff = GCPLosses.deriv(loss_func, x, m)
                auto_diff = ForwardDiff.derivative(auto_diff_func, m)
                @test isapprox(computed_diff, auto_diff, rtol = 1e-6)
            end
        end
    end
end
