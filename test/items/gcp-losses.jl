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
    using ForwardDiff
    using .GCPLosses: value, deriv, domain, AbstractLoss

    # Test that methods are defined
    @testset "type=$type" for type in subtypes(AbstractLoss)
        @test hasmethod(value, Tuple{type,Real,Real})
        @test hasmethod(deriv, Tuple{type,Real,Real})
        @test hasmethod(domain, Tuple{type})
    end

    # Test derivatives with autodiff
    @testset "$loss" for (loss, (xvals, mvals)) in [
        GCPLosses.LeastSquares() => (-2:0.5:2, -2:0.5:2),
        GCPLosses.NonnegativeLeastSquares() => (0:0.5:2, 0:0.5:2),
        GCPLosses.Poisson() => (0:0.5:3, 0:0.5:3),
        GCPLosses.PoissonLog() => (-2:0.5:2, -2:0.5:2),
        GCPLosses.Gamma() => (0:0.5:2, 0.0:0.5:2),
        GCPLosses.Rayleigh() => (0:0.5:2, 0:0.5:2),
        GCPLosses.BernoulliOdds() => (0:0.5:2, 0:0.5:2),
        GCPLosses.BernoulliLogit() => (-2:0.5:2, -2:0.5:2),
        GCPLosses.NegativeBinomialOdds(1) => (0:0.5:2, 0:0.5:2),
        GCPLosses.Huber(1) => (-2:0.5:2, -2:0.5:2),
        GCPLosses.BetaDivergence(0) => (0:0.5:3, 0.1:0.5:3),
        GCPLosses.BetaDivergence(0.5) => (0:0.5:3, 0.1:0.5:3),
        GCPLosses.BetaDivergence(1) => (0:0.5:3, 0.1:0.5:3),
    ]
        for x in xvals, m in mvals
            ad_ref = ForwardDiff.derivative(m -> GCPLosses.value(loss, x, m), m)
            @test GCPLosses.deriv(loss, x, m) ≈ ad_ref
        end
    end
end
