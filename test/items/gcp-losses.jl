## Loss types

@testitem "loss constructors" begin
    # LeastSquares loss
    @test LeastSquaresLoss() isa LeastSquaresLoss

    # Poisson loss
    @test PoissonLoss() isa PoissonLoss{Float64}
    @test PoissonLoss(1.0f-5) isa PoissonLoss{Float32}
    @test_throws DomainError PoissonLoss(-0.1)
end

@testitem "value/deriv/domain methods" begin
    using InteractiveUtils: subtypes
    using ForwardDiff

    # Test that methods are defined
    @testset "type=$type" for type in subtypes(AbstractLoss)
        @test hasmethod(value, Tuple{type,Real,Real})
        @test hasmethod(deriv, Tuple{type,Real,Real})
        @test hasmethod(domain, Tuple{type})
    end

    # Test derivatives with autodiff
    @testset "$loss" for (loss, (xvals, mvals)) in [
        LeastSquaresLoss() => (-2:0.5:2, -2:0.5:2),
        NonnegativeLeastSquaresLoss() => (0:0.5:2, 0:0.5:2),
        PoissonLoss() => (0:0.5:3, 0:0.5:3),
        PoissonLogLoss() => (-2:0.5:2, -2:0.5:2),
        GammaLoss() => (0:0.5:2, 0.0:0.5:2),
        RayleighLoss() => (0:0.5:2, 0:0.5:2),
        BernoulliOddsLoss() => (0:0.5:2, 0:0.5:2),
        BernoulliLogitLoss() => (-2:0.5:2, -2:0.5:2),
        NegativeBinomialOddsLoss(1) => (0:0.5:2, 0:0.5:2),
        HuberLoss(1) => (-2:0.5:2, -2:0.5:2),
        BetaDivergenceLoss(0) => (0:0.5:3, 0.1:0.5:3),
        BetaDivergenceLoss(0.5) => (0:0.5:3, 0.1:0.5:3),
        BetaDivergenceLoss(1) => (0:0.5:3, 0.1:0.5:3),
    ]
        for x in xvals, m in mvals
            ad_ref = ForwardDiff.derivative(m -> value(loss, x, m), m)
            @test deriv(loss, x, m) ≈ ad_ref
        end
    end
end
