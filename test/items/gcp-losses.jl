## Loss types

@testitem "loss constructors" begin
    # LeastSquaresLoss
    @test GCPLosses.LeastSquaresLoss() isa GCPLosses.LeastSquaresLoss

    # PoissonLoss
    @test GCPLosses.PoissonLoss() isa GCPLosses.PoissonLoss{Float64}
    @test GCPLosses.PoissonLoss(1.0f-5) isa GCPLosses.PoissonLoss{Float32}
    @test_throws DomainError GCPLosses.PoissonLoss(-0.1)
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
