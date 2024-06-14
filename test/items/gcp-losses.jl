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
