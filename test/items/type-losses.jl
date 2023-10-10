## Loss types

@testitem "loss constructors" begin
    # LeastSquaresLoss
    @test LeastSquaresLoss() isa LeastSquaresLoss

    # PoissonLoss
    @test PoissonLoss() isa PoissonLoss{Float64}
    @test PoissonLoss(1.0f-5) isa PoissonLoss{Float32}
    @test_throws DomainError PoissonLoss(-0.1)
end

@testitem "value/deriv/domain methods" begin
    using InteractiveUtils: subtypes
    using GCPDecompositions: value, deriv, domain
    @testset "type=$type" for type in subtypes(AbstractLoss)
        @test hasmethod(value, Tuple{type,Real,Real})
        @test hasmethod(deriv, Tuple{type,Real,Real})
        @test hasmethod(domain, Tuple{type})
    end
end
