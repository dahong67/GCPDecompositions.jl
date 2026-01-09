# ExplicitImports

@testitem "Implicit imports" begin
    using ExplicitImports
    @test check_no_implicit_imports(GCPDecompositions) === nothing
end

@testitem "Stale imports" begin
    using ExplicitImports
    @test check_no_stale_explicit_imports(GCPDecompositions) === nothing
end

@testitem "Non-owner imports" begin
    using ExplicitImports
    @test check_all_explicit_imports_via_owners(GCPDecompositions) === nothing
end

@testitem "Non-public imports" begin
    using ExplicitImports
    @test check_all_explicit_imports_are_public(GCPDecompositions) === nothing
end

@testitem "Non-owner qualified accesses" begin
    using ExplicitImports
    @test check_all_qualified_accesses_via_owners(GCPDecompositions) === nothing
end

@testitem "Non-public qualified accesses" begin
    using ExplicitImports
    ignore =
        (:checkbounds_indices, :derivative, :throw_boundserror, :require_one_based_indexing)
    @test check_all_qualified_accesses_are_public(GCPDecompositions; ignore) === nothing
end

@testitem "Self-qualified accesses" begin
    using ExplicitImports
    @test check_no_self_qualified_accesses(GCPDecompositions) === nothing
end
