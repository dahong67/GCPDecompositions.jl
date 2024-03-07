using Documenter, GCPDecompositions

makedocs(;
    modules = [GCPDecompositions],
    sitename = "GCPDecompositions.jl",
    pages = ["Home" => "index.md", "API Reference" => "api.md"],
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/GCPDecompositions.jl",
    ),
)

deploydocs(; repo = "github.com/dahong67/GCPDecompositions.jl.git")
