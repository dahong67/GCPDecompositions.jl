using Documenter, GCPDecompositions
using InteractiveUtils

makedocs(;
    modules = [GCPDecompositions],
    sitename = "GCPDecompositions.jl",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Main functions and types" => "man/main.md",
            "Loss functions" => "man/losses.md",
            "Constraints" => "man/constraints.md",
            "Algorithms" => "man/algorithms.md",
        ],
        "Developer Docs" =>
            ["Tensor Kernels" => "dev/kernels.md", "Private functions" => "dev/private.md"],
    ],
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/GCPDecompositions.jl",
    ),
)

deploydocs(; repo = "github.com/dahong67/GCPDecompositions.jl.git")
