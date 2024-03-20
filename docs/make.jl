using Documenter, GCPDecompositions
using InteractiveUtils

makedocs(;
    modules = [GCPDecompositions],
    sitename = "GCPDecompositions.jl",
    pages = [
        "Home" => "index.md",
        "Quick start guide" => "quickstart.md",
        "Manual" => [
            "Overview" => "man/main.md",
            "Loss functions" => "man/losses.md",
            "Constraints" => "man/constraints.md",
            "Algorithms" => "man/algorithms.md",
        ],
        "Demos" => [
            "Social network data" => "demos/uci-social-network.md",
            "Mouse neuron data" => "demos/mouse-neuron.md",
            "India rainfall data" => "demos/india-rainfall.md",
            "Gas sensor data" => "demos/chemo-sensing.md",
            "Chicago crime data" => "demos/chicago-crime.md",
        ],
        "Developer Docs" =>
            ["Tensor Kernels" => "dev/kernels.md", "Private functions" => "dev/private.md"],
    ],
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/GCPDecompositions.jl",
    ),
)

deploydocs(; repo = "github.com/dahong67/GCPDecompositions.jl.git")
