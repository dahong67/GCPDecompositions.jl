using Documenter, GCPDecompositions
using PlutoStaticHTML
using InteractiveUtils

# Render demos
DEMO_DIR = joinpath(pkgdir(GCPDecompositions), "docs", "src", "demos")
DEMO_DICT = build_notebooks(
    BuildOptions(DEMO_DIR; previous_dir = DEMO_DIR, output_format = documenter_output),
    OutputOptions(; append_build_context = true),
)
DEMO_FILES = keys(DEMO_DICT)

# Make docs
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
            "Overview" => "demos/main.md",
            [
                get(
                    PlutoStaticHTML.Pluto.frontmatter(joinpath(DEMO_DIR, FILE)),
                    "title",
                    FILE,
                ) => joinpath("demos", "$(splitext(FILE)[1]).md") for FILE in DEMO_FILES
            ]...,
        ],
        "Developer Docs" =>
            ["Tensor Kernels" => "dev/kernels.md", "Private functions" => "dev/private.md"],
    ],
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/GCPDecompositions.jl",
    ),
)

# Deploy docs
deploydocs(; repo = "github.com/dahong67/GCPDecompositions.jl.git")
