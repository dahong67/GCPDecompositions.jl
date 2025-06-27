using GCPDecompositions
using Documenter, InteractiveUtils

# Setup
DocMeta.setdocmeta!(
    GCPDecompositions,
    :DocTestSetup,
    :(using GCPDecompositions);
    recursive = true,
)
ENV["LINES"] = 9

# Render demos
using PlutoStaticHTML
DEMO_DIR = joinpath(pkgdir(GCPDecompositions), "docs", "src", "demos")
DEMO_DICT = build_notebooks(
    BuildOptions(DEMO_DIR; previous_dir = DEMO_DIR, output_format = documenter_output),
    OutputOptions(; append_build_context = true),
)
DEMO_FILES = keys(DEMO_DICT)

# Make docs
makedocs(;
    modules = [GCPDecompositions],
    authors = "David Hong <hong@udel.edu> and contributors",
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
        "Related packages" => [
            "Tensor Toolbox (MATLAB)" => "related/tensortoolbox.md",
            "TensorLy (Python)" => "related/tensorly.md",
        ],
        "Developer Docs" =>
            ["Tensor Kernels" => "dev/kernels.md", "Private functions" => "dev/private.md"],
    ],
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/GCPDecompositions.jl",
        edit_link = "master",
        assets = String[],
        size_threshold = 2^21,
    ),
)

# Deploy docs
deploydocs(; repo = "github.com/dahong67/GCPDecompositions.jl", devbranch = "master")
