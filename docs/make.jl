using GCPDecompositions
using Documenter

DocMeta.setdocmeta!(GCPDecompositions, :DocTestSetup, :(using GCPDecompositions); recursive=true)

makedocs(;
    modules=[GCPDecompositions],
    authors="David Hong <dahong67@wharton.upenn.edu> and contributors",
    repo="https://github.com/dahong67/GCPDecompositions.jl/blob/{commit}{path}#{line}",
    sitename="GCPDecompositions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dahong67.github.io/GCPDecompositions.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/dahong67/GCPDecompositions.jl",
    devbranch="master",
)
