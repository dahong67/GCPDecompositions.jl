## Serve a live version of the documentation

# Check that Julia version is at least v1.11
# (needed for the `sources` field used in `docs/Project.toml`)
VERSION >= v"1.11" || error("Julia version must be at least v1.11")

# Activate and instantiate the `docs` environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Make list of demos to ignore
DEMO_DIR = joinpath(@__DIR__, "src", "demos")
DEMO_FILES = ["$(splitext(f)[1]).md" for f in readdir(DEMO_DIR) if splitext(f)[2] == ".jl"]

# Load packages and serve the docs
using GCPDecompositions, LiveServer
Base.exit_on_sigint(false)
cd(pkgdir(GCPDecompositions)) do
    servedocs(; skip_files = joinpath.(DEMO_DIR, DEMO_FILES))
end
