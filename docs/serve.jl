using GCPDecompositions
using LiveServer

# Make list of demos to ignore
DEMO_DIR = joinpath(pkgdir(GCPDecompositions), "docs", "src", "demos")
DEMO_FILES = ["$(splitext(f)[1]).md" for f in readdir(DEMO_DIR) if splitext(f)[2] == ".jl"]

# Serve the docs
servedocs(; launch_browser = true, skip_files = joinpath.(DEMO_DIR, DEMO_FILES))
