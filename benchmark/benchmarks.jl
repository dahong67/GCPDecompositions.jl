using BenchmarkTools

# Benchmark suite modules
const SUITE_MODULES = Dict("gcp" => :BenchmarkGCP, "mttkrp" => :BenchmarkMTTKRP)

# Create top-level suite including only sub-suites
# specified by ENV variable "GCP_BENCHMARK_SUITES"
const SUITE = BenchmarkGroup()
SELECTED_SUITES = split(get(ENV, "GCP_BENCHMARK_SUITES", join(keys(SUITE_MODULES), ' ')))
for suite_name in SELECTED_SUITES
    module_name = SUITE_MODULES[suite_name]
    include(joinpath(@__DIR__, "suites", "$(suite_name).jl"))
    SUITE[suite_name] = eval(module_name).SUITE
end
