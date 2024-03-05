## Algorithm: LBFGSB

"""
    LBFGSB

**L**imited-memory **BFGS** with **B**ox constraints.

Brief description of algorithm parameters:

  - `m::Int`         : max number of variable metric corrections (default: `10`)
  - `factr::Float64` : function tolerance in units of machine epsilon (default: `1e7`)
  - `pgtol::Float64` : (projected) gradient tolerance (default: `1e-5`)
  - `maxfun::Int`    : max number of function evaluations (default: `15000`)
  - `maxiter::Int`   : max number of iterations (default: `15000`)
  - `iprint::Int`    : verbosity (default: `-1`)
      + `iprint < 0` means no output
      + `iprint = 0` prints only one line at the last iteration
      + `0 < iprint < 99` prints `f` and `|proj g|` every `iprint` iterations
      + `iprint = 99` prints details of every iteration except n-vectors
      + `iprint = 100` also prints the changes of active set and final `x`
      + `iprint > 100` prints details of every iteration including `x` and `g`

See documentation of [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl) for more details.
"""
Base.@kwdef struct LBFGSB <: AbstractAlgorithm
    m::Int         = 10
    factr::Float64 = 1e7
    pgtol::Float64 = 1e-5
    maxfun::Int    = 15000
    maxiter::Int   = 15000
    iprint::Int    = -1
end
