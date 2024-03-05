## Algorithm: LBFGSB

"""
    ALS

**A**lternating **L**east **S**quares.
Workhorse algorithm for `LeastSquaresLoss` with no constraints.

Algorithm parameters:

- `maxiters::Int` : max number of iterations (default: `200`)
"""
Base.@kwdef struct ALS <: AbstractAlgorithm
    maxiters::Int = 200
end
