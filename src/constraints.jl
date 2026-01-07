## Constraints

# Abstract type and associated functions

"""
    AbstractConstraint

Abstract type for GCP constraints on the factor matrices `U = (U[1],...,U[N])`.

Concrete types `ConcreteConstraint <: AbstractConstraint` should implement:

+ `satisfies(M::CPD, constraint::ConcreteConstraint)`
  that checks if `M` satisfies the constraint
+ `project!(M::CPD, constraint::ConcreteConstraint)`
  that projects `M` onto the constraint set
"""
abstract type AbstractConstraint end

"""
    satisfies(M::CPD, constraint::ConcreteConstraint)

Return whether `M` satisfies the constraint defined by `constraint`.
"""
function satisfies end

"""
    project!(M::CPD, constraint::ConcreteConstraint)

Project `M` in-place onto the constraint set defined by `constraint`.
"""
function project! end

# Built-in constraints

"""
    LowerBoundConstraint(value::Real)

Lower-bound constraint on the entries of the factor matrices
`U = (U[1],...,U[N])`, i.e., `U[i][j,k] >= value`.
"""
struct LowerBoundConstraint{T} <: AbstractConstraint
    value::T
end
satisfies(M::CPD, constraint::LowerBoundConstraint) =
    all(all(>=(constraint.value), Uk) for Uk in M.U)
function project!(M::CPD, constraint::LowerBoundConstraint)
    for Uk in M.U
        Uk .= max.(constraint.value, Uk)
    end
    return M
end
