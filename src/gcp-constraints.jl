## Constraint types

"""
Constraints for Generalized CP Decomposition.
"""
module GCPConstraints

# Abstract type

"""
    AbstractConstraint

Abstract type for GCP constraints on the factor matrices `U = (U[1],...,U[N])`.
"""
abstract type AbstractConstraint end

# Concrete types

"""
    LowerBound(value::Real)

Lower-bound constraint on the entries of the factor matrices
`U = (U[1],...,U[N])`, i.e., `U[i][j,k] >= value`.
"""
struct LowerBound{T} <: AbstractConstraint
    value::T
end

end
