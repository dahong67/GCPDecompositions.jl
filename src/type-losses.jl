## Loss function types

# Abstract type

"""
    AbstractLoss

Abstract type for GCP loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Concrete types `ConcreteLoss <: AbstractLoss` should implement:

  - `value(loss::ConcreteLoss, x, m)` that computes the value of the loss function ``f(x,m)``
  - `deriv(loss::ConcreteLoss, x, m)` that computes the value of the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
  - `domain(loss::ConcreteLoss)` that returns an `Interval` from IntervalSets.jl defining the domain for ``m``
"""
abstract type AbstractLoss end

# Concrete types

"""
    LeastSquaresLoss()

Loss corresponding to conventional CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct LeastSquaresLoss <: AbstractLoss end
value(::LeastSquaresLoss, x, m) = (x - m)^2
deriv(::LeastSquaresLoss, x, m) = 2 * (m - x)
domain(::LeastSquaresLoss) = Interval(-Inf, +Inf)

"""
    PoissonLoss(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Poisson data `X`
with rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\lambda_i``
  - **Loss function:** ``f(x,m) = m - x \\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct PoissonLoss{T<:Real} <: AbstractLoss
    eps::T
    PoissonLoss{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Poisson loss requires nonnegative `eps`."))
end
PoissonLoss(eps::T = 1e-10) where {T<:Real} = PoissonLoss{T}(eps)
value(loss::PoissonLoss, x, m) = m - x * log(m + loss.eps)
deriv(loss::PoissonLoss, x, m) = one(m) - x / (m + loss.eps)
domain(::PoissonLoss) = Interval(0.0, +Inf)
