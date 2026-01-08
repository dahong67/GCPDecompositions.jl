## Stochastic GCP objective and gradient functions: Abstract types and functions

"""
    AbstractGCPSampler

Abstract type for samplers to use in stochastic evaluation
of the objective and gradients.

Concrete types `ConcreteSampler <: AbstractGCPSampler` should implement

+ `gcp_stoch_objective(rng, M, X, loss, sampler::ConcreteSampler)`
+ `gcp_stoch_grad_U!(rng, GU, M, X, loss, sampler::ConcreteSampler)`
"""
abstract type AbstractGCPSampler end

"""
    gcp_stoch_objective([rng=default_rng()], M::CPD, X::AbstractArray, loss, sampler)

Compute stochastic estimate of the GCP objective function for the
model tensor `M`, data tensor `X`, and loss function `loss`
using the sampler `sampler` with random number generator `rng`.
"""
gcp_stoch_objective(M::CPD, X::AbstractArray, loss, sampler) =
    gcp_stoch_objective(default_rng(), M, X, loss, sampler)

"""
    gcp_stoch_grad_U!([rng=default_rng()], GU, M::CPD, X::AbstractArray, loss, sampler)

Compute stochastic estimate of the GCP gradient with respect to the
factor matrices `U = (U[1],...,U[N])` for the model tensor `M`,
data tensor `X`, and loss function `loss` using the sampler `sampler`
with random number generator `rng`, and store the result in `GU = (GU[1],...,GU[N])`.
"""
gcp_stoch_grad_U!(GU, M::CPD, X::AbstractArray, loss, sampler) =
    gcp_stoch_grad_U!(default_rng(), GU, M::CPD, X::AbstractArray, loss, sampler)

"""
    GCPSampleOnce(X::AbstractArray, sampler::AbstractGCPSampler)

Wrapped sampler that samples entries from `X` using `sampler` only
the first time, then reuses the same indices every time after that.
For use with `gcp_stoch_objective`.

The internal field `cache` stores cached values - the particular choice
of what is stored is an implementation detail defined by each `sampler`.
"""
struct GCPSampleOnce{S<:AbstractGCPSampler,C} <: AbstractGCPSampler
    sampler::S
    cache::C
end
