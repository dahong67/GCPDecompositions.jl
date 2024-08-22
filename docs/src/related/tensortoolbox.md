# Tensor Toolbox (MATLAB)

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

## Translation table

| Tensor Toolbox | This package |
|:-------------- |:------------ |
| `NORMALIZE(X)` | `normalizecomps(X)` |
| `NORMALIZE(X,N)` | `normalizecomps(X; distribute_to=N)` |
| `NORMALIZE(X,0)` | `normalizecomps(X; distribute_to=1:ndims(X))` |
| `NORMALIZE(X,[])` | `normalizecomps(X)` |
| `NORMALIZE(X,'sort')` | TODO |
| `NORMALIZE(X,N,1)` | `normalizecomps(X, 1; distribute_to=N)` |
| `NORMALIZE(X,0,1)` | `normalizecomps(X, 1; distribute_to=1:ndims(X))` |
| `NORMALIZE(X,[],1)` | `normalizecomps(X, 1)` |
| `NORMALIZE(X,'sort',1)` | TODO |
| `NORMALIZE(X,[],1,I)` | `normalizecomps(X, 1; dims=I)` |
| `NORMALIZE(X,[],2,I)` | `normalizecomps(X, 2; dims=I)` |

## Noteworthy differences

### `normalizecomps` v.s. `NORMALIZE` in Tensor Toolbox

- `normalizecomps` does not fix the signs of the weights to be positive.
- `normalizecomps` supports
  - normalizing any subset of the weight vector and factor matrices,
  - with respect to any $\ell_p$ norm, and
  - distributing the excess weight into any subset of the weight vector and factor matrices.
- `normalizecomps` comes in two varieties:
  - `normalizecomps!` normalizes the `CPD` in-place.
  - `normalizecomps` makes a new copy.

Ref: [`https://gitlab.com/tensors/tensor_toolbox/-/blob/v3.6/@ktensor/normalize.m`](https://gitlab.com/tensors/tensor_toolbox/-/blob/v3.6/@ktensor/normalize.m)
