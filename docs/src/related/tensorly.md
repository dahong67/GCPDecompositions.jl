# TensorLy (Python)

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

## Translation table

| Tensor Toolbox | This package |
|:-------------- |:------------ |
| `cp_normalize(cp_tensor)` | `normalizecomps(cp_tensor)` |

## Noteworthy differences

### `normalizecomps` v.s. `cp_normalize` in TensorLy

- `normalizecomps` supports
  - normalizing any subset of the weight vector and factor matrices,
  - with respect to any $\ell_p$ norm, and
  - distributing the excess weight into any subset of the weight vector and factor matrices.
- `normalizecomps` comes in two varieties:
  - `normalizecomps!` normalizes the `CPD` in-place.
  - `normalizecomps` makes a new copy.

Ref: [`http://tensorly.org/stable/modules/generated/tensorly.cp_tensor.cp_normalize.html`](http://tensorly.org/stable/modules/generated/tensorly.cp_tensor.cp_normalize.html)
