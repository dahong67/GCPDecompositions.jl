# TensorLy (Python)

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

## Translation table

| Tensor Toolbox | This package |
|:-------------- |:------------ |
| `cp_normalize(cp_tensor)` | `normalizecomps(cp_tensor)` |

## Noteworthy differences

### `normalizecomps` v.s. `cp_normalize` in TensorLy

- `normalizecomps` supports normalizing any subset of the weight vector and factor matrices
  with respect to any $\ell_p$ norm
  and distributing the excess weight into any subset of the weight vector and factor matrices.
  `cp_normalize` in TensorLy only supports normalizing all the factor matrices
  with respect to the $\ell_2$ norm
  and distributing the excess weight into the weight vector.
- `normalizecomps!` normalizes the `CPD` in-place wherease `normalizecomps` makes a new copy.
  `NORMALIZE` in Tensor Toolbox is in-place only.

Ref: [`http://tensorly.org/stable/modules/generated/tensorly.cp_tensor.cp_normalize.html`](http://tensorly.org/stable/modules/generated/tensorly.cp_tensor.cp_normalize.html)
