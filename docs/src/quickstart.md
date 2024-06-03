# Quick start guide

Let's install GCPDecompositions
and run our first **G**eneralized **CP** Tensor **Decomposition**!

## Step 1: Install Julia

Go to [https://julialang.org/downloads](https://julialang.org/downloads)
and install the current stable release.

To check your installation,
open up Julia and try a simple calculation like `1+1`:
```@repl
1 + 1
```

More info: [https://docs.julialang.org/en/v1/manual/getting-started/](https://docs.julialang.org/en/v1/manual/getting-started/)

## Step 2: Install GCPDecompositions

GCPDecompositions can be installed and updated using
Julia's excellent builtin package manager.

```@setup install
import Pkg; Pkg.activate(; temp=true)
```

```@repl install
import Pkg; Pkg.add("GCPDecompositions")
```

This downloads, installs, and precompiles GCPDecompositions
(and all its dependencies).
Don't worry if it takes a few minutes to complete.

We can see that it's installed by checking the Pkg status:

```@repl install
Pkg.status()
```

!!! tip "Tip: Interactive package management with the Pkg REPL mode"

    Here we used the [functional API](https://pkgdocs.julialang.org/v1/api/)
    for the builtin package manager.
    Pkg also has a very nice interactive interface (called the Pkg REPL)
    that is built right into the Julia REPL!

    Learn more here: [https://pkgdocs.julialang.org/v1/getting-started/](https://pkgdocs.julialang.org/v1/getting-started/)

!!! tip "Tip: Pkg environments"

    The package manager has excellent support
    for creating separate installation environments.
    We strongly recommend using environments to create
    isolated and reproducible setups.

    Learn more here: [https://pkgdocs.julialang.org/v1/environments/](https://pkgdocs.julialang.org/v1/environments/)

## Step 3: Run GCPDecompositions

Let's create a simple three-way (a.k.a. order-three)
data tensor that is approximately rank one!

```@repl quickstart
dims = (10, 20, 30)
X = ones(dims) + randn(dims);    # semicolon suppresses the output
```

Mathematically,
this data tensor can be written as follows:

```math
X
=
\underbrace{
    \mathbf{1}_{10} \circ \mathbf{1}_{20} \circ \mathbf{1}_{30}
}_{\text{rank-one ``signal''}}
+
N
\in
\mathbb{R}^{10 \times 20 \times 30}
,
```

where
``\mathbf{1}_{n} \in \mathbb{R}^n`` denotes a vector of ``n`` ones,
``\circ`` denotes the outer product,
and
``N_{ijk} \overset{iid}{\sim} \mathcal{N}(0,1)``.

Now, to get a rank-one GCP decomposition simply load the package
and run `gcp`.

```@repl quickstart
using GCPDecompositions
M = gcp(X, 1)
```

This returns a `CPD` (short for **CP** **D**ecomposition)
with weights ``\lambda`` and factor matrices ``U_1``, ``U_2``, and ``U_3``.
Mathematically, this is the following decomposition
(read [Overview](@ref) to learn more):

```math
M
=
\lambda[1]
\cdot
U_1[:,1] \circ U_2[:,1] \circ U_3[:,1]
\in
\mathbb{R}^{10 \times 20 \times 30}
.
```

We can extract each of these as follows:

```@repl quickstart
M.λ    # to write `λ`, type `\lambda` then hit tab
M.U[1]
M.U[2]
M.U[3]
```

The power of **Generalized** CP Decomposition
is that we can fit CP decompositions to data
using different losses (i.e., different notions of fit).
By default, `gcp` uses the (conventional) least-squares loss,
but we can easily try another!
For example,
to try non-negative least-squares simply run

```@repl quickstart
M_nonneg = gcp(X, 1; loss = GCPLosses.NonnegativeLeastSquaresLoss())
```

!!! tip "Congratulations!"

    Congratulations!
    You have successfully installed GCPDecompositions
    and run some Generalized CP decompositions!

## Next steps

Ready to learn more?

- If you are new to tensor decompositions (or to GCP decompositions in particular), check out the [Overview](@ref) page in the manual. Also check out some of the demos!
- To learn about all the different loss functions you can use or about how to add your own, check out the [Loss functions](@ref) page in the manual.
- To learn about different constraints you can add, check out the [Constraints](@ref) page in the manual.
- To learn about different algorithms you can choose, check out the [Algorithms](@ref) page in the manual.

Want to understand the internals and possibly contribute?
Check out the developer docs.
