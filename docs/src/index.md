```@meta
CurrentModule = GCPDecompositions
```

# GCPDecompositions: Generalized CP Decompositions

Documentation for [GCPDecompositions](https://github.com/dahong67/GCPDecompositions.jl).

> 👋 *This package provides research code and work is ongoing.
> If you are interested in using it in your own research,
> **I'd love to hear from you and collaborate!**
> Feel free to write: [hong@udel.edu](mailto:hong@udel.edu)*

Please cite the following papers for this technique:

> David Hong, Tamara G. Kolda, Jed A. Duersch.
> "Generalized Canonical Polyadic Tensor Decomposition",
> *SIAM Review* 62:133-163, 2020.
> [https://doi.org/10.1137/18M1203626](https://doi.org/10.1137/18M1203626)
> [https://arxiv.org/abs/1808.07452](https://arxiv.org/abs/1808.07452)
>
> Tamara G. Kolda, David Hong.
> "Stochastic Gradients for Large-Scale Tensor Decomposition",
> *SIAM Journal on Mathematics of Data Science* 2:1066-1095, 2020.
> [https://doi.org/10.1137/19M1266265](https://doi.org/10.1137/19M1266265)
> [https://arxiv.org/abs/1906.01687](https://arxiv.org/abs/1906.01687)

In BibTeX form:
```bibtex
@Article{hkd2020gcp,
  title =        "Generalized Canonical Polyadic Tensor Decomposition",
  author =       "David Hong and Tamara G. Kolda and Jed A. Duersch",
  journal =      "{SIAM} Review",
  year =         "2020",
  volume =       "62",
  number =       "1",
  pages =        "133--163",
  DOI =          "10.1137/18M1203626",
}

@Article{kh2020sgf,
  title =        "Stochastic Gradients for Large-Scale Tensor Decomposition",
  author =       "Tamara G. Kolda and David Hong",
  journal =      "{SIAM} Journal on Mathematics of Data Science",
  year =         "2020",
  volume =       "2",
  number =       "4",
  pages =        "1066--1095",
  DOI =          "10.1137/19M1266265",
}
```

## Docstrings

```@index
```

```@autodocs
Modules = [GCPDecompositions]
Filter = t -> t !== ncomponents
```
