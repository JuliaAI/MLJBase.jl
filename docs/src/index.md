# MLJBase.jl

[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) (Machine
Learning in Julia) is a toolbox written in Julia providing a common
interface and meta-algorithms for selecting, tuning, evaluating,
composing and comparing machine learning models written in Julia and
other languages.

MLJ is released under the MIT licensed and sponsored by the [Alan
Turing Institute](https://www.turing.ac.uk/).

This repository, MLJBase, provides core functionality for MLJ,
including:

- completing the functionality for methods defined "minimally" in
  MLJ's light-weight model interface
  [MLJModelInterface](https://github.com/alan-turing-institute/MLJModelInterface.jl)

- definition of **machines** and their associated methods, such as
  `fit!` and `predict`/`transform`

- MLJ's **model composition** interface, including **learning
  networks** and **pipelines**

- basic utilities for **manipulating data**

- an extension to
  [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
  called `UnivariateFinite` for randomly sampling *labeled*
  categorical data

- a [small
  interface](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Custom-resampling-strategies-1)
  for **resampling strategies** and implementations, including `CV()`,
  `StratifiedCV` and `Holdout`

- methods for **performance evaluation**, based on those resampling strategies

- **one-dimensional hyperparameter range types**, constructors and
  associated methods, for use with
  [MLJTuning](https://github.com/alan-turing-institute/MLJTuning.jl)

- a [small
  interface](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/#Traits-and-custom-measures-1)
  for **performance measures** (losses and scores), enabling the
  integration of the
  [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
  library, user-defined measures, as well as many natively
  defined measures.

- integration with [OpenML](https://www.openml.org)


Previously MLJBase provided the model interface for integrating third
party machine learning models into MLJ. That role has now shifted to
the light-weight
[MLJModelInterface](https://github.com/alan-turing-institute/MLJModelInterface.jl)
package.
