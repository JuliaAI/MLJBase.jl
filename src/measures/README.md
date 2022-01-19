## Adding new measures

This document assumes familiarity with the traits provided for
measures. For a summary, query the docstring for
`MLJBase.metadata_measures`.

A measure is ordinarily called on data directly, as in

```julia
ŷ = rand(3) # predictions
y = rand(3) # ground truth observations

m = LPLoss(p=3)

julia> m(ŷ, y)
3-element Vector{Float64}:
 0.07060087052171798
 0.003020044780949528
 0.019067038457889922
```

To call a measure without performing dimension or pool checks, one
uses `MLJBase.call` instead:

```julia
MLJBase.call(m, ŷ, y)
```

A new measure reporting an aggregate measurement, such as
`AreaUnderCurve`, will subtype `Aggregate`, and only needs to
implement `call`. A measure that reports a measurement for each
observation , such as `LPLoss`, subtypes `Unaggregated` and only needs
to implement an evaluation method for single observations called
`single`.

Recall also that if a measure reports each observation, it does so
even in the case that weights are additionally specified:

```julia
w = rand(3) # per-observation weights

julia> m(ŷ, y, rand(3))
3-element Vector{Float64}:
 0.049333392516241206
 0.0017612002314472718
 0.003157450446692638
 ```

This behaviour differs from other places where weights can only be
specified as part of an aggregation of multi-observation measurements.


### Unaggregated measures implement `single`

To implement an `Unaggregated` measure, it suffices to implement
`single(measure, η̂, η)`, which should return a measurement (e.g., a
float) for a single example `(η̂, η)` (e.g., a pair of
floats). Behavior on `missing` values is handled by fallbacks:

```julia
single(::Unaggregated, η̂::Missing, η)          = missing
single(::Unaggregated, η̂,          η::Missing) = missing
```

Be sure to annotate the type of `η̂` and `η` in the implementation
of `single` to avoid a type ambiguity with these fallbacks. For example

```julia
single(::MyUnaggregatedMeasure, η̂::Real, η::Real) = ...
```
or

```
single(::MyAggregatedMeasure, η̂::UnivariateFinite, η::Label) = ...
```

Here `Label` is just a large union type defined in MLJBase excluding
`missing` as an instance:

```julia
const Label = Union{CategoricalValue,Number,AbstractString,Symbol,AbstractChar}
```

If only `single` is implemented, then the measure will automatically
support per-observation weights and, where that makes sense, per-class
weights. However, `supports_class_weights` may need to be overloaded,
as this defaults to `false`.

#### Special cases

If `single` is *not* implemented, then `call(measure, ŷ, y)`, and
optionally `call(measure, ŷ, y, w)`, must be implemented (the
fallbacks call `single`).  In this case `y` and `ŷ` are arrays of
matching size and the method should return an array of that size
*without performing size or pool checks*. The method should handle
`missing` and `NaN` values if possible, which should be propagated to
relevant elements of the returned array.

The `supports_weights` trait, which defaults to `true`, will need to
be overloaded to return `false` if neither `single(::MyMeasure,
args...)` nor `call(::MyMeasure, ŷ, y, w::AbstractArray)` are
overloaded.

### Aggregated measures implement `call`

To implement an `Aggregated` measure, implement
`call(measure::MyMeasure, ŷ, y)`. Optionally implement 
`call(measure::MyMeasure, ŷ, y, w)`.


### Trait declarations 

Measure traits can be set using the `metadata_measure`
function (query the doc-string) or individually, as in 

```julia
supports_weights(::Type{<:MyMeasure}) = false
```

Defaults are shown below

trait                    | allowed values               | default 
-------------------------|------------------------------|--------------
`target_scitype`         | some scientific type         | `Unknown`
`human_name`             | any `String`                 | string version of type name
`instances`              | any `Vector{String}`         | empty
`prediction_type`        | `:deterministic`, `:probabilistic`, `:interval` `:unknown` | `:unknown`
`orientation`            | `:score`, `:loss`, `:unknown`| `:unknown`
`aggregation`            | `Mean()`, `Sum()`, `RootMeanSqaure()` | `Mean()`
`supports_weights`       | `true` or `false`            | `true`
`supports_class_weights` | `true` or `false`            | `false`
`docstring`              | any `String`                 | includes `name`, `human_name` and `instances`
`distribution_type`      | any `Distribution` subtype or `Unknown`   | `Unknown`

### Exporting the measure and its aliases

If you create a type alias, as in `const MAE = MeanAbsoluteValue`,
then you must add this alias to the constant
`MEASURE_TYPE_ALIASES`. That is the only step needed, as the the macro
`@export_measures` programmatically exports all measure types and
their instances, and those aliases listed in = MeanAbsoluteValue`,
then you must add this alias to the constant `MEASURE_TYPE_ALIASES`.
