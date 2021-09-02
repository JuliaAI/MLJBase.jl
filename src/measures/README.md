## Adding new measures

A measure is ordinarily called on data directly, as in

```julia
ŷ = rand(3) # predictions
y = rand(3) # ground truth observations

m = LPLoss(p=3)
m(ŷ, y) 
```

To call a measure without performing dimension or pool checks, one
uses `MLJBase.call` instead:

```julia
MLJBase.call(m, ŷ, y)
```

However, measures defined in this package ordinarily implement
functionality using the lower level `single` or `mutli` methods,
according to whether the measure reports a measurement for every
observation (e.g., `LPLoss`) or only an aggregate (e.g.,
`AreaUnderCurve`). Such measures are subtypes of `Unaggregated` or
`Aggregated` respectively. This avoids the need for the implementer to
worry about:

- handling of invalid data (`missing` or `NaN`)

- dimension checking and class pool checking

- incorporating per-observation or per-class weights, unless the
  measure is `Aggregated`,


### Unaggregated measures implement `single`

To implement an `Unaggregated` measure, it suffices to implement
`single(measure, η̂, η)`, which should return a measurement (e.g., a
float) for a single example `(η̂, η)` (e.g., a pair of
floats). Behaviour on `missing` values is handled by fallbacks:

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
support per-obervation weights and, where that makes sense, per-class
weights. However, if necessary, `supports_class_weights` may need to
be overloaded, as this is `false` by default.

#### Special cases

If `single` is *not* implemented, then `call(measure, ŷ, y)`, and
optionally `call(measure, ŷ, y, w)`, must be implemented (the
fallbacks call `single`).  In this case `y` and `ŷ` are arrays of
matching size and the method should return an array of that size
*without performing size or pool checks*.

The `supports_weights` trait, which defaults to `true`, will need to
be overloaded if neither `single` nor `call(ŷ, y, w::AbstractDict)`
are overloaded.


`supports_class_weights(measure)` is be declared true, then
 must also be implemented.

Note that by default `supports_weights(typeof(measure))` is `true` and
`supports_class_weights(typeof(measure))` is `false` whenever `measure isa
Union{Aggregated,Unaggregated}`. Both are false for direct subtypes of 


### Aggregated measures implement `multi`

To implement an `Aggregated` measure, it suffices to implement the
`multi(measure::MyMeasure, ŷ, y)`, which retuns an aggregated
(scalar) value which should:

- perform no dimension or pool checks, but

- can safely assume all argument elements are valid (non-missing and
  non-NaN).

Implementing `multi(measure, ŷ, y, w)` for `w::Arr{<:Real}` or
`w::AbstractDict` is optional, but keep in mind that the traits
`supports_weights` default to `true` and `supports_class_weights`
defaults to `false`.

There is also a `call` method for `Unaggregated` measures, which
accepts invalid data. Like it's `Unaggregated` counterpart, it skips
dimension and pool checks. However, it is not expected the implementer of a
new `Aggregated` measure should need to overload `call` whose
fallback calls `multi`.
