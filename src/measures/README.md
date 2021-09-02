## Adding new measures

A measure is ordinarily called on data directly, as in

```julia
m = LPLoss(p=3)
m(rand(3), rand(3)) 
```

To call a measure without performing dimension or pool checks, one
uses `MLJBase.call` instead:

```julia
MLJBase.call(m, rand(3), rand(3))
```

However, measures defined in this package ordinarily implement
functionality using the lower level `single` or `multi` methods,
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
float) for a single example. Behaviour on `missing` values is handled
by fallbacks:

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
Label = Union{CategoricalValue,Number,AbstractString,Symbol,AbstractChar}
```

If `single` is *not* implemented, then `call(measure, yhat, y)`,
whose fallback calls `single`, must be implemented.  In this case `y`
and `yhat` are arrays of matching size and the method should return
an array of that size *without performing size or pool checks*. If
`supports_weights(measure)` or `supports_class_weights(measure)` is
be declared true, then `call(measure, yhat, y, w)` must also be
implemented.

### Aggregated measures implement `multi`

To implement an `Aggregated` measure, it suffices to implement the
`multi(measure::MyMeasure, yhat, y)`, which retuns an aggregated
(scalar) value which should:

- perform no dimension or pool checks, but

- can safely assume all argument elements are valid (non-missing and
  non-NaN).

If `supports_weights(measure)` or `supports_class_weights(measure)` is
to be declared true, then `multi(measure, yhat, y, w)` must also be
implemented.

There is also a `call` method for `Unaggregated` measures, which
accepts invalid data. Like it's `Unaggregated` counterpart, it skips
dimension and pool checks. However, it is not expected the implementer of a
new `Aggregated` measure should need to overload `call` whose
fallback calls `multi`.
