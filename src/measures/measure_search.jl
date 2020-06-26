# *Important:* If there is a measure trait that depends on the value
# of a type parameter (eg, `target_scitype(Brier_Score{D})` depends on
# the value of `D`) then we need a separate listing below for each
# case. Otherwise, the UnionAll type suffices. So, for example,
# although `FScore{β}` has a type parameter `β`, no trait depends on the
# value of `β`, so we need only an entry for `FScore`


const LOCAL_MEASURE_TYPES = subtypes(MLJBase.Measure)
filter!(M -> M != BrierScore, LOCAL_MEASURE_TYPES)
push!(LOCAL_MEASURE_TYPES, BrierScore{UnivariateFinite})

const LOSSFUNCTIONS_MEASURE_TYPES =
    vcat(subtypes(LossFunctions.MarginLoss),
         subtypes(LossFunctions.DistanceLoss))

const MEASURE_TYPES = vcat(LOCAL_MEASURE_TYPES, LOSSFUNCTIONS_MEASURE_TYPES)

push!(MEASURE_TYPES, Confusion)

const MeasureProxy = NamedTuple{Tuple(MEASURE_TRAITS)}

Base.show(stream::IO, p::MeasureProxy) =
    print(stream, "(name = $(p.name), ...)")

function Base.show(stream::IO, ::MIME"text/plain", p::MeasureProxy)
    printstyled(IOContext(stream, :color=> MLJBase.SHOW_COLOR),
                p.docstring, bold=false, color=:magenta)
    println(stream)
    MLJBase.fancy_nt(stream, p)
end


## QUERYING MEASURES

## For using `matching(y)` or `matching(X, y)` as a filter.

(f::Checker{false,false,yS,missing})(m) where yS =
    yS <: m.target_scitype

(f::Checker{true,false,XS,yS})(m) where {XS,yS} =
    yS <: m.target_scitype

(f::Checker{true,true,XS,yS})(m) where {XS,yS} =
    yS <: m.target_scitype &&
    m.supports_weights


"""
    measures()

List all measures as named-tuples keyed on measure traits.

    measures(filters...)

List all measures `m` for which `filter(m)` is true, for each `filter`
in `filters`.

    measures(matching(y))

List all measures compatible with the target `y`.


### Example

Find all classification measures supporting sample weights:

    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
                  m.supports_weights)

Find all classification measures where the number of classes is three:

    y  = categorical(1:3)
    measures(matching(y))

"""
function measures(conditions...)
    all_measures = map(info, MEASURE_TYPES)
    return filter(all_measures) do measure
        all(c(measure) for c in conditions)
    end
end

measures() = measures(x->true)
