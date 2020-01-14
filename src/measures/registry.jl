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

"""
    measures()

List all measures as named-tuples keyed on measure traits.

    measures(conditions...)

List all measures satisifying the specified `conditions`. A *condition*
is any `Bool`-valued function on the named-tuples.


### Example

Find all classification measures supporting sample weights:

    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
                  m.supports_weights)

"""
function measures(conditions...)
    all_measures = map(info, MEASURE_TYPES)
    return filter(all_measures) do measure
        all(c(measure) for c in conditions)
    end
end

measures() = measures(x->true)
