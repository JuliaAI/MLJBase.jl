const LOCAL_MEASURE_TYPES = filter(x->x != SupervisedLoss,
                                   vcat(subtypes(MLJBase.Unaggregated),
                                        subtypes(MLJBase.Aggregated)))

const LOSS_FUNCTIONS_MEASURE_TYPES =
    [eval(:($Loss)) for Loss in LOSS_FUNCTIONS]

const MEASURE_TYPES = vcat(LOCAL_MEASURE_TYPES, LOSS_FUNCTIONS_MEASURE_TYPES)

const MeasureProxy = NamedTuple{Tuple(MEASURE_TRAITS)}

function Base.show(stream::IO, p::MeasureProxy)
    instances = "["*join(p.instances, ", ")*"]"
    print(stream, "(name = $(p.name), instances = $instances, ...)")
end

function Base.show(stream::IO, ::MIME"text/plain", p::MeasureProxy)
    printstyled(IOContext(stream, :color=> MLJBase.SHOW_COLOR),
                p.docstring, bold=false, color=:magenta)
    println(stream)
    MLJBase.fancy_nt(stream, p)
end

"""
    measures()

List all measures as named-tuples keyed on measure traits.

    measures(filters...)

List all measures compatible with the target `y`.

    measures(needle::Union{AbstractString,Regex}

List all measures with `needle` in a measure's `name`, `instances`, or
`docstring`


### Example

Find all classification measures supporting sample weights:

    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
                  m.supports_weights)

Find all measures in the "rms" family:

    measures("rms")

"""
function measures(conditions...)
    all_measures = map(info, MEASURE_TYPES)
    return filter(all_measures) do measure
        all(c(measure) for c in conditions)
    end
end

function measures(needle::Union{AbstractString,Regex})
    f = m -> occursin(needle, m.name) ||
        occursin(needle, m.docstring) ||
        occursin(needle, join(m.instances, " "))
    return MLJBase.measures(f)
end

measures() = measures(x->true)
