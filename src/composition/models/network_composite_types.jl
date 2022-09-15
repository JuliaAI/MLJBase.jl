# For example, we want to define

# abstract type ProbabilisticNetwork <: Probabilistic end

# but also want this for all the abstract `Model` subtypes:

const NETWORK_COMPOSITE_TYPES = Symbol[]
const network_composite_types = Any[]

for T in MLJModelInterface.ABSTRACT_MODEL_SUBTYPES
    network_composite_type_name = string(T, "NetworkComposite") |> Symbol
    @eval(abstract type $network_composite_type_name <: $T end)
    push!(NETWORK_COMPOSITE_TYPES, network_composite_type_name)
    push!(network_composite_types, @eval($network_composite_type_name))
end

const NetworkComposite = Union{network_composite_types...}

MLJModelInterface.is_wrapper(::Type{<:NetworkComposite}) = true
MLJModelInterface.package_name(::Type{<:NetworkComposite}) = "MLJBase"

export MLJType, Model, NetworkComposite
for T in vcat(MMI.ABSTRACT_MODEL_SUBTYPES, NETWORK_COMPOSITE_TYPES)
    @eval(export $T)
end
