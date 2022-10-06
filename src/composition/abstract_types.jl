##  COMPOSITE AND SURRUGOTE MODEL TYPES

# For example, we want to define

# abstract type ProbabilisticComposite <: Probabilistic end
# struct ProbabilisticSurrogate <: Probabilistic end
# Probabilistic() = ProbablisiticSurrogate()

# but also want this for all the abstract `Model` subtypes:

const COMPOSITE_TYPES = Symbol[]
const SURROGATE_TYPES = Symbol[]
const composite_types = Any[]
const surrogate_types = Any[]

for T in MLJModelInterface.ABSTRACT_MODEL_SUBTYPES
    composite_type_name = string(T, "Composite") |> Symbol
    surrogate_type_name = string(T, "Surrogate") |> Symbol

    @eval(abstract type $composite_type_name <: $T end)
    @eval(struct $surrogate_type_name <: $T end)

    push!(COMPOSITE_TYPES, composite_type_name)
    push!(SURROGATE_TYPES, surrogate_type_name)
    push!(composite_types, @eval($composite_type_name))
    push!(surrogate_types, @eval($surrogate_type_name))

    # shorthand surrogate constructor:
    @eval($T() = $surrogate_type_name())
end


const Surrogate = Union{surrogate_types...}
const Composite = Union{composite_types...}

MLJModelInterface.is_wrapper(::Type{<:Union{Composite,Surrogate}}) = true
MLJModelInterface.package_name(::Type{<:Union{Composite,Surrogate}}) = "MLJBase"
for T in surrogate_types
    MLJModelInterface.load_path(::Type{T}) = string("MLJBase.", T)
end
