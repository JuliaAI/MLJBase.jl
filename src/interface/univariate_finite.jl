# Extend Ditributions type hiearchy to account for non-euclidean supports
abstract type NonEuclidean <: Dist.ValueSupport end

const UnivariateFiniteSuper = Dist.Distribution{Dist.Univariate,NonEuclidean}

# V - type of class labels (eg, Char in `categorical(['a', 'b'])`)
# R - reference type <: Unsigned
# P - raw probability type

# Note that the keys of `prob_given_class` need not exhaust all the
# refs of all classes but will be ordered (LittleDicts preserve order)
struct UnivariateFinite{V,R,P<:Real} <: UnivariateFiniteSuper
    decoder::CategoricalDecoder{V,R}
    prob_given_class::LittleDict{R,P} # here "class" actually means "ref"
end


## HELPERS

"""$(MMI.UNIVARIATE_FINITE_DOCSTRING)"""
UnivariateFinite(a...; kwargs...) = MMI.UnivariateFinite(a...; kwargs...)

# Helpers:
_classes(A::Any) = classes(A)
_classes(::Nothing) = throw(ArgumentError(
    "No `CategoricalValue` found from which to extract a "*
    "complete pool of classes.\n"*
    "Possible remedies: \n(i) Specify `pool=c` where `c` is a "*
    "`CategoricalArray` or "*
    "CategoricalValue` with a complete pool;\n"*
    "(ii) Specify `pool=missing` to create a "*
    "new pool with given classes as labels.\nIn case (ii) you must "*
    "also specify `ordered=true` if samples are to be `OrderedFactor`. "))

prob_error = ArgumentError("Probabilities must have `Real` type. ")


## CONSTRUCTORS - FROM DICTIONARY

MMI.UnivariateFinite(::FI, d::AbstractDict; kwargs...) = throw(prob_error)

function MMI.UnivariateFinite(::FI, d::AbstractDict{V,P};
                              pool=nothing,
                              ordered=false) where {V,P<:Real}

    if ismissing(pool)
        v = categorical(collect(keys(d)), ordered=ordered, compress=true)
        support = _classes(v)
    else
        ordered && @warn "Ignoring `ordered` key-word argument as using "*
        "existing pool. "
        raw_support = keys(d) |> collect
        classes = _classes(pool)
        issubset(raw_support, classes) ||
            error("Specified support, $raw_support, not contained in "*
                  "specified pool, $(levels(classes)). ")
        support = filter(classes) do c
            c in raw_support
        end
    end

    prob_given_class = LittleDict([c=>d[get(c)] for c in support])

    return UnivariateFinite(FI(), prob_given_class)
end

function MMI.UnivariateFinite(::FI,
        prob_given_class::AbstractDict{<:CategoricalValue,P}) where P<:Real

    # retrieve decoder and classes from element
    an_element     = first(keys(prob_given_class))
    parent_decoder = decoder(an_element)
    parent_classes = classes(an_element)

    # `LittleDict`s preserve order of keys, which we need for rand():

    given_classes  = keys(prob_given_class) |> collect |> sort
    given_probs    = values(prob_given_class)

    # check that the probabilities form a probability vector
    Dist.@check_args(UnivariateFinite, Dist.isprobvec(given_probs |> collect))

    issubset(given_classes, parent_classes) ||
        error("Categorical elements are not from the same pool. ")


    pairs = [int(c) => prob_given_class[c]
                for c in given_classes]

    return UnivariateFinite(parent_decoder, LittleDict(pairs...))
end


## CONSTRUCTORS - FROM VECTORS

MMI.UnivariateFinite(::FI, ::AbstractVector, ::AbstractVector; kwargs...) =
    throw(prob_error)

# Univariate Finite from a vector of classes and vector of probs
function MMI.UnivariateFinite(::FI,
                              support::AbstractVector{V},
                              probs::AbstractVector{P};
                              kwargs...) where {V,P<:Real}
    # check that the vectors have appropriate length
    Dist.@check_args(UnivariateFinite, length(support) == length(probs))

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):
    prob_given_class = LittleDict{V,P}(support[i] => probs[i]
                                       for i in eachindex(support))
    return MMI.UnivariateFinite(FI(), prob_given_class; kwargs...)
end

function MMI.UnivariateFinite(::FI,
                              probs::AbstractVector{P};
                              pool=nothing,
                              ordered=false,
                              kwargs...) where P<:Real

    ismissing(pool) ||
        error("No support specified. To automatically generate labels for "*
              "a new categorical pool, specify `pool=missing`. "*
              "Additionally specify `ordered=true` if samples "*
              "are to be `OrderedFactor`. ")

    support = categorical([Symbol("class_$i") for i in 1:length(probs)],
                          ordered=ordered,
                          compress=true)

    return MMI.UnivariateFinite(FI(), support, probs; pool=pool, kwargs...)
end
