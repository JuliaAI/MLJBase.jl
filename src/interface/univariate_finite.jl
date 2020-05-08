# Extend Distributions type hiearchy to account for non-euclidean supports
abstract type NonEuclidean <: Dist.ValueSupport end

const UnivariateFiniteSuper = Dist.Distribution{Dist.Univariate,NonEuclidean}

# C - original type (eg, Char in `categorical(['a', 'b'])`)
# U - reference type <: Unsigned
# T - raw probability type
# L - subtype of CategoricalValue, eg CategoricalValue{Char,UInt32}


# Note that the keys of `prob_given_class` need not exhaust all the
# refs of all classes but will be ordered (LittleDicts preserve order)
struct UnivariateFinite{C,U,T<:Real} <: UnivariateFiniteSuper
    decoder::CategoricalDecoder{C,U}
    prob_given_class::LittleDict{U,T} # here "class" actually means "ref"
end

"""$(MMI.UNIVARIATE_FINITE_DOCSTRING)"""
UnivariateFinite(a...) = MMI.UnivariateFinite(a...)

# Fallbacks
MMI.UnivariateFinite(::FI, d::AbstractDict) = throw(
    ArgumentError("The support of a `UnivariateFinite` can consist " *
                  "only `CategoricalValue` objects. " *
                  "elements, and probabilities must be `<: Real`."))
MMI.UnivariateFinite(::FI, c::AbstractVector, p) =
    throw(ArgumentError("The `classes` must have type "*
                        "`AbstractVector{<:CategoricalValue}`. "*
                        "Perhaps you have `T=Any`?"))

# Univariate Finite from a dictionary of pairs CLASS => PROB
function MMI.UnivariateFinite(
                prob_given_class::AbstractDict{L,T}
                ) where {L<:CategoricalValue,T<:Real}
    # retrieve decoder and classes from element
    an_element     = first(keys(prob_given_class))
    parent_decoder = decoder(an_element)
    parent_classes = classes(an_element)

    # `LittleDict`s preserve order of keys, which we need for rand():

    # After https://github.com/JuliaData/CategoricalArrays.jl/issues/269
    # is resolved:
    # OPTION A BEGINS
    # given_classes  = keys(prob_given_class) |> collect |> sort
    # OPTION A ENDS

    # Hack for now:
    # OPTION B BEGINS
    given_classes_unsorted = keys(prob_given_class) |> collect
    given_classes_int = int.(given_classes_unsorted) |> sort
    given_classes = parent_classes[given_classes_int]
    # OPTION B ENDS

    given_probs    = values(prob_given_class)

    # check that the probabilities form a probability vector
    Dist.@check_args(UnivariateFinite, Dist.isprobvec(given_probs |> collect))

    issubset(given_classes, parent_classes) ||
        error("Categorical elements are not from the same pool. ")


    pairs = [int(c) => prob_given_class[c]
                for c in given_classes]

    return UnivariateFinite(parent_decoder, LittleDict(pairs...))
end

# Univariate Finite from a vector of classes and vector of probs
function MMI.UnivariateFinite(::FI, c::AbstractVector{C}, p::AbstractVector{P}
                              ) where C <: CategoricalValue where P <: Real
    # check that the vectors have appropriate length
    Dist.@check_args(UnivariateFinite, length(c) == length(p))

    # it's necessary to force the typing of the LittleDict otherwise it
    # may just convert to an abstractvector without keeping the 'categorical'.
    prob_given_class = LittleDict{C,P}(c[i] => p[i] for i in eachindex(c))
    return MMI.UnivariateFinite(prob_given_class)
end


