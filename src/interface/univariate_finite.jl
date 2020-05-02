# Extend Distributions type hiearchy to account for non-euclidean supports
abstract type NonEuclidean <: Dist.ValueSupport end

const UnivariateFiniteSuper = Dist.Distribution{Dist.Univariate,NonEuclidean}

# C - original type (eg, Char in `categorical(['a', 'b'])`)
# U - reference type <: Unsigned
# T - raw probability type
# L - subtype of CategoricalValue, eg CategoricalValue{Char,UInt32}

struct UnivariateFinite{C,U,T<:Real} <: UnivariateFiniteSuper
    decoder::CategoricalDecoder{C,U}
    prob_given_class::LittleDict{U,T}
end

"""$(MMI.UNIVARIATE_FINITE_DOCSTRING)"""
UnivariateFinite(a...) = MMI.UnivariateFinite(a...)

# Fallbacks
MMI.UnivariateFinite(::FI, d::AbstractDict) = throw(
    ArgumentError("The support of a `UnivariateFinite` can consist " *
                  "only of `CategoricalString` or `CategoricalValue` " *
                  "elements, and probabilities must be `<: Real`."))
MMI.UnivariateFinite(::FI, c::AbstractVector, p) = throw(
    ArgumentError("The `classes` must have type `AbstractVector{T}` where " *
                  "`T` is of type `CategoricalString` or `CategoricalValue` " *
                  "Perhaps you have `T=Any`?"))

# Univariate Finite from a dictionary of pairs CLASS => PROB
function MMI.UnivariateFinite(
                prob_given_class::AbstractDict{L,T}
                ) where {L<:CategoricalElement,T<:Real}
    # retrieve decoder and classes from element
    an_element     = first(keys(prob_given_class))
    parent_decoder = decoder(an_element)
    parent_classes = classes(an_element)
    given_classes  = keys(prob_given_class)
    given_probs    = values(prob_given_class)

    # check that the probabilities form a probability vector
    Dist.@check_args(UnivariateFinite, Dist.isprobvec(given_probs |> collect))

    # it's expected here that given_classes ⊆ parent_classes
    # then we form a dictionary mapping int(key) => prob
    pairs = ((int(c) => prob_given_class[c])
                for c in intersect(given_classes, parent_classes))

    return UnivariateFinite(parent_decoder, LittleDict(pairs...))
end

# Univariate Finite from a vector of classes and vector of probs
function MMI.UnivariateFinite(::FI, c::AbstractVector{C}, p::AbstractVector{P}
                              ) where C <: CategoricalElement where P <: Real
    # check that the vectors have appropriate length
    Dist.@check_args(UnivariateFinite, length(c) == length(p))
    # it's necessary to force the typing of the LittleDict otherwise it
    # may just convert to an abstractvector without keeping the 'categorical'.
    prob_given_class = LittleDict{C,P}(c[i] => p[i] for i in eachindex(c))
    return MMI.UnivariateFinite(prob_given_class)
end

# ------------------------------------------------------------------------
# utils for univariate finite objects

"""
    classes(d::UnivariateFinite)

A list of categorial elements in the common pool of classes used to
construct `d`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    classes(d) # CategoricalArray{String,1,UInt32}["maybe", "no", "yes"]

"""
MMI.classes(d::UnivariateFinite) = classes(d.decoder.pool)
