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
    "new pool limited to the classes given. \n In case (ii) you must "*
    "also specify `ordered=true` to order the pool. "))

prob_error = ArgumentError("Probabilities must have `Real` type. ")


## CONSTRUCTORS - FROM DICTIONARY

MMI.UnivariateFinite(::FI, d::AbstractDict; kwargs...) = throw(prob_error)

function MMI.UnivariateFinite(::FI, d::AbstractDict{C,T};
                              pool=nothing,
                              ordered=false) where {C,T<:Real}
    if ismissing(pool)
        v = categorical(collect(keys(d)), ordered=ordered, compress=true)
        classes = _classes(v)
    else
        raw_support = keys(d) |> collect
        classes = filter(_classes(pool)) do c
            c in raw_support
        end
    end

    prob_given_class = LittleDict([c=>d[get(c)] for c in classes])

    return UnivariateFinite(FI(), prob_given_class)
end

function MMI.UnivariateFinite(::FI,
        prob_given_class::AbstractDict{<:CategoricalValue,T}) where T<:Real

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


## CONSTRUCTORS - FROM VECTORS

MMI.UnivariateFinite(::FI, c::AbstractVector, p; kwargs...) =
    throw(prob_error)

# function MMI.UnivariateFinite(::FI, c::AbstractVector,
#                      p::AbstractVector{P};
#                      kwargs...) where P<:Real
#     # check that the vectors have appropriate length
#     Dist.@check_args(UnivariateFinite, length(c) == length(p))
# end


# Univariate Finite from a vector of classes and vector of probs
function MMI.UnivariateFinite(::FI,
                              c::AbstractVector{C},
                              p::AbstractVector{P};
                              kwargs...) where {C,P<:Real}
    # check that the vectors have appropriate length
    Dist.@check_args(UnivariateFinite, length(c) == length(p))

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):
    prob_given_class = LittleDict{C,P}(c[i] => p[i] for i in eachindex(c))
    return MMI.UnivariateFinite(FI(), prob_given_class; kwargs...)
end
