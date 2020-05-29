# Extend Ditributions type hiearchy to account for non-euclidean supports
abstract type NonEuclidean <: Dist.ValueSupport end

const UnivariateFiniteSuper = Dist.Distribution{Dist.Univariate,NonEuclidean}

# R - reference type <: Unsigned
# V - type of class labels (eg, Char in `categorical(['a', 'b'])`)
# P - raw probability type
# S - scitype of samples

# Note that the keys of `prob_given_ref` need not exhaust all the
# refs of all classes but will be ordered (LittleDicts preserve order)
struct UnivariateFinite{S,V,R,P<:Real} <: UnivariateFiniteSuper
    scitype::Type{S}
    decoder::CategoricalDecoder{V,R}
    prob_given_ref::LittleDict{R,P,Vector{R}, Vector{P}}
end

struct UnivariateFiniteArray{S,V,R,P,N} <:
    AbstractArray{UnivariateFinite{S,V,R,P},N}
    scitype::Type{S}
    decoder::CategoricalDecoder{V,R}
    prob_given_ref::LittleDict{R,Array{P,N},Vector{R}, Vector{Array{P,N}}}
end


## HELPERS

const Prob{P} = Union{P, AbstractArray{P}} where P <: Real

prob_error = ArgumentError("Probabilities must have `Real` type. ")

_err_01() = throw(DomainError("Probabilities must be in [0,1]."))
_err_sum_1() = throw(DomainError(
    "Probabilities must sum to one. If "*
    "specifying as a single array, the array should sum to one "*
    "along the first axis. "))

_check_probs_01(probs) =
    all(0 .<= probs .<= 1) || _err_01()

_check_probs_sum(probs::Vector{<:Prob{P}}) where P<:Real =
    all(x -> x≈one(P), sum(probs)) || _err_sum_1()

_check_probs(probs) = (_check_probs_01(probs); _check_probs_sum(probs))

_check_dims(support, probs) = size(probs, 1) == length(support) ||
    throw(DimensionMismatch(
        "Encountered a support of size  $(length(support)), "*
        "which does not match the first dimension of specified "*
        "probability array, "*
        "$(size(probs, 1)). "))


## CONSTRUCTORS - FROM DICTIONARY

"""$(MMI.UNIVARIATE_FINITE_DOCSTRING)"""
UnivariateFinite(a...; kwargs...) = MMI.UnivariateFinite(a...; kwargs...)

# The following constructor will get called by all the others. It
# returns a UnivariateFinite object *or* a
# UnivariateFiniteArray, depending on the values of the dictionary -
# scalar or array - which represent the probabilities, one for each
# class in the support.
function MMI.UnivariateFinite(
    ::FI,
    prob_given_class::AbstractDict{<:CategoricalValue, <:Prob})

    probs = values(prob_given_class) |> collect
    _check_probs_01.(probs)
    _check_probs_sum(probs)

    # retrieve decoder and classes from element
    class1         = first(keys(prob_given_class))
    S = scitype(class1)
    parent_decoder = decoder(class1)
    parent_classes = classes(class1)

    # `LittleDict`s preserve order of keys, which we need for rand():

    _classes  = keys(prob_given_class) |> collect |> sort

    issubset(_classes, parent_classes) ||
        error("Categorical elements are not from the same pool. ")

    pairs = [int(c) => prob_given_class[c]
                for c in _classes]

    probs1 = first(values(prob_given_class))
    if probs1 isa Real
        return UnivariateFinite(S, parent_decoder, LittleDict(pairs...))
    else
        return UnivariateFiniteArray(S, parent_decoder, LittleDict(pairs...))
    end
end

function MMI.UnivariateFinite(::FI, d::AbstractDict{V,<:Prob};
                              pool=nothing,
                              ordered=false) where V

    if pool === nothing || ismissing(pool)
        if pool === nothing
            @warn "No `CategoricalValue` found from which to extract a "*
            "complete pool of classes. "*
            "Creating a new pool (ordered=$ordered) "*
            "from labels specified. You can:\n"*
            " (i) specify `pool=missing` to suppress this warning; or\n"*
            " (ii) use an existing pool by specifying `pool=c` "*
            "where `c` is a "*
            "`CategoricalArray`, `CategoricalArray` or "*
            "CategoricalPool`.\n"*
            "In case (i) "*
            "specify `ordered=true` if samples are to be `OrderedFactor`. "
        end
        v = categorical(collect(keys(d)), ordered=ordered, compress=true)
        support = classes(v)
    else
        ordered && @warn "Ignoring `ordered` key-word argument as using "*
        "existing pool. "
        raw_support = keys(d) |> collect
        _classes = classes(pool)
        issubset(raw_support, _classes) ||
            error("Specified support, $raw_support, not contained in "*
                  "specified pool, $(levels(classes)). ")
        support = filter(_classes) do c
            c in raw_support
        end
    end

    prob_given_class = LittleDict([c=>d[get(c)] for c in support])

    return UnivariateFinite(FI(), prob_given_class)
end


## CONSTRUCTORS - FROM VECTORS

# example: _get(A, 4) = A[4, :, :] if A has 3 dims:
_get(probs::Array{<:Any,N}, i) where N = probs[i,fill(:,N-1)...]


# Univariate Finite from a vector of classes and array of probs
# summing to one along first axis (M is one more than array dimension):
function MMI.UnivariateFinite(::FI,
                              support::AbstractVector{V},
                              probs::AbstractArray{P,M};
                              kwargs...) where {V,P<:Real,M}
    N = M - 1

    _check_dims(support, probs)

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):

    if N == 0
        prob_given_class = LittleDict{V,P}()
    else
        prob_given_class = LittleDict{V, AbstractArray{P,N}}()
    end
    for i in eachindex(support)
        prob_given_class[support[i]] = _get(probs, i)
    end

    return MMI.UnivariateFinite(FI(), prob_given_class; kwargs...)

end

function MMI.UnivariateFinite(::FI,
                              probs::AbstractArray{<:Real};
                              pool=nothing,
                              ordered=false,
                              kwargs...)

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
