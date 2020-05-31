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


## CHECKS AND ERROR MESSAGES

const Prob{P} = Union{P, AbstractArray{P}} where P <: Real

prob_error = ArgumentError("Probabilities must have `Real` type. ")

_err_01() = throw(DomainError("Probabilities must be in [0,1]."))
_err_sum_1() = throw(DomainError(
"Probability arrays must sum to one along the last axis. "))
_err_dim(support, probs) = throw(DimensionMismatch(
"Probability array is incompatible "*
"with the number of classes, $(length(support)), which should "*
"be equal to `$(size(probs)[end])`, the last dimension "*
"of the array. Perhaps you meant to set `augment=true`? "))
_err_dim_augmented(support, probs) = throw(DimensionMismatch(
"Probability array to be augmented is incompatible "*
"with the number of classes, $(length(support)), which should "*
"be one more than `$(size(probs)[end])`, the last dimension "*
    "of the array. "))
_err_aug() = throw(ArgumentError(
    "Array cannot be augmented. There are "*
    "sums along the last axis exceeding one. "))

function _check_pool(pool)
    ismissing(pool) || pool == nothing ||
        @warn "Specified pool ignored, as class labels being "*
    "generated automatically. "
    return nothing
end
_check_probs_01(probs) =
    all(0 .<= probs .<= 1) || _err_01()
_check_probs_sum(probs::Vector{<:Prob{P}}) where P<:Real =
    all(x -> x≈one(P), sum(probs)) || _err_sum_1()
_check_probs(probs) = (_check_probs_01(probs); _check_probs_sum(probs))
_check_augmentable(support, probs) = _check_probs_01(probs) &&
    size(probs)[end] + 1 == length(support) ||
    _err_dim_augmented(support, probs)


## AUGMENTING ARRAYS TO MAKE THEM PROBABILITY ARRAYS

_unwrap(A::Array) = A
_unwrap(A::Vector) = first(A)

# augmentation inserts the sum-subarray *before* the array:
function _augment_probs(support, probs::AbstractArray{P,N}) where {P,N}
    C = length(support)
    _check_augmentable(support, probs)
    size(probs)[end] + 1 == C || _err_dim_augmented(support, probs)
    aug_size = size(probs) |> collect
    aug_size[end] += 1
    augmentation = _unwrap(1 .- sum(probs, dims=N))
    all(0 .<= augmentation .<= 1) || _err_aug()
    aug_probs = Array{P}(undef, aug_size...)
    aug_probs[fill(:, N - 1)..., 2:end] = probs
    aug_probs[fill(:, N - 1)..., 1] = augmentation
    return aug_probs
end


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


# Univariate Finite from a vector of classes and array of probs.
function MMI.UnivariateFinite(::FI,
                              support::AbstractVector{V},
                              probs::AbstractArray{P,M};
                              augment=false,
                              kwargs...) where {V,P<:Real,M}

    if augment
        _probs = _augment_probs(probs)
        N = M
    else
        _probs = probs
        N = M -1
    end

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):

    if N == 0
        prob_given_class = LittleDict{V,P}()
    else
        prob_given_class = LittleDict{V, AbstractArray{P,N}}()
    end
    for i in eachindex(support)
        prob_given_class[support[i]] = _get(_probs, i)
    end

    return MMI.UnivariateFinite(FI(), prob_given_class; kwargs...)

end

function MMI.UnivariateFinite(::FI,
                              probs::AbstractArray{<:Real};
                              pool=nothing,
                              ordered=false,
                              kwargs...)
    _check_pool(pool)
    support = categorical([Symbol("class_$i") for i in 1:size(probs)[end]],
                          ordered=ordered,
                          compress=true)
    return MMI.UnivariateFinite(FI(),
                                support,
                                probs;
                                pool=pool,
                                ordered=ordered,
                                kwargs...)
end
