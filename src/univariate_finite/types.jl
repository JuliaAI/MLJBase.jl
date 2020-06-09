# Extend Ditributions type hiearchy to account for non-euclidean supports
abstract type NonEuclidean <: Dist.ValueSupport end

const UnivariateFiniteSuper = Dist.Distribution{Dist.Univariate,NonEuclidean}

# R - reference type <: Unsigned
# V - type of class labels (eg, Char in `categorical(['a', 'b'])`)
# P - raw probability type
# S - scitype of samples
# L - raw type of labels, eg `Symbol` or `String`

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

const UnivariateFiniteVector{S,V,R,P} = UnivariateFiniteArray{S,V,R,P,1}


## CONSTRUCTORS TO EXTEND MLJModelInterface

"""$(MMI.UNIVARIATE_FINITE_DOCSTRING)"""
UnivariateFinite(a...; kwargs...) = MMI.UnivariateFinite(a...; kwargs...)


## CHECKS AND ERROR MESSAGES

const Prob{P} = Union{P, AbstractArray{P}} where P <: Real

prob_error = ArgumentError("Probabilities must have `Real` type. ")

_err_01() = throw(DomainError("Probabilities must be in [0,1]."))
_err_sum_1() = throw(DomainError(
    "Probability arrays must sum to one along the last axis. Perhaps "*
"you meant to specify `augment=true`? "))
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

isbinary(support) = length(support) == 2

# augmentation inserts the sum-subarray *before* the array:
_augment_probs(support, probs) =
    _augment_probs(Val(isbinary(support)), support, probs,)
function _augment_probs(::Val{false},
                        support,
                        probs::AbstractArray{P,N}) where {P,N}
    _check_augmentable(support, probs)
    aug_size = size(probs) |> collect
    aug_size[end] += 1
    augmentation = _unwrap(one(P) .- sum(probs, dims=N))
    all(0 .<= augmentation .<= 1) || _err_aug()
    aug_probs = Array{P}(undef, aug_size...)
    aug_probs[fill(:, N - 1)..., 2:end] = probs
    aug_probs[fill(:, N - 1)..., 1] = augmentation
    return aug_probs
end
function _augment_probs(::Val{true},
                        support,
                        probs::AbstractArray{P,N}) where {P,N}
    _check_probs_01(probs)
    aug_size = [size(probs)..., 2]
    augmentation = one(P) .- probs
    all(0 .<= augmentation .<= 1) || _err_aug()
    aug_probs = Array{P}(undef, aug_size...)
    aug_probs[fill(:, N)..., 2] = probs
    aug_probs[fill(:, N)..., 1] = augmentation
    return aug_probs
end


## CONSTRUCTORS - FROM DICTIONARY

# The following constructor will get called by all the others. It
# returns a UnivariateFinite object *or* a UnivariateFiniteArray,
# depending on the values of the dictionary - scalar or array - which
# represent the probabilities, one for each class in the support.
function MMI.UnivariateFinite(
    ::FI,
    prob_given_class::AbstractDict{<:CategoricalValue, <:Prob};
    kwargs...)

    # this constructor ignores kwargs

    probs = values(prob_given_class) |> collect
    _check_probs_01.(probs)
    _check_probs_sum(probs)

    # retrieve decoder and classes from element
    class1         = first(keys(prob_given_class))
    parent_decoder = decoder(class1)
    parent_classes = classes(class1)

    # `LittleDict`s preserve order of keys, which we need for rand():

    support  = keys(prob_given_class) |> collect |> sort

    issubset(support, parent_classes) ||
        error("Categorical elements are not from the same pool. ")

    pairs = [int(c) => prob_given_class[c]
                for c in support]

    probs1 = first(values(prob_given_class))
    S = scitype(class1)
    if probs1 isa Real
        return UnivariateFinite(S, parent_decoder, LittleDict(pairs...))
    else
        return UnivariateFiniteArray(S, parent_decoder, LittleDict(pairs...))
    end
end

function MMI.UnivariateFinite(::FI, d::AbstractDict{V,<:Prob};
                              pool=nothing,
                              ordered=false) where V

    ismissing(pool) &&
        throw(ArgumentError(
            "You cannot specify `pool=missing` "*
            "if passing `UnivariateFinite` a dictionary"))

    pool === nothing && throw(ArgumentError(
        "You must specify `pool=c` "*
        "where `c` is a "*
        "`CategoricalArray`, `CategoricalArray` or "*
        "CategoricalPool`"))

    ordered && @warn "Ignoring `ordered` key-word argument as using "*
    "specified pool to order. "

    raw_support = keys(d) |> collect
    _classes = classes(pool)
    issubset(raw_support, _classes) ||
        error("Specified support, $raw_support, not contained in "*
              "specified pool, $(levels(classes)). ")
    support = filter(_classes) do c
        c in raw_support
    end

    prob_given_class = LittleDict([c=>d[get(c)] for c in support])

    return UnivariateFinite(FI(), prob_given_class)
end


## CONSTRUCTORS - FROM ARRAYS

# example: _get(A, 4) = A[:, :, 4] if A has 3 dims:
_get(probs::AbstractArray{<:Any,N}, i) where N = probs[fill(:,N-1)..., i]

# 1. Univariate Finite from a vector of classes or raw labels and
# array of probs; first, a dispatcher:
function MMI.UnivariateFinite(
    ::FI,
    support::AbstractVector,
    probs::Union{AbstractArray,Real};
    kwargs...)

    if support isa AbstractArray{<:CategoricalValue}
        if :pool in keys(kwargs)
            @warn "Ignoring value of `pool` as the specified "*
            "support defines one already. "
        end
        if :ordered in keys(kwargs)
            @warn "Ignoring value of `ordered` as the "*
            "specified support defines an order already. "
        end
    end

    return _UnivariateFinite(Val(isbinary(support)),
                             support,
                             probs;
                             kwargs...)
end

# The core method, ultimately called by 1.0, 1.1, 1.2, 1.3 below, or
# directly from the dispatcher 1. above
function _UnivariateFinite(support::AbstractVector{CategoricalValue{V,R}},
                           probs::AbstractArray{P},
                           N;
                           augment=false,
                           kwargs...) where {V,R,P<:Real}

    unique(support) == support ||
        error("Non-unique vector of classes specified")

    _probs = augment ? _augment_probs(support, probs) : probs

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):

    if N == 0
        prob_given_class = LittleDict{CategoricalValue{V,R},P}()
    else
        prob_given_class =
            LittleDict{CategoricalValue{V,R}, AbstractArray{P,N}}()
    end
    for i in eachindex(support)
        prob_given_class[support[i]] = _get(_probs, i)
    end

    # calls dictionary constructor above:
    return MMI.UnivariateFinite(FI(), prob_given_class; kwargs...)
end

# 1.0 support does not consist of categorical elements:
function _UnivariateFinite(support::AbstractVector{L},
                           probs::AbstractArray{P},
                           N;
                           augment=false,
                           pool=nothing,
                           ordered=false) where {L,P<:Real}

    # If we got here, then L<:CategoricalValue is not true, ie L is a
    # raw label type

    if pool === nothing || ismissing(pool)
        if pool === nothing
            @warn "No `CategoricalValue` found from which to extract a "*
            "complete pool of classes. "*
            "Creating a new pool (ordered=$ordered) "*
            "You can:\n"*
            " (i) specify `pool=missing` to suppress this warning; or\n"*
            " (ii) use an existing pool by specifying `pool=c` "*
            "where `c` is a "*
            "`CategoricalArray`, `CategoricalArray` or "*
            "CategoricalPool`.\n"*
            "In case (i) "*
            "specify `ordered=true` if samples are to be `OrderedFactor`. "
        end
        v = categorical(support, ordered=ordered, compress=true)
        levels!(v, support)
        _support = classes(v)
    else
        _classes = classes(pool)
        issubset(support, _classes) ||
            error("Specified support, $support, not contained in "*
                  "specified pool, $(levels(classes)). ")
        _support = filter(_classes) do c
            c in support
        end
    end

    # calls core method:
    return _UnivariateFinite(_support, probs, N;
                             augment=augment, pool=pool, ordered=ordered)
end

# 1.1 generic (non-binary) case:
_UnivariateFinite(::Val{false},
                  support::AbstractVector,
                  probs::AbstractArray{<:Any,M};
                  augment=false,
                  kwargs...) where M =
                      _UnivariateFinite(support,
                                        probs,
                                        M - 1;
                                        augment=augment,
                                        kwargs...)

# 1.2 degenerate (binary) case:
_UnivariateFinite(::Val{true},
                  support::AbstractVector,
                  probs::AbstractArray{<:Any,M};
                  augment=false,
                  kwargs...) where M =
                      _UnivariateFinite(support,
                                        probs,
                                        augment ? M : M - 1;
                                        augment=augment,
                                        kwargs...)

# 1.3 corner case, probs a scalar:
_UnivariateFinite(::Val{true},
                  support::AbstractVector,
                  probs::Real;
                  kwargs...) =
                      UnivariateFinite(support, [probs,]; kwargs...)[1]

# 2. probablity only; unspecified support:
function MMI.UnivariateFinite(::FI,
                              probs::AbstractArray{<:Real,N};
                              pool=nothing,
                              augment=false,
                              kwargs...) where N
    _check_pool(pool)

    # try to infer number of classes:
    if N == 1
        if augment
            c = 2
        else
            c = length(probs)
        end
    elseif N == 2
        if augment
            c = size(probs, 2) + 1
        else
            c = size(probs, 2)
        end
    else
        throw(ArgumentError(
            "You need to explicitly specify a support for "*
            "probablility arrays of three "*
            "or more dimensions. "))
    end

    support = [Symbol("class_$i") for i in 1:c]

    return MMI.UnivariateFinite(FI(),
                                support,
                                probs;
                                pool=pool,
                                augment=augment,
                                kwargs...)
end
