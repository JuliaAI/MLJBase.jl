## CONSTANTS

const Dist = Distributions

## EQUALITY OF DISTRIBUTIONS (minor type piracy)

# TODO: We should get rid of this. I think it is used only in
# MLJModels/test.

function ==(d1::D, d2::D) where D<:Dist.Sampleable
    ret = true
    for fld in fieldnames(D)
        ret = ret && getfield(d1, fld) == getfield(d2, fld)
    end
    return ret
end


## DISTRIBUTION AS TRAIT (needed?)

# fallback:
isdistribution(d) = isdistribution(typeof(d))
isdistribution(::Type{<:Any}) = false

# for anything sampleable in Distributions.jl:
isdistribution(::Type{<:Dist.Sampleable}) = true


## ADD TO distributions.jl TYPE HIERARCHY TO ACCOUNT FOR NON-EUCLIDEAN
## SUPPORTS

abstract type NonEuclidean <: Distributions.ValueSupport end


## A NEW TRAIT FOR DISTRIBUTIONS

support_scitype(::Type) = Unknown
support_scitype(d) = support_scitype(typeof(d))


## UNIVARIATE NOMINAL PROBABILITY DISTRIBUTION

"""
    UnivariateFinite(classes, p)

A discrete univariate distribution whose finite support is the
elements of the vector `classes`, and whose corresponding
probabilities are elements of the vector `p`, which must sum to
one. Here `classes` must have type
`AbstractVector{<:CategoricalElement}` where

    CategoricalElement = Union{CategoricalValue,CategoricalString}

and all classes are assumed to share the same categorical pool.

    UnivariateFinite(prob_given_class)

A discrete univariate distribution whose finite support is the set of
keys of the provided dictionary, `prob_given_class`. The dictionary
keys must be of type `CategoricalElement` (see above) and the dictionary
values specify the corresponding probabilities.

    classes(d::UnivariateFinite)

A list of categorial elements in the common pool of classes used to
construct `d`.

    levels(d::UnivariateFinite)

A list of the raw levels in the common pool of classes used to
construct `d`, equal to `get.(classes(d))`.

    Distributions.support(d::UnivariateFinite)

Those classes associated with non-zero probabilities.

````julia
v = categorical(["yes", "maybe", "no", "yes"])
d = UnivariateFinite(v[1:2], [0.3, 0.7])
pdf(d, "yes")     # 0.3
pdf(d, v[1])      # 0.3
pdf(d, "no")      # 0.0
pdf(d, "house")   # throws error
classes(d) # CategoricalArray{String,1,UInt32}["maybe", "no", "yes"]
levels(d) # Array{String, 1}["maybe", "no", "yes"]
support(d) # CategoricalArray{String,1,UInt32}["maybe", "no"]
mode(d)    # CategoricalString{UInt32} "maybe"
rand(d, 5) # CategoricalArray{String,1,UInt32}["maybe", "no", "maybe", "maybe", "no"] or similar
d = fit(UnivariateFinite, v)
pdf(d, "maybe") # 0.25
```

One can also do weighted fits:

```julia
w = [1, 4, 5, 1] # some weights
d = fit(UnivariateFinite, v, w)
pdf(d, "maybe") ≈ 4/11 # true
````

*Warning:* The `pdf` function will give wrong answers if the order of
 levels of any categorical element passed to the UnivariateFinite
 constructor is changed.

See also `classes`, `support`.

"""
struct UnivariateFinite{L,U,T<:Real} <: Dist.Distribution{Dist.Univariate,NonEuclidean}
    decoder::CategoricalDecoder{L,U}
    prob_given_class::LittleDict{U,T}
end

UnivariateFinite(prob_given_class::AbstractDict) =
    throw(ArgumentError("The support of a UnivariateFinite "*
                        "can consist only of `CategoricalString` "*
                        "or `CategoricalValue` elements, and "*
                        "probabilities must be `AbstractFloat`. "))

function UnivariateFinite(prob_given_class::AbstractDict{L,T}) where {U<:Unsigned,L<:CategoricalElement{U},T<:Real}

    an_element = first(keys(prob_given_class))
    decoder_ = decoder(an_element)

    p = values(prob_given_class) |> collect
    Dist.@check_args(UnivariateFinite, Dist.isprobvec(p))

    d = LittleDict{U,T}()
    for key in classes(an_element)
        haskey(prob_given_class, key) && (d[int(key)] = prob_given_class[key] )
    end
    return UnivariateFinite(decoder_, d)
end

function UnivariateFinite(classes::AbstractVector{<:CategoricalElement},
                          p::AbstractVector{<:Real})
    Dist.@check_args(UnivariateFinite, length(classes)==length(p))
    prob_given_class = LittleDict([classes[i]=>p[i] for i in eachindex(p)])
    return  UnivariateFinite(prob_given_class)
end
UnivariateFinite(classes::AbstractVector, p) =
    throw(ArgumentError("`classes` must have type `AbstractVector{T}` where "*
                        "`T <: Union{CategoricalValue,CategoricalString}. "*
                        "Perhaps you have `T=Any`? "))

classes(d::UnivariateFinite) = classes(d.decoder.pool)
levels(d::UnivariateFinite)  = d.decoder.pool.levels


# get the internal integer representations of the support
raw_support(d::UnivariateFinite) =
    sort!(collect(keys(d.prob_given_class)))

Distributions.support(d::UnivariateFinite) =
    map(d.decoder, raw_support(d))

function Base.show(stream::IO, d::UnivariateFinite)
    raw = sort!(raw_support(d)) # reflects order of pool at
                                # instantiation of d
    x1 = d.decoder(first(raw))
    p1 = d.prob_given_class[first(raw)]
    str = "UnivariateFinite($x1=>$(round(p1, sigdigits=3))"
    pairs = (d.decoder(r)=>d.prob_given_class[r] for r in raw[2:end])
    for pair in pairs
        str *= ", $(pair[1])=>$(round(pair[2], sigdigits=3))"
    end
    str *= ")"
    print(stream, str)
end

support_scitype(::Type{<:UnivariateFinite}) = Finite

"""
    isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)

Returns `true` if and only if `Set(classes(d1) == Set(classes(d2))`
and the corresponding probabilities are approximately equal. The
key-word arguments `kwargs` are passed through to each call of
`isapprox` on probabiliity pairs. Returns `false` otherwise.

"""
function Base.isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)

    classes1 = classes(d1)
    classes2 = classes(d2)

    for c in classes1
        c in classes2 || return false
        isapprox(pdf(d1, c), pdf(d2, c); kwargs...) ||
            return false # pdf defined below
    end
    return true
end

function average(dvec::AbstractVector{UnivariateFinite{L,U,T}};
                 weights=nothing) where {L,U,T}

    n = length(dvec)

    Dist.@check_args(UnivariateFinite, weights == nothing || n==length(weights))

    # check all distributions have consistent pool:
    first_index = first(dvec).decoder.pool.index
    for d in dvec
        d.decoder.pool.index == first_index ||
            error("Averaging UnivariateFinite distributions with incompatible"*
                  " pools. ")
    end

    # get all refs:
    refs = Tuple(reduce(union, [keys(d.prob_given_class) for d in dvec]))

    # initialize the prob dictionary for the distribution sum:
    prob_given_class = LittleDict{U,T}(refs, zeros(T, length(refs)))

    # make vector of all the distributions dicts padded to have same common keys:
    prob_given_class_vec = map(dvec) do d
        merge(prob_given_class, d.prob_given_class)
    end

    # sum up:
    if weights == nothing
        scale = 1/n
        for x in refs
            for k in 1:n
                prob_given_class[x] += scale*prob_given_class_vec[k][x]
            end
        end
    else
        scale = 1/sum(weights)
        for x in refs
            for k in 1:n
                prob_given_class[x] +=
                    weights[k]*prob_given_class_vec[k][x]*scale
            end
        end
    end

    return UnivariateFinite(first(dvec).decoder, prob_given_class)
end

function _pdf(d::UnivariateFinite{L,U,T}, ref) where {L,U,T}
    return get(d.prob_given_class, ref, zero(T))
end

Distributions.pdf(d::UnivariateFinite{L,U,T},
                  x::CategoricalElement) where {L,U,T} = _pdf(d, int(x))

function Distributions.pdf(d::UnivariateFinite{L,U,T},
                           level::L) where {L,U,T}
    p = d.decoder.pool
    if level in p.levels
        return _pdf(d, int(p, level))
    else
        throw(ArgumentError(""))
    end
end


function Distributions.mode(d::UnivariateFinite)
    dic = d.prob_given_class
    p = values(dic)
    max_prob = maximum(p)
    m = first(first(dic)) # mode, just some ref for now
    for (x, prob) in dic
        if prob == max_prob
            m = x
            break
        end
    end
    return d.decoder(m)
end

"""
    _cummulative(d::UnivariateFinite)

Return the cummulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_class`. Used only for to implement random sampling from
`d`.

"""
function _cummulative(d::UnivariateFinite{L,U,T}) where {L,U,T<:Real}
    p = collect(values(d.prob_given_class))
    K = length(p)
    p_cummulative = Array{T}(undef, K + 1)
    p_cummulative[1] = zero(T)
    p_cummulative[K + 1] = one(T)
    for i in 2:K
        p_cummulative[i] = p_cummulative[i-1] + p[i-1]
    end
    return p_cummulative
end


"""
_rand(p_cummulative)

Randomly sample the distribution with discrete support `1:n` which has
cummulative probability vector `p_cummulative=[0, ..., 1]` (of length
`n+1`). Does not check the first and last elements of `p_cummulative`
but does not use them either.

"""
function _rand(p_cummulative)
    real_sample = rand()
    K = length(p_cummulative)
    index = K
    for i in 2:K
        if real_sample < p_cummulative[i]
            index = i - 1
            break
        end
    end
    return index
end

function Base.rand(d::UnivariateFinite)
    p_cummulative = _cummulative(d)
    classes = d.decoder.pool.valindex[collect(keys(d.prob_given_class))]
    return classes[_rand(p_cummulative)]
end

function Base.rand(d::UnivariateFinite, n::Int)
    p_cummulative = _cummulative(d)
    classes = d.decoder.pool.valindex[collect(keys(d.prob_given_class))]
    return [classes[_rand(p_cummulative)] for i in 1:n]
end

function Distributions.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{L}) where L
    L <: CategoricalElement ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "CategoricalValue or CategoricalString type. ")
    y = skipmissing(v) |> collect
    isempty(y) && error("No non-missing data to fit. ")
    N = length(y)
    count_given_class = Dist.countmap(y)
    classes = Tuple(keys(count_given_class))
    probs = values(count_given_class)./N
    prob_given_class = LittleDict(classes, probs)
    return UnivariateFinite(prob_given_class)
end

function Distributions.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{L},
                           weights::AbstractVector{<:Real}) where L
    L <: CategoricalElement ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "CategoricalValue or CategoricalString type. ")
    y = broadcast(identity, skipmissing(v))
    isempty(y) && error("No non-missing data to fit. ")
    classes_seen = filter(in(unique(y)), classes(y[1]))

    # instantiate and initialize prob dictionary:
    prob_given_class = LittleDict{L,Float64}()
    for c in classes_seen
        prob_given_class[c] = 0
    end

    # compute unnormalized  probablilities:
    for i in eachindex(y)
        prob_given_class[y[i]] += weights[i]
    end

    # normalize the probabilities:
    S = sum(values(prob_given_class))
    for c in keys(prob_given_class)
        prob_given_class[c] /=S
    end

    return UnivariateFinite(prob_given_class)

end


## NORMAL DISTRIBUTION ARITHMETIC

# *(lambda::Number, d::Dist.Normal) = Dist.Normal(lambda*d.μ, lambda*d.σ)
# function +(ds::Dist.Normal...)
#     μ = sum([d.μ for d in ds])
#     σ = sum([d.σ^2 for d in ds]) |> sqrt
#     return Dist.Normal(μ, σ)
# end

# hack for issue #809 in Dist.jl:
Distributions.fit(::Type{Dist.Normal{T}}, args...) where T =
    Dist.fit(Dist.Normal, args...)
