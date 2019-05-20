## CONSTANTS

const Dist = Distributions


## EQUALITY OF DISTRIBUTIONS (minor type piracy)

function ==(d1::D, d2::D) where D<:Dist.Sampleable
    ret = true
    for fld in fieldnames(D)
        ret = ret && getfield(d1, fld) == getfield(d2, fld)
    end
    return ret
end


# DISTRIBUTION AS TRAIT (needed?)

# fallback:
isdistribution(d) = isdistribution(typeof(d))
isdistribution(::Type{<:Any}) = false

# for anything sampleable in Distributions.jl:
isdistribution(::Type{<:Dist.Sampleable}) = true


# ADD TO distributions.jl TYPE HIERARCHY TO ACCOUNT FOR NON-EUCLIDEAN
# SUPPORTS

abstract type NonEuclidean <: Distributions.ValueSupport end


## UNIVARIATE NOMINAL PROBABILITY DISTRIBUTION

# to pad a dictionary of probabilities with zeros for unseen levels,
# if the key type is CategoricalElement:
pad_probabilities(prob_given_level) = prob_given_level
function pad_probabilities(prob_given_level::Dict{<:CategoricalElement, T}) where T
    proto_element = first(keys(prob_given_level))
    return merge!(Dict([y=>zero(T) for y in classes(proto_element)]),
                  prob_given_level)
end

"""
    UnivariateFinite(levels, p)

A discrete univariate distribution whose finite support is the
elements of the vector `levels`, and whose corresponding probabilities
are elements of the vector `p`, which must sum to one.

In the special case that `levels` has type `AbstractVector{L}` where
`L <: CategoricalValue` or `L <: CategoricalString` (for example
`levels` is a `CategoricalVector`) the constructor adds the unobserved
classes (from the common pool) with probability zero.

    UnivariateFinite(prob_given_level)

A discrete univariate distribution whose finite support is the set of
keys of the provided dictionary, `prob_given_level`. The dictionary
values specify the corresponding probabilities, which must be
nonnegative and sum to one. 

In the special case that `keys(prob_given_level)` has type
`AbstractVector{L}` where `L <: CategoricalValue` or `L <:
CategoricalString` (for example it is a `CategoricalVector`) the
constructor adds the unobserved classes from the common pool with
probability zero.

    levels(d::UnivariateFinite)

Return the levels of `d`.

````julia
d = UnivariateFinite(["yes", "no", "maybe"], [0.1, 0.2, 0.7])
pdf(d, "no") # 0.2
mode(d) # "maybe"
rand(d, 5) # ["maybe", "no", "maybe", "maybe", "no"]
d = fit(UnivariateFinite, ["maybe", "no", "maybe", "yes"])
pdf(d, "maybe") ≈ 0.5 # true
levels(d) # ["yes", "no", "maybe"]
````

If the element type of `v` is a `CategoricalValue` or
`CategoricalString`, then `fit(UnivariateFinite, v)` assigns a
probability of zero to unobserved classes from the common pool.

See also `classes`.

"""
struct UnivariateFinite{L,T<:Real} <: Dist.Distribution{Dist.Univariate,NonEuclidean}
    prob_given_level::Dict{L,T}
    function UnivariateFinite{L,T}(prob_given_level::Dict{L,T}) where {L,T<:Real}
        p = values(prob_given_level) |> collect
        Dist.@check_args(UnivariateFinite, Dist.isprobvec(p))
        return new{L,T}(pad_probabilities(prob_given_level))
    end
end

UnivariateFinite(prob_given_level::Dict{L,T}) where {L,T<:Real} =
    UnivariateFinite{L,T}(prob_given_level)

function UnivariateFinite(levels::AbstractVector, p::AbstractVector{<:Real})
        Dist.@check_args(UnivariateFinite, length(levels)==length(p))
    prob_given_level = Dict([levels[i]=>p[i] for i in eachindex(p)])
    return  UnivariateFinite(prob_given_level)
end

CategoricalArrays.levels(d::UnivariateFinite) = collect(keys(d.prob_given_level))

function average(dvec::Vector{UnivariateFinite{L,T}}; weights=nothing) where {L,T}

    n = length(dvec)
    
    Dist.@check_args(UnivariateFinite, weights == nothing || n==length(weights))

    if weights == nothing
        weights = fill(1/n, n)
    else
        weights = weights/sum(weights)
    end
            
    # get all levels:
    levels = reduce(union, [keys(d.prob_given_level) for d in dvec])

    z = Dict{L,T}([x => zero(T) for x in levels]...)
    prob_given_level_vec = map(dvec) do d
        merge(z, d.prob_given_level)
    end

    # initialize the prob dictionary for the distribution sum:
    prob_given_level = Dict{L,T}()
    for x in levels
        prob_given_level[x] = zero(T)
    end
    
    # sum up:
    for x in levels
        for k in 1:n
            prob_given_level[x] += weights[k]*prob_given_level_vec[k][x]
        end
    end

    return UnivariateFinite(prob_given_level)

end        

function Distributions.mode(d::UnivariateFinite)
    dic = d.prob_given_level
    p = values(dic)
    max_prob = maximum(p)
    m = first(first(dic)) # mode, just some level for now
    for (x, prob) in dic
        if prob == max_prob
            m = x
            break
        end
    end
    return m
end

Distributions.pdf(d::UnivariateFinite, x) = d.prob_given_level[x]

"""
    _cummulative(d::UnivariateFinite)

Return the cummulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_level`. Used only for to implement random sampling from
`d`.

"""
function _cummulative(d::UnivariateFinite{L,T}) where {L,T<:Real}
    p = collect(values(d.prob_given_level))
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
    levels = collect(keys(d.prob_given_level))
    return levels[_rand(p_cummulative)]
end

function Base.rand(d::UnivariateFinite, n::Int)
    p_cummulative = _cummulative(d)
    levels = collect(keys(d.prob_given_level))
    return [levels[_rand(p_cummulative)] for i in 1:n]
end

function Distributions.fit(d::Type{<:UnivariateFinite}, v::AbstractVector)
    vpure = skipmissing(v) |> collect
    isempty(vpure) && error("No non-missing data to fit. ")
    N = length(vpure)
    count_given_level = Dist.countmap(vpure)
    prob_given_level = Dict([x=>c/N for (x, c) in count_given_level])
    return UnivariateFinite(prob_given_level)
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

