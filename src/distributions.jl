abstract type Distribution end

function ==(d1::D, d2::D) where D<:Distribution
    ret = true
    for fld in fieldnames(D)
        ret = ret && getfield(d1, fld) == getfield(d2, fld)
    end
    return ret
end


# DISTRIBUTION AS TRAIT

# fallback
isdistribution(::Type{<:Any}) = false

# for distributions in Distributions.jl:
isdistribution(::Type{<:Distributions.Distribution}) = true

# for MLJ custom distributions defined below:
isdistribution(::Type{Distribution}) = true  

isdistribution(d) = isdistribution(typeof(d))


## UNIVARIATE NOMINAL PROBABILITY DISTRIBUTION

"""
    UnivariateNominal(prob_given_level)

A discrete univariate distribution whose finite support is the set of keys of the
provided dictionary, `prob_given_level`. The dictionary values specify
the corresponding probabilities, which must be nonnegative and sum to
one.

    UnivariateNominal(levels, p)

A discrete univariate distribution whose finite support is the
elements of the vector `levels`, and whose corresponding probabilities
are elements of the vector `p`.

    levels(d::UnivariateNominal)

Return the levels of `d`.

````julia
d = UnivariateNominal(["yes", "no", "maybe"], [0.1, 0.2, 0.7])
pdf(d, "no") # 0.2
mode(d) # "maybe"
rand(d, 5) # ["maybe", "no", "maybe", "maybe", "no"]
d = fit(UnivariateNominal, ["maybe", "no", "maybe", "yes"])
pdf(d, "maybe") ≈ 0.5 # true
levels(d) # ["yes", "no", "maybe"]
````

If `v` is a `CategoricalVector` then `fit(UnivariateNominal, v)`
includes *all* levels in pool of `v` in its support, assigning unseen
levels probability zero.

"""
struct UnivariateNominal{L,T<:Real} <: Distribution
    prob_given_level::Dict{L,T}
    function UnivariateNominal{L,T}(prob_given_level::Dict{L,T}) where {L,T<:Real}
        p = values(prob_given_level) |> collect
        Distributions.@check_args(UnivariateNominal, Distributions.isprobvec(p))
        return new{L,T}(prob_given_level)
    end
end
UnivariateNominal(prob_given_level::Dict{L,T}) where {L,T<:Real} =
    UnivariateNominal{L,T}(prob_given_level)

function UnivariateNominal(levels::Union{Vector{L},CategoricalVector{L}},
                           p::Vector{T}) where {L,T<:Real}
        Distributions.@check_args(UnivariateNominal, length(levels)==length(p))
        prob_given_level = Dict{L,T}()
        for i in eachindex(p)
            prob_given_level[levels[i]] = p[i]
        end
        return  UnivariateNominal(prob_given_level)
end

CategoricalArrays.levels(d::UnivariateNominal) = keys(d.prob_given_level) |> collect

function average(dvec::Vector{UnivariateNominal{L,T}}; weights=nothing) where {L,T}

    n = length(dvec)
    
    Distributions.@check_args(UnivariateNominal, weights == nothing || n==length(weights))

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

    return UnivariateNominal(prob_given_level)

end        

function Distributions.mode(d::UnivariateNominal)
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

Distributions.pdf(d::UnivariateNominal, x) = d.prob_given_level[x]

"""
    _cummulative(d::UnivariateNominal)

Return the cummulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_level`. Used only for to implement random sampling from
`d`.

"""
function _cummulative(d::UnivariateNominal{L,T}) where {L,T<:Real}
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

function Base.rand(d::UnivariateNominal)
    p_cummulative = _cummulative(d)
    levels = collect(keys(d.prob_given_level))
    return levels[_rand(p_cummulative)]
end

function Base.rand(d::UnivariateNominal, n::Int)
    p_cummulative = _cummulative(d)
    levels = collect(keys(d.prob_given_level))
    return [levels[_rand(p_cummulative)] for i in 1:n]
end
    
function Distributions.fit(d::Type{<:UnivariateNominal}, v::AbstractVector{L}) where L
    N = length(v)
    prob_given_level = Dict{L,Float64}()
    count_given_level = Distributions.countmap(v)
    for (x, c) in count_given_level
        prob_given_level[x] = c/N
    end
    return UnivariateNominal(prob_given_level)
end


# if fitting to categorical array, must include missing levels with prob zero
function Distributions.fit(d::Type{<:UnivariateNominal}, v::CategoricalVector{L,R,V}) where {L,R,V}
    N = length(skipmissing(v) |> collect)
    prob_given_level = Dict{V,Float64}() # V is the type of levels
    for x in levels(v)
        prob_given_level[x] = 0
    end
    count_given_level = Distributions.countmap(skipmissing(v) |> collect)
    for (x, c) in count_given_level
        prob_given_level[x] = c/N
    end
    return UnivariateNominal(prob_given_level)
end
    
    
## NORMAL DISTRIBUTION ARITHMETIC

# *(lambda::Number, d::Distributions.Normal) = Distributions.Normal(lambda*d.μ, lambda*d.σ)
# function +(ds::Distributions.Normal...)
#     μ = sum([d.μ for d in ds])
#     σ = sum([d.σ^2 for d in ds]) |> sqrt
#     return Distributions.Normal(μ, σ)
# end

# hack for issue #809 in Distributions.jl:
Distributions.fit(::Type{Distributions.Normal{T}}, args...) where T = Distributions.fit(Distributions.Normal, args...)

