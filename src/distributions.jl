support_scitype(::Type) = Unknown
support_scitype(d) = support_scitype(typeof(d))
support_scitype(::Type{<:UnivariateFinite}) = Finite

# NOTE: see also interface/univariate_finite.jl

"""
    levels(d::UnivariateFinite)

A list of the raw levels in the common pool of classes used to
construct `d`, equal to `get.(classes(d))`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    levels(d) # Array{String, 1}["maybe", "no", "yes"]
"""
levels(d::UnivariateFinite)  = d.decoder.pool.levels


# get the internal integer representations of the support
raw_support(d::UnivariateFinite) =
    sort!(collect(keys(d.prob_given_class)))

"""
    Distributions.support(d::UnivariateFinite)

Those classes associated with non-zero probabilities.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    support(d) # CategoricalArray{String,1,UInt32}["maybe", "no"]
"""
Distributions.support(d::UnivariateFinite) = map(d.decoder, raw_support(d))


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


"""
    isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)

Returns `true` if and only if `Set(classes(d1) == Set(classes(d2))` and the
corresponding probabilities are approximately equal. The key-word arguments
`kwargs` are passed through to each call of `isapprox` on probability pairs.
Returns `false` otherwise.
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

"""
    Distributions.pdf(d::UnivariateFinite, x)

Probability of `d` at `x`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    pdf(d, "yes")     # 0.3
    pdf(d, v[1])      # 0.3
    pdf(d, "no")      # 0.0
    pdf(d, "house")   # throws error

Other similar methods are available too:

    mode(d)    # CategoricalString{UInt32} "maybe"
    rand(d, 5) # CategoricalArray{String,1,UInt32}["maybe", "no", "maybe", "maybe", "no"] or similar
    d = fit(UnivariateFinite, v)
    pdf(d, "maybe") # 0.25

One can also do weighted fits:

    w = [1, 4, 5, 1] # some weights
    d = fit(UnivariateFinite, v, w)
    pdf(d, "maybe") ≈ 4/11 # true

*Warning:* The `pdf` function will give wrong answers if the order of
 levels of any categorical element passed to the UnivariateFinite
 constructor is changed.

See also `classes`, `support`.
"""
Distributions.pdf(d::UnivariateFinite, x::CategoricalElement) = _pdf(d, int(x))

function Distributions.pdf(d::UnivariateFinite, level)
    p = d.decoder.pool
    if level in p.levels
        ref = p.order[p.invindex[level]]
        return _pdf(d, ref)
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
    _cumulative(d::UnivariateFinite)

Return the cumulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_class`. Used only for to implement random sampling from
`d`.

"""
function _cumulative(d::UnivariateFinite{L,U,T}) where {L,U,T<:Real}
    p = collect(values(d.prob_given_class))
    K = length(p)
    p_cumulative = Array{T}(undef, K + 1)
    p_cumulative[1] = zero(T)
    p_cumulative[K + 1] = one(T)
    for i in 2:K
        p_cumulative[i] = p_cumulative[i-1] + p[i-1]
    end
    return p_cumulative
end


"""
_rand(p_cumulative)

Randomly sample the distribution with discrete support `1:n` which has
cumulative probability vector `p_cumulative=[0, ..., 1]` (of length
`n+1`). Does not check the first and last elements of `p_cumulative`
but does not use them either.

"""
function _rand(p_cumulative)
    real_sample = rand()
    K = length(p_cumulative)
    index = K
    for i in 2:K
        if real_sample < p_cumulative[i]
            index = i - 1
            break
        end
    end
    return index
end

function Base.rand(d::UnivariateFinite)
    p_cumulative = _cumulative(d)
    classes = d.decoder.pool.valindex[collect(keys(d.prob_given_class))]
    return classes[_rand(p_cumulative)]
end

function Base.rand(d::UnivariateFinite, n::Int)
    p_cumulative = _cumulative(d)
    classes = d.decoder.pool.valindex[collect(keys(d.prob_given_class))]
    return [classes[_rand(p_cumulative)] for i in 1:n]
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
                           weights::Nothing) where L
    return Distributions.fit(d, v)
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
