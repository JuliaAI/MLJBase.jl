# NOTE: see also interface/univariate_finite.jl

const UnivariateFiniteUnion =
    Union{UnivariateFinite, UnivariateFiniteArray}

# NOT EXPORTED!!!!

"""
    classes(d::UnivariateFinite)
    classes(d::UnivariateFiniteArray)

A list of categorial elements in the common pool of classes used to
construct `d`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    classes(d) # CategoricalArray{String,1,UInt32}["maybe", "no", "yes"]

"""
MMI.classes(d::UnivariateFiniteUnion) = d.decoder.classes

"""
    levels(d::UnivariateFinite)

A list of the raw levels in the common pool of classes used to
construct `d`, equal to `get.(classes(d))`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    levels(d) # Array{String, 1}["maybe", "no", "yes"]

"""
levels(d::UnivariateFinite)  = get.(classes(d))

# get the internal integer representations of the support
raw_support(d::UnivariateFiniteUnion) = collect(keys(d.prob_given_ref))

"""
    Distributions.support(d::UnivariateFinite)
    Distributions.support(d::UnivariateFiniteArray)

Ordered list of classes associated with non-zero probabilities.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    support(d) # CategoricalArray{String,1,UInt32}["maybe", "no"]

"""
Distributions.support(d::UnivariateFiniteUnion) =
    map(d.decoder, raw_support(d))

sample_scitype(d::UnivariateFiniteUnion) = d.scitype

CategoricalArrays.isordered(d::UnivariateFinite) = isordered(classes(d))
CategoricalArrays.isordered(u::UnivariateFiniteArray) = isordered(classes(u))


## DISPLAY

function Base.show(stream::IO, d::UnivariateFinite)
    raw = raw_support(d) # reflects order of pool at
                         # instantiation of d
    x1 = d.decoder(first(raw))
    p1 = d.prob_given_ref[first(raw)]
    str = "UnivariateFinite{$(d.scitype)}($x1=>$(round(p1, sigdigits=3))"
    pairs = (d.decoder(r)=>d.prob_given_ref[r] for r in raw[2:end])
    for pair in pairs
        str *= ", $(pair[1])=>$(round(pair[2], sigdigits=3))"
    end
    str *= ")"
    print(stream, str)
end

show_prefix(u::UnivariateFiniteArray{S,V,R,P,1}) where {S,V,R,P} =
    "$(length(u))-element"
show_prefix(u::UnivariateFiniteArray) = join(size(u),'x')

# function show_compact(io::IO, v)
#     K = 4
#     n = length(v)
#     if !isempty(v)
#         indent = div(length(string(v[1])), 2)
#         L = min(n, K - 1)
#         for i in 1:min(n, L - 1)
#             println(io, " ", v[i])
#         end
#         print(io, " ", v[L])
#         if n > K
#             println(io, "\n", fill(" ", indent)..., " ⋮")
#         else
#             println(io)
#         end
#         if n > K - 1
#             println(io, " ", last(v))
#         end
#     end
# end

# function Base.show(io::IO,
#                    u::UnivariateFiniteArray{S,V,R,P,N}) where {S,V,R,P,N}
#     print(io, show_prefix(u), " UnivariateFiniteArray{$S,$V,$R,$P,$N}(...)")
# end

# function Base.show(io::IO, m::MIME"text/plain",
#                    u::UnivariateFiniteArray{S,V,R,P,1}) where {S,V,R,P}
#     support = get.(Dist.support(u))
#     print(io, show_prefix(u), " UnivariateFiniteArray{$S,$V,$R,$P,1}")
#     if !isempty(u)
#         println(io, ":")
#         show_compact(io, u)
#     end
# end


"""
    isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)

Returns `true` if and only if `d1` and `d2` have the same support and
the corresponding probabilities are approximately equal. The key-word
arguments `kwargs` are passed through to each call of `isapprox` on
probability pairs.  Returns `false` otherwise.

"""
function Base.isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)
    support1 = Dist.support(d1)
    support2 = Dist.support(d2)
    for c in support1
        c in support2 || return false
        isapprox(pdf(d1, c), pdf(d2, c); kwargs...) ||
            return false # pdf defined below
    end
    return true
end
function Base.isapprox(d1::UnivariateFiniteArray,
                       d2::UnivariateFiniteArray; kwargs...)
    support1 = Dist.support(d1)
    support2 = Dist.support(d2)
    for c in support1
        c in support2 || return false
        isapprox(pdf.(d1, c), pdf.(d2, c); kwargs...) ||
            return false 
    end
    return true
end

function average(dvec::AbstractVector{UnivariateFinite{S,V,R,P}};
                 weights=nothing) where {S,V,R,P}

    n = length(dvec)

    Dist.@check_args(UnivariateFinite, weights == nothing || n==length(weights))

    # check all distributions have consistent pool:
    first_index = first(dvec).decoder.classes
    for d in dvec
        d.decoder.classes == first_index ||
            error("Averaging UnivariateFinite distributions with incompatible"*
                  " pools. ")
    end

    # get all refs:
    refs = reduce(union, [keys(d.prob_given_ref) for d in dvec]) |> collect

    # initialize the prob dictionary for the distribution sum:
    prob_given_ref = LittleDict{R,P}([refs...], zeros(P, length(refs)))

    # make vector of all the distributions dicts padded to have same common keys:
    prob_given_ref_vec = map(dvec) do d
        merge(prob_given_ref, d.prob_given_ref)
    end

    # sum up:
    if weights == nothing
        scale = 1/n
        for x in refs
            for k in 1:n
                prob_given_ref[x] += scale*prob_given_ref_vec[k][x]
            end
        end
    else
        scale = 1/sum(weights)
        for x in refs
            for k in 1:n
                prob_given_ref[x] +=
                    weights[k]*prob_given_ref_vec[k][x]*scale
            end
        end
    end
    d1 = first(dvec)
    return UnivariateFinite(sample_scitype(d1), d1.decoder, prob_given_ref)
end


function _pdf(d::UnivariateFinite{S,V,R,P}, ref) where {S,V,R,P}
    return get(d.prob_given_ref, ref, zero(P))
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

    mode(d)    # CategoricalValue{String, UInt32} "maybe"
    rand(d, 5) # CategoricalArray{String,1,UInt32}["maybe", "no", "maybe", "maybe", "no"] or similar
    d = fit(UnivariateFinite, v)
    pdf(d, "maybe") # 0.25

One can also do weighted fits:

    w = [1, 4, 5, 1] # some weights
    d = fit(UnivariateFinite, v, w)
    pdf(d, "maybe") ≈ 4/11 # true

See also `classes`, `support`.
"""
Distributions.pdf(d::UnivariateFinite, cv::CategoricalValue) = _pdf(d, int(cv))

function Distributions.pdf(
    d::UnivariateFinite{S,V,R,P}, c::V) where {S,V,R,P}

    _classes = classes(d)
    c in _classes || throw(DomainError("Value $c not in pool. "))
    pool = CategoricalArrays.pool(_classes)
    class = pool[get(pool, c)]
    return pdf(d, class)
end

Distributions.logpdf(d::UnivariateFinite, cv::CategoricalValue) = log(pdf(d,cv))

Distributions.logpdf(d::UnivariateFinite{S,V,R,P}, c::V) where {S,V,R,P} = log(pdf(d,c))


function Distributions.mode(d::UnivariateFinite)
    dic = d.prob_given_ref
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

# mode(v::Vector{UnivariateFinite}) = mode.(v)
# mode(u::UnivariateFiniteVector{2}) =
#     [u.support[ifelse(s > 0.5, 2, 1)] for s in u.scores]
# mode(u::UnivariateFiniteVector{C}) where {C} =
#     [u.support[findmax(s)[2]] for s in eachrow(u.scores)]

"""
    _cumulative(d::UnivariateFinite)

Return the cumulative probability vector `[0, ..., 1]` for the
distribution `d`, using only classes in the support of `d`, ordered
according to the categorical elements used at instantiation of
`d`. Used only to implement random sampling from `d`.

"""
function _cumulative(d::UnivariateFinite{S,V,R,P}) where {S,V,R,P<:Real}

    # the keys of `d` are in order; see constructor
    p = collect(values(d.prob_given_ref))
    K = length(p)
    p_cumulative = Array{P}(undef, K + 1)
    p_cumulative[1] = zero(P)
    p_cumulative[K + 1] = one(P)
    for i in 2:K
        p_cumulative[i] = p_cumulative[i-1] + p[i-1]
    end
    return p_cumulative
end

"""
_rand(rng, p_cumulative, R)

Randomly sample the distribution with discrete support `R(1):R(n)`
which has cumulative probability vector `p_cumulative=[0, ..., 1]` (of
length `n+1`). Does not check the first and last elements of
`p_cumulative` but does not use them either.

"""
function _rand(rng, p_cumulative, R)
    real_sample = rand(rng)
    K = R(length(p_cumulative))
    index = K
    for i in R(2):R(K)
        if real_sample < p_cumulative[i]
            index = i - R(1)
            break
        end
    end
    return index
end

function Base.rand(rng::AbstractRNG,
                   d::UnivariateFinite{<:Any,<:Any,R}) where R
    p_cumulative = _cumulative(d)
    return Distributions.support(d)[_rand(rng, p_cumulative, R)]
end

function Base.rand(rng::AbstractRNG,
                   d::UnivariateFinite{<:Any,<:Any,R},
                   dim1::Int, moredims::Int...) where R # ref type
    p_cumulative = _cumulative(d)
    A = Array{R}(undef, dim1, moredims...)
    for i in eachindex(A)
        @inbounds A[i] = _rand(rng, p_cumulative, R)
    end
    support = Distributions.support(d)
    return broadcast(i -> support[i], A)
end

rng(d::UnivariateFinite, args...) = rng(Random.GLOBAL_RNG, d, args...)

function Distributions.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{C}) where C
    C <: CategoricalValue ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "`CategoricalValue type`. ")
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
                           v::AbstractVector{C},
                           weights::Nothing) where C
    return Distributions.fit(d, v)
end

function Distributions.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{C},
                           weights::AbstractVector{<:Real}) where C
    C <: CategoricalValue ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "`CategoricalValue` type. ")
    y = broadcast(identity, skipmissing(v))
    isempty(y) && error("No non-missing data to fit. ")
    classes_seen = filter(in(unique(y)), classes(y[1]))

    # instantiate and initialize prob dictionary:
    prob_given_class = LittleDict{C,Float64}()
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
