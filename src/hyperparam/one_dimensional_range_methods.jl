## SCALE TRANSFORMATIONS

#     Scale = SCALE()

# Object for dispatching on scales and functions when generating
# parameter ranges. We require different behaviour for scales and
# functions:

#      transform(Scale, scale(:log10), 100) = 2
#      inverse_transform(Scale, scale(:log10), 2) = 100

# but
#     transform(Scale, scale(log10), 100) = 100       # identity
#     inverse_transform(Scale, scale(log10), 100) = 2

struct SCALE end
Scale = SCALE()

scale(s::Symbol)   = Val(s)
scale(f::Function) = f

transform(::SCALE, ::Val{:linear}, x) = x
inverse_transform(::SCALE, ::Val{:linear}, x) = x

transform(::SCALE, ::Val{:log}, x) = log(x)
inverse_transform(::SCALE, ::Val{:log}, x) = exp(x)

transform(::SCALE, ::Val{:logminus}, x) = log(-x)
inverse_transform(::SCALE, ::Val{:logminus}, x) = -exp(x)

transform(::SCALE, ::Val{:log10}, x) = log10(x)
inverse_transform(::SCALE, ::Val{:log10}, x) = 10^x

transform(::SCALE, ::Val{:log2}, x) = log2(x)
inverse_transform(::SCALE, ::Val{:log2}, x) = 2^x

transform(::SCALE, f::Function, x) = x            # not a typo!
inverse_transform(::SCALE, f::Function, x) = f(x) # not a typo!

## SCALE INSPECTION (FOR EG PLOTTING)

"""
    scale(r::ParamRange)

Return the scale associated with a `ParamRange` object `r`. The possible
return values are: `:none` (for a `NominalRange`), `:linear`, `:log`, `:log10`,
`:log2`, or `:custom` (if `r.scale` is a callable object).
"""
scale(r::NominalRange) = :none
scale(r::NumericRange) = :custom
scale(r::NumericRange{B,T,Symbol}) where {B<:Boundedness,T} = r.scale


## ITERATOR METHOD (FOR GENERATING A 1D GRID)

"""
    MLJTuning.iterator(r::NominalRange, [,n, rng])
    MLJTuning.iterator(r::NumericRange, n, [, rng])

Return an iterator (currently a vector) for a `ParamRange` object `r`.
In the first case iteration is over all `values` stored in the range
(or just the first `n`, if `n` is specified). In the second case, the iteration
is over approximately `n` ordered values, generated as follows:

First, exactly `n` values are generated between `U` and `L`, with a
uniform spacing determined by `r.scale`, where `U` and `L` are given
by the following table:

| `r.lower`   | `r.upper`  | `L`                 | `U`                 |
|-------------|------------|---------------------|---------------------|
| finite      | finite     | `r.lower`           | `r.upper`           |
| `-Inf`      | finite     | `r.upper - 2r.unit` | `r.upper`           |
| finite      | `Inf`      | `r.lower`           | `r.lower + 2r.unit  |
| `-Inf`      | `Inf`      | `r.origin - r.unit` | `r.origin + r.unit  |

If `r` isa a discrete range (`r isa
NumericRange{<:Any,<:Any,<:Integer}`) then the values are rounded,
with any duplicate values removed. Otherwise all the values are used
as is (and there are exacltly `n` of them).

If a random number generator `rng` is specified, then the values are
returned in random order (sampling without replacement), and otherwise
they are returned in numeric order, or in the order provided to the
range constructor, in the case of a `NominalRange`.
"""
iterator(r::ParamRange, n::Integer, rng::AbstractRNG) =
    StatsBase.sample(rng, iterator(r, n), n, replace=false)

iterator(r::NominalRange, n::Integer) =
    collect(r.values[1:min(n, length(r.values))])
iterator(r::NominalRange) = collect(r.values)
iterator(r::NominalRange, rng::AbstractRNG) =
    iterator(r, length(r.values), rng)

# nominal range, top level dispatch
function iterator(r::NumericRange{<:Bounded,T},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:LeftUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.upper - 2r.unit
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:RightUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.lower + 2r.unit
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:DoublyUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.origin - r.unit
    U = r.origin + r.unit
    return iterator(T, L, U, r.scale, n)
end

# middle level

iterator(::Type{<:Real}, L, U, s, n) =
    iterator(L, U, s, n)

function iterator(I::Type{<:Integer}, L, U, s, n)
    raw = iterator(L, U, s, n)
    rounded = map(x -> round(I, x), raw)
    return unique(rounded)
end

# ground level

function iterator(L, U, s, n)
    transformed = range(transform(Scale, scale(s), L),
                stop=transform(Scale, scale(s), U),
                        length=n)
    inverse_transformed = map(transformed) do value
        inverse_transform(Scale, scale(s), value)
    end
    return inverse_transformed
end


## SAMPLER (FOR RANDOM SAMPLING A 1D RANGE)

struct NumericSampler{T,D<:Distributions.Sampleable} <: MLJType
    distribution::D
    NumericSampler(::Type{T}, d::D) where {T,D} = new{T,D}(d)
end

### Numeric case

struct NumericSampler{T,D<:Distributions.Sampleable} <: MLJType
    distribution::D
    NumericSampler(::Type{T}, d::D) where {T,D} = new{T,D}(d)
end

sampler(r::NumericRange{<:Any,T},
        d::Distributions.UnivariateDistribution) where T =
            NumericSampler(T, Distributions.truncated(d, r.lower, r.upper))

# fallback:
Base.rand(rng::AbstractRNG, s::NumericSampler, dims::Integer...) =
    rand(rng, s.distribution, dims...)

# round for integer ranges:
Base.rand(s::Sampler{I}, dims::Integer...) where I<:Integer =
    map(x -> round(I, x), rand(s.distribution, dims...))
Base.rand(rng::AbstractRNG, s::Sampler{I}, dims::Integer...) where I<:Integer =
    map(x -> round(I, x), rand(rng, s.distribution, dims...))

### Nominal case:

struct NominalSampler{T,D<:Distributions.Sampleable} <: MLJType
    distribution::D
    values::Vector{T}
    NumericSampler(::Type{T}, d::D, values) where {T,D} = new{T,D}(d, values)
end

sampler(r::NominalRange{T},
        probs::AbstractVector{<:Real}) where T =
            NominalSampler(T, D.Categorical(probs), r.values)

Base.rand(s::NominalSampler, dims::Integer...) where I<:Integer =
    broadcast(idx -> s.values[idx], rand(s.distribution, dims...))


