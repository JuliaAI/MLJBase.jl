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
scale(r::NumericRange{T,B,Symbol}) where {B<:Boundedness,T} = r.scale


## ITERATOR METHOD (FOR GENERATING A 1D GRID)

"""
    iterator(r::NominalRange, [,n, rng])
    iterator(r::NumericRange, n, [, rng])

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

If `r` isa a discrete range (`r isa NumericRange{<:Integer}`) then the
values are rounded, with any duplicate values removed. Otherwise all
the values are used as is (and there are exacltly `n` of them).

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
function iterator(r::NumericRange{T,<:Bounded},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{T,<:LeftUnbounded},
                  n::Int) where {T<:Real}
    L = r.upper - 2r.unit
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{T,<:RightUnbounded},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.lower + 2r.unit
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{T,<:DoublyUnbounded},
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

## FITTING DISTRIBUTIONS TO A RANGE

### Helper

function _truncated(d::Dist.Distribution, r::NumericRange)
    if minimum(d) >= r.lower && maximum(d) <= r.upper
        return d
    else
        return Dist.truncated(d, r.lower, r.upper)
    end
end

### Trait

is_range_fittable(::Type) = false

for D in [
    :Arcsine, :Uniform, :Biweight, :Cosine, :Epanechnikov,
    :SymTriangularDist, :Triweight, :Cauchy, :Gumbel,
    :Normal, :Laplace, :Logistic, :Gamma, :InverseGaussian,
    :LogNormal]
    
    @eval is_range_fittable(::Type{<:Dist.$D}) = true
end

### Fallback and docstring

"""
    Distributions.fit(D, r::MLJBase.NumericRange)

Fit and return a distribution `d` of type `D` to the one-dimensional
range `r`. 

Only types `D` in the table below are supported. 

The distribution `d` is constructed in two stages. First the
attributes `r.origin` and `r.unit` are used to fit a distributon `d0`
characterized by the conditions in the second column of the
table. Then `d0` is truncated between `r.lower` and `r.upper` to
obtain `d`.

Distribution type `D`  | Characterization of `d0`
:----------------------|:-------------------------
`Arcsine`, `Uniform`, `Biweight`, `Cosine`, `Epanechnikov`, `SymTriangularDist`, `Triweight` |   `minimum(d) = r.lower`, `maximum(d) = r.upper`
`Normal`, `Gamma`, `InverseGaussian`, `Logistic`, `LogNormal` | `mean(d) = r.origin`, `std(d) = r.unit`
`Cauchy`, `Gumbel`, `Laplace`, (`Normal`) | `Dist.location(d) = r.origin`, `Dist.scale(d)  = r.unit`

"""
Dist.fit(::Type{D}, r::NumericRange) where D =
    error("Cannot automatically generate an instance of type `$D`. "*
          "Try passing an explicit instance, or a supported type. ")


### Continuous support

#### two-parameters

##### bounded

for D in [:Arcsine, :Uniform]
    @eval Dist.fit(::Type{<:Dist.$D}, r::NumericRange) =
        Dist.$D(r.lower, r.upper)
end
for D in [:Biweight, :Cosine, :Epanechnikov, :SymTriangularDist, :Triweight]
    @eval Dist.fit(::Type{<:Dist.$D}, r::NumericRange) =
        Dist.$D(r.origin, r.unit)
end


##### doubly-unbounded

# corresponding to values of `Dist.location` and `Dist.scale`:
for D in [:Cauchy, :Gumbel, :Normal, :Laplace]
    @eval Dist.fit(::Type{<:Dist.$D}, r::NumericRange) =
        _truncated(Dist.$D(r.origin, r.unit), r)
end

# Logistic:
function Dist.fit(::Type{<:Dist.Logistic}, r::NumericRange)
    μ = r.origin
    θ = sqrt(3)*r.unit/pi
    return _truncated(Dist.Logistic(μ, θ), r)
end


#### right-unbounded

# Gamma:
function Dist.fit(::Type{<:Dist.Gamma}, r::NumericRange)
    α = (r.origin/r.unit)^2
    θ = r.origin/α
    _truncated(Dist.Gamma(α, θ), r)
end

# InverseGaussian:
function Dist.fit(::Type{<:Dist.InverseGaussian}, r::NumericRange)
    mu = r.origin
    lambda = mu^3/r.unit^2
    return _truncated(Dist.InverseGaussian(mu, lambda), r)
end

# LogNormal:
function Dist.fit(::Type{<:Dist.LogNormal}, r::NumericRange)
    sig2 = log((r.unit/r.origin)^2 + 1)
    sig = sqrt(sig2)
    mu = log(r.origin) - sig2/2
    return _truncated(Dist.LogNormal(mu, sig), r)
end


## SAMPLER (FOR RANDOM SAMPLING A 1D RANGE)

"""
    sampler(r::NominalRange, probs::AbstractVector{<:Real})
    sampler(r::NumericRange{T}, d)

Construct an object `s` for generating random samples from a
`ParamRange` object `r` (a one-dimensional range) using one of the
following calls:

    rand(s)             # for one sample
    rand(s, n)          # for n samples
    rand(rng, s [, n])  # to specify an RNG

The argument `probs` can be any probability vector with the same
length as `r.values`.

The argument `d`, can be either an arbitrary instance of
`UnivariateDistribution` from the Distributions.jl package, or one of
the Distributions.jl types specified in the table below. 

If `d` is an *instance*, then sampling is from a truncated form of the
supplied distribution `d`, the truncation bounds being `r.lower` and
`r.upper` (the attributes `r.origin` and `r.unit` attributes are
ignored). For discrete numeric ranges (`T <: Integer`) the samples are
rounded.

If `d` is a *type* then a suitably truncated distribution is
automatically generated using `Distributions.fit(d, r)`.

### Examples

    r = range(Char, :letter, values=collect("abc"))
    s = sampler(r, [0.1, 0.2, 0.7])
    samples =  rand(s, 1000);
    StatsBase.countmap(samples)
    Dict{Char,Int64} with 3 entries:
      'a' => 107
      'b' => 205
      'c' => 688

    r = range(Int, :k, lower=2, upper=6) # numeric but discrete 
    s = sampler(r, Normal)
    samples = rand(s, 1000);
    UnicodePlots.histogram(samples)
               ┌                                        ┐ 
    [2.0, 2.5) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 119                        
    [2.5, 3.0) ┤ 0                                        
    [3.0, 3.5) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 296   
    [3.5, 4.0) ┤ 0                                        
    [4.0, 4.5) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 275     
    [4.5, 5.0) ┤ 0                                        
    [5.0, 5.5) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 221            
    [5.5, 6.0) ┤ 0                                        
    [6.0, 6.5) ┤▇▇▇▇▇▇▇▇▇▇▇ 89                            
               └                                        ┘ 

"""
function sampler end


### Numeric case

struct NumericSampler{T,D<:Distributions.Sampleable} <: MLJType
    distribution::D
    NumericSampler(::Type{T}, d::D) where {T,D} = new{T,D}(d)
end

# constructor for distribution *instances*:
sampler(r::NumericRange{T},
        d::Distributions.UnivariateDistribution) where T =
            NumericSampler(T, _truncated(d, r))

# constructor for *types*:
sampler(r::NumericRange, D::Type{<:Dist.UnivariateDistribution}) =
    sampler(r, Dist.fit(D, r))

# rand fallbacks (non-integer ranges):
Base.rand(s::NumericSampler, dims::Integer...) =
    rand(s.distribution, dims...)
Base.rand(rng::AbstractRNG, s::NumericSampler, dims::Integer...) =
    rand(rng, s.distribution, dims...)

# rand for integer ranges:
Base.rand(s::NumericSampler{I}, dims::Integer...) where I<:Integer =
    map(x -> round(I, x), rand(s.distribution, dims...))
Base.rand(rng::AbstractRNG,
          s::NumericSampler{I},
          dims::Integer...) where I<:Integer =
    map(x -> round(I, x), rand(rng, s.distribution, dims...))


## Nominal case:

struct NominalSampler{T,N,D<:Distributions.Sampleable} <: MLJType
    distribution::D
    values::NTuple{N,T}
    NominalSampler(::Type{T}, d::D, values::NTuple{N,T}) where {T,N,D} =
        new{T,N,D}(d, values)
end

# constructor for probability vectors:
function sampler(r::NominalRange{T},
                 probs::AbstractVector{<:Real}) where T
    length(probs) == length(r.values) ||
        error("Length of probability vector must match number "*
              "of range values. ")
    return NominalSampler(T, Distributions.Categorical(probs), r.values)
end

# constructor for uniform sampling:
function sampler(r::NominalRange{T,N}) where {T, N}
    return sampler(r, fill(1/N, N))
end

Base.rand(s::NominalSampler, dims::Integer...) where I<:Integer =
    broadcast(idx -> s.values[idx], rand(s.distribution, dims...))
Base.rand(rng::AbstractRNG,
          s::NominalSampler,
          dims::Integer...) where I<:Integer =
    broadcast(idx -> s.values[idx], rand(rng, s.distribution, dims...))




#using Plots
#plotly()


    

