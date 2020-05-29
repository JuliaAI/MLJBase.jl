#
# Constructor utils
#

_err_01() = throw(DomainError("Probabilities must be in [0,1]."))
_err_sum_1() = throw(DomainError("Probablity arrays must sum to one "*
                                 "along the first axis. "))

_check_probs_01(probs) =
    all(0 .<= probs .<= 1) || _err_01()

_check_probs_sum(probs::Arr{P}) where P<:Real =
    all(x -> x ≈ one(P), sum(probs, dims=1)) || _err_sum_1()

_check_probs(probs) = (_check_probs_01(probs); _check_probs_sum(probs))

_check_dims(support, probs) = size(probs, 1) == length(support) ||
    throw(DimensionMismatch(
        "Encountered a support of size  $(length(support)), "*
        "which does not match the first dimension of specified "*
        "probability array, "*
        "$(size(probs, 1)). "))

_V_and_R(::CategoricalArray{<:Any,<:Any,R,V}) where {V,R} = (V, R)

# pre-process user-supplied support object:
_categorical(c, ordered) = ArgumentError("Inadmissible support. ")
_categorical(c::Tuple; ordered) = _categorical(c)
_categorical(c::Array, ordered) = categorical(c, ordered=ordered, compress=true)
_categorical(c::CategoricalArray, ordered) = c

# notation consistent with
# https://juliadata.github.io/CategoricalArrays.jl/stable/implementation.html
# and UnivariateFinite code:

# S - size of support (>=2)
# R - reference type <: Unsigned
# V - type of class labels (eg, Char in `categorical(['a', 'b'])`)
# P - type of probabilities
# N - dimension of array

# for probability arrays the *first* dimension corresponds to the
# support, ie, size(probs, 1) = length(support) > 1.

struct UnivariateFiniteArray{V,R,P<:Real,OneMoreThanN,N} <:
    AbstractArray{UnivariateFinite{V,R,P},N}
    support::CategoricalVector{V, R, V, CategoricalValue{V,R}, Union{}}
    probs::Array{P,OneMoreThanN}
    function UnivariateFiniteArray(raw_support, probs::AbstractArray{P,N};
                                   ordered=false) where {P<:Real,N}
        OneMoreThanN = N + 1
        _check_dims(raw_support, probs)
        _check_probs(s)
        support = _categorical(raw_support, ordered) # categorizes if necessary
        V, R = _V_and_R(support)
        new{V,R,P,OneMoreThanN,N}(support, s)
    end
end

# convenience shortcut for here (only)
const UFA = UnivariateFiniteArray
const UnivariateFiniteVector{V,R,P} = UnivariateFiniteArray{V,R,P,1}
const UnivariateFiniteMatrix{V,R,P} = UnivariateFiniteArray{V,R,P,2}

#
# Resolve MLJModelInterface
#

MMI.UnivariateFiniteArray(::FI, a...; kw...) = UFA(a...; kw...)

#
# Convenience functions for auto support and display
#

# Auto support - *vector* of probablities (degenerate case):
function UFA(probs::Arr{<:Real,1}; ordered=false)
    cat = categorical([:negative, :positive],
                      ordered=ordered, compress=true)
    support = classes(cat)
    return UFA(support, hcat(probs, 1 .- probs))
end

# Auto support - non-degenerate probability array:
function UFA(probs::Arr{<:Real}; ordered=false)
    cat = categorical([Symbol("class_$i") for i in 1:size(probs, 2)],
                      ordered=ordered, compress=true)
    support = classes(cat)
    return UFA(support, probs)
end

function Base.show(io::IO, u::UFA{V,R,P,OneMoreThanN,N}) where
    {V,R,P,OneMoreThanN,N}
    print(io, "UnivariateFiniteVector{$V,$R,$P,$OneMoreThanN, $N} "*
          "of size $(size(u))")
end

function Base.show(io::IO, m::MIME"text/plain", u::UFA)
    support = get.(u.support)
    println(io, u)
    println(first(u), "\n...", last(u))
    print(io, """\n
        element samples: $(_scitype(first(u)))
        support: $support""")
end

#
# Functions for array-like behaviour
#

Base.size(u::UFA)   = size(u.probs)[2:end]
# Base.length(u::UFA) = prod(size(u))

Base.getindex(u::UFA, i::Integer) = UnivariateFinite(u.support, u.probs[i,:])

# Base.getindex(u::UFA{S,C}, I) where {S,C} = UFA(u, u.probs[I, :])
# # cast back to UnivariateFinite // Binary case
# function Base.getindex(u::UFA{2,C,P}, i::Int) where {C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[1] => 1 - u.probs[i],
#         u.support[2] => u.probs[i])
#     MMI.UnivariateFinite(prob_given_class)
# end
# # cast back to UnivariateFinite // Multiclass case
# function Base.getindex(u::UFA{S,C,P}, i::Int) where {S,C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[j] => u.probs[i, j] for j in 1:S)
#     MMI.UnivariateFinite(prob_given_class)
# end


# function Base.setindex!(u::UFA{S}, s, i::Int) where {S}
#     _check_probs(s)
#     if S == 2
#         u.probs[i] = s
#     else
#         u.probs[i,:] = s
#     end
# end
# function Base.setindex!(u::UFA{S}, s, I) where {S}
#     _check_probs(s)
#     if S == 2
#         u.probs[I] = s
#     else
#         u.probs[I,:] = s
#     end
# end

# constructor keeping track with similar support than parent object
UFA(u::UFA, s::Arr{<:Real}) = UFA(u.support, s)

# Base.getindex(u::UFA{2,C}, I) where {C}   = UFA(u, u.probs[I])
# Base.getindex(u::UFA{S,C}, I) where {S,C} = UFA(u, u.probs[I, :])
# # cast back to UnivariateFinite // Binary case
# function Base.getindex(u::UFA{2,C,P}, i::Int) where {C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[1] => 1 - u.probs[i],
#         u.support[2] => u.probs[i])
#     MMI.UnivariateFinite(prob_given_class)
# end
# # cast back to UnivariateFinite // Multiclass case
# function Base.getindex(u::UFA{S,C,P}, i::Int) where {S,C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[j] => u.probs[i, j] for j in 1:S)
#     MMI.UnivariateFinite(prob_given_class)
# end

# Hijack broadcasting for pdf and mode to use more efficient functions

Base.Broadcast.broadcasted(::typeof(mode), u::UFA)   = mode(u)
Base.Broadcast.broadcasted(::typeof(pdf), u::UFA, c) = pdf(u, c)
