#
# Constructor utils
#

_err_01() = throw(DomainError("Scores must be in [0,1]."))
_err_sum_1() = throw(DomainError("Score vectors must sum to 1."))

_check_scores_01(s) =
    all(0 .<= s .<= 1) || _err_01()

_check_scores_sum(s::Real) = nothing
_check_scores_sum(s::Arr{<:Real,1}) = nothing
_check_scores_sum(s::Arr{<:Real,2}) =
    all(sum(r) ≈ 1 for r in eachrow(s)) || _err_sum_1()

_check_scores(s) = (_check_scores_01(s); _check_scores_sum(s))

_check_dims(c, exp) = length(c) == exp || throw(DimensionMismatch(
    "Number of classes does not match the dimensions of the scores. " *
    "Expected $exp, got $(length(c))."))

const CVal = CategoricalValue

# R: type of probabilities
# C: number of classes (>=2)
# L: type of the class labels
# E: type of slices of probs, either R (binary) or Vector{R} (multiclass)
struct UnivariateFiniteVector{R,C,L<:CVal,E<:Union{R,Vector{R}}} <: Vec{E}
    scores::Array{R}
    classes::NTuple{C,L}
    function UnivariateFiniteVector(s::Arr{<:Real,N}, c) where N
        N <= 2 || throw(DimensionMismatch("Scores must be a VecOrMat."))
        N == 1 ? _check_dims(c, 2) : _check_dims(c, size(s, 2))
        _check_scores(s)
        R, C, L = eltype(s), length(c), eltype(c)
        # auto categorical if labels passed as simple vector
        if !(L <: CategoricalValue)
            c = categorical(c)
            L = eltype(c)
        end
        E = N == 1 ? R : Vector{R}
        cl = c isa NTuple ? c : tuple(c...)
        new{R,C,L,E}(s, cl)
    end
end

#
# Resolve MLJModelInterface
#

MMI.UnivariateFiniteVector(::FI, a...) = UnivariateFiniteVector(a...)

#
# Convenience functions for auto classes and display
#

const UFV = UnivariateFiniteVector

# Auto classes
function UFV(s::Arr{<:Real,1})
    c = classes(categorical([:negative, :positive])[1])
    return UFV(s, c)
end
function UFV(s::Arr{<:Real,2})
    c = classes(categorical([Symbol("class_$i") for i in 1:size(s, 2)])[1])
    return UFV(s, c)
end

# keep track of the support
UFV(s::Arr{<:Real}, u::UFV) = UFV(s, u.classes)

function Base.show(io::IO, m::MIME"text/plain", u::UFV{R,C}) where {R,C}
    Base.show(io, m, u.scores)
    classes = get.(u.classes)
    type = C == 2 ? "(binary)" : "(multiclass)"
    write(io, """\n
        UnivariateFiniteVector $type
        → length:  $(length(u)),
        → classes: $classes""")
end

#
# Functions for array-like behaviour
#

Base.length(u::UFV) = size(u.scores, 1)
Base.size(u::UFV)   = (size(u.scores, 1),)

function Base.setindex!(u::UFV{R,C}, s, i::Int) where {R,C}
    _check_scores(s)
    if C == 2
        u.scores[i] = s
    else
        u.scores[i,:] = s
    end
end
function Base.setindex!(u::UFV{R,C}, s, I) where {R,C}
    _check_scores(s)
    if C == 2
        u.scores[I] = s
    else
        u.scores[I,:] = s
    end
end

Base.getindex(u::UFV{R,2,L}, I) where {R,L}   = UFV(u.scores[I], u)
Base.getindex(u::UFV{R,C,L}, I) where {R,C,L} = UFV(u.scores[I, :], u)
# cast back to UnivariateFinite // Binary case
function Base.getindex(u::UFV{R,2,L}, i::Int) where {R,L}
    prob_given_class = LittleDict{L,R}(
        u.classes[1] => 1 - u.scores[i],
        u.classes[2] => u.scores[i])
    MMI.UnivariateFinite(prob_given_class)
end
# cast back to UnivariateFinite // Multiclass case
function Base.getindex(u::UFV{R,C,L}, i::Int) where {R,C,L}
    prob_given_class = LittleDict{L,R}(
        u.classes[j] => u.scores[i, j] for j in 1:C)
    MMI.UnivariateFinite(prob_given_class)
end

# Hijack broadcasting for pdf and mode to use more efficient functions

Base.Broadcast.broadcasted(::typeof(mode), u::UFV)   = mode(u)
Base.Broadcast.broadcasted(::typeof(pdf), u::UFV, c) = pdf(u, c)
