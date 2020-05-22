# R: type of probabilities
# C: number of classes (>=2)
# L: type of the class labels
# E: type of slices of probs, either R (binary) or Vector{R} (multiclass)
struct UnivariateFiniteVector{R,C,L<:CategoricalValue,E<:Union{R,Vector{R}}} <: Vec{E}
    scores::Array{R}
    classes::NTuple{C,L}
    # Simple constructor for binary case (vector of scores)
    function UnivariateFiniteVector(s::Arr{<:Real,1}, c)
        length(c) == 2 ||
            throw(DimensionMismatch("Number of classes don't match."))
        all(0 .<= s .<= 1) ||
            throw(DomainError("Scores must be between [0,1]."))
        R = eltype(s)
        C = length(c)
        L = eltype(c)
        if !(L <: CategoricalValue)
            c = categorical(c)
            L = eltype(c)
        end
        E = R
        cl = c isa NTuple ? c : tuple(c...)
        new{R,C,L,E}(s, cl)
    end
    # Simple constructor for multi-class case (array of scores)
    function UnivariateFiniteVector(s::Arr{<:Real,2}, c)
        size(s, 2) == length(c) ||
            throw(DimensionMismatch("Number of classes don't match."))
        all(sum(s[i,:]) .≈ 1. for i in size(s, 1)) ||
            throw(DomainError("Scores must sum to one."))
        R = eltype(s)
        C = length(c)
        L = eltype(c)
        if !(L <: CategoricalValue)
            c = categorical(c)
            L = eltype(c)
        end
        E = Vector{R}
        cl = c isa NTuple ? c : tuple(c...)
        new{R,C,L,E}(s, cl)
    end
end

# Used only here

const UFV = UnivariateFiniteVector

# Auto classes
function UFV(s::Arr{<:Real,1})
    c = classes(categorical([:class_0, :class_1])[1])
    return UFV(s, c)
end
function UFV(s::Arr{<:Real,2})
    c = classes(categorical([Symbol("class_$i") for i in 1:size(s, 2)])[1])
    return UFV(s, c)
end

# keep track of the support (define both explicitly to avoid ambiguities)
UFV(s::Arr{<:Real,1}, u::UFV) = UFV(s, u.classes)
UFV(s::Arr{<:Real,2}, u::UFV) = UFV(s, u.classes)


MMI.UnivariateFiniteVector(::FI, a...) = UnivariateFiniteVector(a...)


function Base.show(io::IO, ::MIME"text/plain", u::UFV{R,C}) where {R,C}
    write(io, "UnivariateFiniteVector: \n")
    header = [String(Symbol(e)) for e in u.classes]
    table = u.scores
    if C === 2
        table = hcat(1 .- table, table)
    end
    if R <: AbstractFloat
        PrettyTables.pretty_table(io, round.(table, digits=2), header;
                                  row_names=1:size(table, 1))
    else
        PrettyTables.pretty_table(io, table, header;
                                  row_names=1:size(table, 1))
    end
end

Base.length(u::UFV) = size(u.scores, 1)
Base.size(u::UFV) = (size(u.scores, 1),)

Base.setindex!(u::UFV{R,2}, val, i::Int) where {R} = u.scores[i] = val
Base.setindex!(u::UFV{R,C}, val, i::Int) where {R,C} = u.scores[i,:] = val
Base.setindex!(u::UFV{R,C}, val, I) where {R,C} = setindex!(u.scores, val, I)

Base.getindex(u::UFV{R,C,L}, I) where {R,C,L} = UFV(getindex(u.scores, I), u)
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
