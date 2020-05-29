# function UFA(probs::Arr{<:Real,1}; ordered=false)
#     cat = categorical([:negative, :positive],
#                       ordered=ordered, compress=true)
#     support = classes(cat)
#     return UFA(support, hcat(probs, 1 .- probs))
# end

# # Auto support - non-degenerate probability array:
# function UFA(probs::Arr{<:Real}; ordered=false)
#     cat = categorical([Symbol("class_$i") for i in 1:size(probs, 2)],
#                       ordered=ordered, compress=true)
#     support = classes(cat)
#     return UFA(support, probs)
# end

const UniFinArr = UnivariateFiniteArray

# TODO: make this faster by making size a field value?
Base.size(u::UniFinArr, args...) =
    size(first(values(u.prob_given_ref)), args...)
# Base.length(u::UniFinArr) = prod(size(u)) # needed? maybe there's a fallback
# Base.isempty(u::UniFinArr) = length(u) == 0
# Base.lastindex(u::UniFinArr) = length(u)

# Base.isiterable(::UniFinArr) = true
# function Base.iterate(u::UniFinArr)
#     isempty(u) && return nothing
#     return (u[1], 1)
# end
# function Base.iterate(u::UniFinArr, state)
#     state == length(u) && return nothing
#     return (u[state], state + 1)
# end
# Base.IteratorSize(::UnivFinArr{S,V,R,P,N}) where {S,V,R,P,N} = HasShape{N}()
# Base.eltype(::UnivFinArr{S,V,R,P,N}) where {S,V,R,P,N} =
#     UnivariateFinite{S,V,R,P}

function Base.getindex(u::UniFinArr{<:Any,<:Any,R,P,N},
                       idxs::Integer...) where {R,P,N}
    prob_given_ref = LittleDict{R,P}()
    for ref in keys(u.prob_given_ref)
        prob_given_ref[ref] = getindex(u.prob_given_ref[ref], idxs...)
    end
    return UnivariateFinite(u.scitype, u.decoder, prob_given_ref)
end

# Base.getindex(u::UniFinArr{S,C}, I) where {S,C} = UniFinArr(u, u.probs[I, :])
# # cast back to UnivariateFinite // Binary case
# function Base.getindex(u::UniFinArr{2,C,P}, i::Int) where {C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[1] => 1 - u.probs[i],
#         u.support[2] => u.probs[i])
#     MMI.UnivariateFinite(prob_given_class)
# end
# # cast back to UnivariateFinite // Multiclass case
# function Base.getindex(u::UniFinArr{S,C,P}, i::Int) where {S,C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[j] => u.probs[i, j] for j in 1:S)
#     MMI.UnivariateFinite(prob_given_class)
# end


# function Base.setindex!(u::UniFinArr{S}, s, i::Int) where {S}
#     _check_probs(s)
#     if S == 2
#         u.probs[i] = s
#     else
#         u.probs[i,:] = s
#     end
# end
# function Base.setindex!(u::UniFinArr{S}, s, I) where {S}
#     _check_probs(s)
#     if S == 2
#         u.probs[I] = s
#     else
#         u.probs[I,:] = s
#     end
# end

# constructor keeping track with similar support than parent object
# UniFinArr(u::UniFinArr, s::Arr{<:Real}) = UniFinArr(u.support, s)

# Base.getindex(u::UniFinArr{2,C}, I) where {C}   = UniFinArr(u, u.probs[I])
# Base.getindex(u::UniFinArr{S,C}, I) where {S,C} = UniFinArr(u, u.probs[I, :])
# # cast back to UnivariateFinite // Binary case
# function Base.getindex(u::UniFinArr{2,C,P}, i::Int) where {C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[1] => 1 - u.probs[i],
#         u.support[2] => u.probs[i])
#     MMI.UnivariateFinite(prob_given_class)
# end
# # cast back to UnivariateFinite // Multiclass case
# function Base.getindex(u::UniFinArr{S,C,P}, i::Int) where {S,C,P}
#     prob_given_class = LittleDict{C,P}(
#         u.support[j] => u.probs[i, j] for j in 1:S)
#     MMI.UnivariateFinite(prob_given_class)
# end

# Hijack broadcasting for pdf and mode to use more efficient functions

# Base.Broadcast.broadcasted(::typeof(mode), u::UniFinArr)   = mode(u)
# Base.Broadcast.broadcasted(::typeof(pdf), u::UniFinArr, c) = pdf(u, c)
