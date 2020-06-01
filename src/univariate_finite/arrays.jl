const UniFinArr = UnivariateFiniteArray

# TODO: make this faster by making size a field value?
Base.size(u::UniFinArr, args...) =
    size(first(values(u.prob_given_ref)), args...)

function Base.getindex(u::UniFinArr{<:Any,<:Any,R,P,N},
                       i::Integer...) where {R,P,N}
    prob_given_ref = LittleDict{R,P}()
    for ref in keys(u.prob_given_ref)
        prob_given_ref[ref] = getindex(u.prob_given_ref[ref], i...)
    end
    return UnivariateFinite(u.scitype, u.decoder, prob_given_ref)
end

function Base.getindex(u::UniFinArr{<:Any,<:Any,R,P,N},
                       I...) where {R,P,N}
    prob_given_ref = LittleDict{R,Array{P,N}}()
    for ref in keys(u.prob_given_ref)
        prob_given_ref[ref] = getindex(u.prob_given_ref[ref], I...)
    end
    return UnivariateFiniteArray(u.scitype, u.decoder, prob_given_ref)
end

function Base.setindex!(u::UniFinArr{S,V,R,P,N},
                        v::UnivariateFinite{S,V,R,P},
                        i::Integer...) where {S,V,R,P,N}
    for ref in keys(u.prob_given_ref)
       setindex!(u.prob_given_ref[ref], v.prob_given_ref[ref], i...)
    end
    return u
end

# performant broadcasting of pdf:
Base.Broadcast.broadcasted(::typeof(pdf),
                           u::UniFinArr{S,V,R,P,N},
                           cv::CategoricalValue) where {S,V,R,P,N} =
    get(u.prob_given_ref, int(cv), zero(P))

function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UnivariateFiniteArray{S,V,R,P,N},
    c::V) where {S,V,R,P,N}

    _classes = classes(u)
    c in _classes || throw(DomainError("Value $c not in pool. "))
    pool = CategoricalArrays.pool(_classes)
    class = pool[get(pool, c)]
    return broadcast(pdf, u, class)
end

