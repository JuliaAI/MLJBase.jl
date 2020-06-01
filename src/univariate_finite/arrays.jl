const UniFinArr = UnivariateFiniteArray

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

# performant broadcasting of pdf - sample is a cat. value:
Base.Broadcast.broadcasted(::typeof(pdf),
                           u::UniFinArr{S,V,R,P,N},
                           cv::CategoricalValue) where {S,V,R,P,N} =
    get(u.prob_given_ref, int(cv), zero(P))

# performant broadcasting of pdf - sample is a raw label:
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

# performant broadcasting of mode:
function Base.Broadcast.broadcasted(::typeof(mode),
                                    u::UniFinArr{S,V,R,P,N}) where {S,V,R,P,N}
    dic = u.prob_given_ref

    # using linear indexing:
    mode_flat = map(1:length(u)) do i
        max_prob = maximum(dic[ref][i] for ref in keys(dic))
        m = zero(R)
        for ref in keys(dic)
            if dic[ref][i] == max_prob
                m = ref
                break
            end
        end
        return u.decoder(m)
    end
    return reshape(mode_flat, size(u))
end

