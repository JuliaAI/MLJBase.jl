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

_err_incompatible_levels() = throw(DomainError(
    "Cannot concatenate `UnivariateFiniteArray`s with "*
    "different categorical levels (classes), "*
    "or whose levels, when ordered, are not  "*
    "consistently ordered. "))

# terminology:

# "classes"  - full pool of `CategoricalElement`s, even "unseen" ones (those
#             missing from support)
# "levels"   - same thing but in raw form (eg, `Symbol`s) aka "labels"
# "suppport" - those classes with a corresponding probability (the ones
#              named at time of construction of the `UnivariateFiniteArray`)

function Base.cat(us::UnivariateFiniteArray{S,V,R,P,N}...;
                      dims::Integer) where {S,V,R,P,N}

    isempty(us) && return []

    # build combined raw_support and check compatibility of levels:
    u1 = first(us)
    ordered = isordered(classes(u1))
    support_with_duplicates = Dist.support(u1)
    _classes = classes(u1)
    for i in 2:length(us)
        isordered(us[i]) == ordered || _err_incompatible_levels()
        if ordered
            classes(us[i]) ==
                _classes|| _err_incompatible_levels()
        else
            Set(classes(us[i])) ==
                Set(_classes) || _err_incompatible_levels()
        end
        support_with_duplicates =
            vcat(support_with_duplicates, Dist.support(us[i]))
    end
    _support = unique(support_with_duplicates) # no-longer categorical!

    # build the combined `prob_given_class` dictionary:
    prob_given_class = LittleDict{V, Array{P,N}}()
    for class in _support
        concatenation = vcat((pdf.(u, class) for u in us)...)
        prob_given_class[class] = concatenation
    end

    return UnivariateFinite(prob_given_class, pool=_classes)
end

Base.vcat(us::UnivariateFiniteArray...) = cat(us..., dims=1)
Base.hcat(us::UnivariateFiniteArray...) = cat(us..., dims=2)

# performant broadcasting of pdf - sample is a cat. value:
Base.Broadcast.broadcasted(::typeof(pdf),
                           u::UniFinArr{S,V,R,P,N},
                           cv::CategoricalValue) where {S,V,R,P,N} =
    get(u.prob_given_ref, int(cv), zeros(P, size(u)))

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
