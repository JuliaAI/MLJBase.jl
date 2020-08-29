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
    pairs = (class => cat((pdf.(u, class) for u in us)..., dims=dims)
             for class in _support)
    prob_given_class = Dict(pairs)

    return UnivariateFinite(prob_given_class, pool=_classes)
end

Base.vcat(us::UnivariateFiniteArray...) = cat(us..., dims=1)
Base.hcat(us::UnivariateFiniteArray...) = cat(us..., dims=2)


## CONVENIENCE METHODS pdf(array_of_univariate_finite, labels)
## AND logpdf(array_of_univariate_finite, labels)

# next bit is not specific to `UnivariateFiniteArray` but is for any
# abstract array with eltype `UnivariateFinite`.

# this is type piracy that has been adopted only after much
# agonizing over alternatives. Note that pdf.(u, labels) must
# necessarily have a different meaning (and only makes sense if u and
# labels have the same length or labels is a scalar)

for func in [:pdf, :logpdf]
    eval(quote
        function Distributions.$func(
            u::AbstractArray{UnivariateFinite{S,V,R,P},N},
            C::AbstractVector{<:Union{V, CategoricalValue{V,R}}}) where {S,V,R,P,N}
        
            ret = Array{P,N+1}(undef, size(u)..., length(C))
            for i in eachindex(C)
                ret[fill(:,N)...,i] = broadcast($func, u, C[i])
            end
            return ret
        end
    end)
end


##
## PERFORMANT BROADCASTING OF pdf and logpdf
##

# u - a UnivariateFiniteArray
# cv - a CategoricalValue
# v - a vector of CategoricalArrays

# dummy function
# returns `x[i]` for `Array` inputs `x` 
# For non-Array inputs returns `zero(dtype)`
#This avoids using an if statement 
_getindex(x::Array,i, dtype)=x[i]
_getindex(::Nothing, i, dtype) = zero(dtype)

# pdf.(u, cv)
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    cv::CategoricalValue) where {S,V,R,P,N}

    cv in classes(u) || _err_missing_class(cv)

    return get(u.prob_given_ref, int(cv), zeros(P, size(u)))
end
        
# pdf.(u, v)
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    v::AbstractArray{<:CategoricalValue{V,R},N}) where {S,V,R,P,N}

    length(u) == length(v) ||throw(DimensionMismatch(
        "Arrays could not be broadcast to a common size; "*
        "got a dimension with lengths $(length(u)) and $(length(v))"))
    for cv in v
        cv in classes(u) || _err_missing_class(c)
    end

    # will use linear indexing:
    v_flat = ((v[i], i) for i in 1:length(v))
    
    getter((cv, i), dtype) = _getindex(get(u.prob_given_ref, int(cv), nothing), i, dtype)
    
    ret_flat = getter.(v_flat, P)
    return reshape(ret_flat, size(u))
end

# pdf.(u, raw) where raw is scalar or vec
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UnivariateFiniteArray{S,V,R,P,N},
    raw::Union{V,AbstractArray{V,N}}) where {S,V,R,P,N}

    cat = transform(classes(u), raw)
    return broadcast(pdf, u, cat)
end

# logpdf.(u::UniFinArr{S,V,R,P,N}, cv::CategoricalValue)
# logpdf.(u::UniFinArr{S,V,R,P,N}, v::AbstractArray{<:CategoricalValue{V,R},N})
# logpdf.(u::UniFinArr{S,V,R,P,N}, raw::V)  
# logpdf.(u::UniFinArr{S,V,R,P,N}, raw::AbstractArray{V,N})  
for typ in (:CategoricalValue, 
	    :(AbstractArray{<:CategoricalValue{V,R},N}), 
	    :V,
	    :(AbstractArray{V,N}))
    if typ == :CategoricalValue || typ == :V
    eval(quote 
       function Base.Broadcast.broadcasted(
    		        ::typeof(logpdf),
    	     	        u::UniFinArr{S,V,R,P,N},
    	     		c::$typ) where {S,V,R,P,N}

    	   # Start with the pdf array
    	   pdf_arr = pdf.(u, c)
    	   
    	   # Create an uninitialized array similar to pdf_arr
	   # this avoids mutating the initial pdf_arr  
	   result = similar(pdf_arr)
	   
    	   # Take the log of each entry in-place
    	   @simd for j in eachindex(result)
    	       @inbounds result[j] = log(pdf_arr[j])
    	   end
	
    	   return result
        end
    end)
    else
    	eval(quote
    	function Base.Broadcast.broadcasted(
	    		        ::typeof(logpdf),
 	    	     	        u::UniFinArr{S,V,R,P,N},
	    	     		c::$typ) where {S,V,R,P,N}

    	    # Start with the pdf array
    	    result = pdf.(u, c) 
	    	    
	    # Take the log of each entry in-place
  	    @simd for j in eachindex(result)
  	    	@inbounds result[j] = log(result[j])
	    end
		
 	    return result
    	end
    	end)
    end
end

## PERFORMANT BROADCASTING OF mode:

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
