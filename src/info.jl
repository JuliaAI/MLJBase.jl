target_quantity(m) =
    target_scitype_union(m) <: Tuple ? true : false
is_probabilistic(::Type{<:Model}) = false
is_probabilistic(::Type{<:Deterministic}) = false
is_probabilistic(::Type{<:Probabilistic}) = true


# TODO: depreciate? currently used by ensembles.jl
# output_is(modeltype::Type{<:Model}) =
#     [is_probabilistic(modeltype), output_kind(modeltype), target_quantity(modeltype)]
# output_is(model::Model) = output_is(typeof(model))

function coretype(M)
    if isdefined(M, :name)
        return M.name
    else
        return coretype(M.body)
    end
end
    
name(M::Type{<:Model}) = split(string(coretype(M)), '.')[end] |> String

if VERSION < v"1.0.0"
    import Base.info
end

function info(M::Type{<:Supervised})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"MLJBase.load_path($M) should be defined so that "* 
                                    "using MLJ; import MLJ.load_path($M) loads $M into current namespace.")

    # check target_scitype_union:
    T = target_scitype_union(M)
    T <: Union{Found,NTuple{N,Found}} where N ||
        error(message*"target_scitype_union($M) (defining upper bound of target scitype) "*
              "is not a subtype of Found. ")

    input_scitype_union(M) <: Union{Missing, Found} ||
        error(message*"input_scitype_union($M) (defining upper bound of input scitypes) not a subtype of Union{Missing,Found}. ")

    input_is_multivariate(M) in [false, true] ||
        error(message*"input_is_multivariate($M) must return true or false. ")
    
    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = Dict{Symbol,Any}()
    d[:is_supervised] = true
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:target_scitype_union] = target_scitype_union(M)
    d[:input_scitype_union] = input_scitype_union(M)
    d[:input_is_multivariate] = input_is_multivariate(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    d[:is_probabilistic] = is_probabilistic(M)
    d[:is_wrapper] = is_wrapper(M)
    return d
end

function info(M::Type{<:Unsupervised})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"MLJBase.load_path($M) should be defined so that "* 
                                    "using MLJ; import MLJ.load_path($M) loads $M into current namespace.")

    output_scitype_union(M) <: Union{Missing,Found} ||
        error(message*"output_scitype_union($M) (defining upper bound of output scitypes) is not a subtype of Union{Missing,Found}. ")

    output_is_multivariate(M) in [false, true] ||
        error(message*"output_is_multivariate($M) must return true or false.")

    input_scitype_union(M) <: Union{Missing, Found} ||
        error(message*"input_scitype_union($M) (defining upper bound of input scitypes) not a subtype of Union{Missing,Found}. ")

    input_is_multivariate(M) in [false, true] ||
        error(message*"input_is_multivariate($M) must return true or false.")
    
    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = Dict{Symbol,Any}()
    d[:is_supervised] = false
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:output_scitype_union] = output_scitype_union(M)
    d[:output_is_multivariate] = output_is_multivariate(M)
    d[:input_scitype_union] = input_scitype_union(M)
    d[:input_is_multivariate] = input_is_multivariate(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    d[:is_wrapper] = is_wrapper(M)

    return d
end

info(model::Model) = info(typeof(model))
