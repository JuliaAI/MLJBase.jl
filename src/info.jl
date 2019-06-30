target_quantity(m) =
    target_scitype_union(m) <: Tuple ? true : false
is_probabilistic(::Type{<:Model}) = false
is_probabilistic(::Type{<:Deterministic}) = false
is_probabilistic(::Type{<:Probabilistic}) = true

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

    is_scitype(scitype_X(M)) ||
        error(message*"scitype_X($M) is not a scitype. ")

    is_scitype(scitype_y(M)) ||
        error(message*"scitype_y($M) is not a scitype. ")

    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = Dict{Symbol,Any}()
    d[:is_supervised] = true
    d[:is_probabilistic] = is_probabilistic(M)
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:scitype_X] = scitype_X(M)
    d[:scitype_y] = scitype_y(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    d[:is_wrapper] = is_wrapper(M)
    return d
end

function info(M::Type{<:Unsupervised})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"MLJBase.load_path($M) should be defined so that "* 
                                    "using MLJ; import MLJ.load_path($M) loads $M into current namespace.")

    is_scitype(scitype_X(M)) ||
        error(message*"scitype_X($M) is not a scitype. ")

    is_scitype(scitype_output(M)) ||
        error(message*"scitype_ouput($M) is not a scitype. ")


    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = Dict{Symbol,Any}()
    d[:is_supervised] = false
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:scitype_X] = scitype_X(M)
    d[:scitype_output] = scitype_output(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    d[:is_wrapper] = is_wrapper(M)
\
    return d
end

info(model::Model) = info(typeof(model))
