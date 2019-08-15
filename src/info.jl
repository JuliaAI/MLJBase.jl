target_quantity(m) =
    target_scitype(m) <: Tuple ? true : false
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

    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = LittleDict{Symbol,Any}()
    d[:name] = name(M)
    d[:package_name] = package_name(M)
    d[:package_url] = package_url(M)
    d[:package_license] = package_license(M)
    d[:load_path] = load_path(M)
    d[:is_wrapper] = is_wrapper(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_uuid] = package_uuid(M)
    d[:supports_sample_weights] = supports_sample_weights(M)
    d[:is_supervised] = true
    d[:is_probabilistic] = is_probabilistic(M)
    d[:input_scitype] = input_scitype(M)
    d[:target_scitype] = target_scitype(M)
    return d
end

function info(M::Type{<:Unsupervised})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"MLJBase.load_path($M) should be defined so that "* 
                                    "using MLJ; import MLJ.load_path($M) loads $M into current namespace.")

    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = LittleDict{Symbol,Any}()
    d[:name] = name(M)
    d[:package_name] = package_name(M)
    d[:package_url] = package_url(M)
    d[:package_uuid] = package_uuid(M)
    d[:load_path] = load_path(M)
    d[:is_wrapper] = is_wrapper(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:is_supervised] = false
    d[:input_scitype] = input_scitype(M)
    d[:output_scitype] = output_scitype(M)

    return d
end

info(model::Model) = info(typeof(model))
