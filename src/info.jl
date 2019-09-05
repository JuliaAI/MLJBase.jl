function info(M::Type{<:Supervised})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"MLJBase.load_path($M) should be defined so that "* 
                                    "using MLJ; import MLJ.load_path($M) loads $M into current namespace.")

    is_pure_julia(M) in [true, false, :unknown] ||
        error(message*"is_pure_julia($M) must return true or false. ")

    d = LittleDict{Symbol,Any}()
    d[:name] = name(M)
    d[:docstring] = docstring(M)
    d[:package_name] = package_name(M)
    d[:package_url] = package_url(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_license] = package_license(M)
    d[:load_path] = load_path(M)
    d[:is_wrapper] = is_wrapper(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:supports_weights] = supports_weights(M)
    d[:is_supervised] = true
    d[:prediction_type] = prediction_type(M)
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
    d[:docstring] = docstring(M)
    d[:package_name] = package_name(M)
    d[:package_url] = package_url(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_license] = package_license(M)
    d[:load_path] = load_path(M)
    d[:is_wrapper] = is_wrapper(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:is_supervised] = false
    d[:input_scitype] = input_scitype(M)
    d[:output_scitype] = output_scitype(M)

    return d
end

info(model::Model) = info(typeof(model))
