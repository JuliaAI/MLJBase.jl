"""
docstring_ext

Helper function to generate the docstring for a package.
"""
function docstring_ext(T; descr::String="")
    package_name = MLJBase.package_name(T)
    package_url  = MLJBase.package_url(T)
    model_name   = MLJBase.name(T)
    # the message to return
    message      = "$descr"
    message     *= "\n→ based on [$package_name]($package_url)"
    message     *= "\n→ do `@load $model_name` to use the model"
    message     *= "\n→ do `?$model_name` for documentation."
end

"""
metadata_pkg

Helper function to write the metadata for a package.
"""
function metadata_pkg(T; name::String="unknown", uuid::String="unknown", url::String="unknown",
                         julia::Union{Missing,Bool}=missing, license::String="unknown",
                         is_wrapper::Bool=false)
    ex = quote
        package_name(::Type{<:$T})    = $name
        package_uuid(::Type{<:$T})    = $uuid
        package_url(::Type{<:$T})     = $url
        is_pure_julia(::Type{<:$T})   = $julia
        package_license(::Type{<:$T}) = $license
        is_wrapper(::Type{<:$T})      = $is_wrapper
    end
    eval(ex)
end

"""
metadata_model

Helper function to write the metadata for a single model of a package (complements
[`metadata_ext`](@ref)).
"""
function metadata_model(T; input=Unknown, target=Unknown,
                         output=Unknown, weights::Bool=false,
                         descr::String="", path::String="")
    if isempty(path)
        path = "MLJModels.$(package_name(T))_.$(name(T))"
    end
    ex = quote
        input_scitype(::Type{<:$T})    = $input
        output_scitype(::Type{<:$T})   = $output
        target_scitype(::Type{<:$T})   = $target
        supports_weights(::Type{<:$T}) = $weights
        docstring(::Type{<:$T})        = docstring_ext($T, descr=$descr)
        load_path(::Type{<:$T})        = $path
    end
    eval(ex)
end
