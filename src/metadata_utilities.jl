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
    message     *= "\n→ based on [$package_name]($package_url)."
    message     *= "\n→ do `@load $model_name pkg=\"$package_name\"` to use the model."
    message     *= "\n→ do `?$model_name` for documentation."
end

"""
metadata_pkg

Helper function to write the metadata for a package.
"""
function metadata_pkg(T; name::String="unknown", uuid::String="unknown",
                      url::String="unknown", julia::Union{Missing,Bool}=missing,
                      license::String="unknown", is_wrapper::Bool=false)
    ex = quote
        MLJBase.package_name(::Type{<:$T}) = $name
        MLJBase.package_uuid(::Type{<:$T}) = $uuid
        MLJBase.package_url(::Type{<:$T}) = $url
        MLJBase.is_pure_julia(::Type{<:$T}) = $julia
        MLJBase.package_license(::Type{<:$T}) = $license
        MLJBase.is_wrapper(::Type{<:$T}) = $is_wrapper
    end
    parentmodule(T).eval(ex)
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
        path = "MLJModels.$(MLJBase.package_name(T))_.$(MLJBase.name(T))"
    end
    ex = quote
        MLJBase.input_scitype(::Type{<:$T}) = $input
        MLJBase.output_scitype(::Type{<:$T}) = $output
        MLJBase.target_scitype(::Type{<:$T}) = $target
        MLJBase.supports_weights(::Type{<:$T}) = $weights
        MLJBase.docstring(::Type{<:$T}) = MLJBase.docstring_ext($T; descr=$descr)
        MLJBase.load_path(::Type{<:$T}) = $path
    end
    parentmodule(T).eval(ex)
end


"""
metadata_measure

Helper function to write the metadata for a single measure.
"""
function metadata_measure(T; name::String="",
                          target_scitype=Unknown,
                          prediction_type::Symbol=:unknown,
                          orientation::Symbol=:unknown,
                          reports_each_observation::Bool=true,
                          aggregation=Mean(),
                          is_feature_dependent::Bool=false,
                          supports_weights::Bool=false,
                          docstring::String="")
    pred_str = "$prediction_type"
    orientation_str = "$orientation"
    ex = quote
        isempty($name) || (MLJBase.name(::Type{<:$T}) = $name)
        isempty($docstring) || (MLJBase.docstring(::Type{<:$T}) = $docstring)
        MLJBase.target_scitype(::Type{<:$T}) = $target_scitype
        MLJBase.prediction_type(::Type{<:$T}) = Symbol($pred_str)
        MLJBase.orientation(::Type{<:$T}) = Symbol($orientation_str)
        MLJBase.reports_each_observation(::Type{<:$T}) = $reports_each_observation
        MLJBase.aggregation(::Type{<:$T}) = $aggregation
        MLJBase.is_feature_dependent(::Type{<:$T}) = $is_feature_dependent
        MLJBase.supports_weights(::Type{<:$T}) = $supports_weights
    end
    parentmodule(T).eval(ex)
end
