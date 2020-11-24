const DOC_OBSERVATIONS =
    "on observations `ŷ`, "*
    "given ground truth values, `y`. "
const DOC_WEIGHTS =
    "Optionally specify per-sample weights, `w`. "
const DOC_CLASS_WEIGHTS =
    "An optional `AbstractDict`, denoted `class_w` above, "*
    "keyed on `levels(y)`, specifies class weights. "

macro create_aliases(ex)
    esc(quote
        M = $ex
        for alias in Symbol.(instances($ex))
            eval(:(const $alias = $M()))
        end
        end)
end

function detailed_doc_string(M; leader= "", body="", footer="")
    _instances = _decorate(instances(M))
    if isempty(leader)
        if isempty(fieldnames(M))
            leader = "Evaluate the $(informal_name(M)) "
        else
            leader = "Evaluate the default instance of the measure "
        end
        leader *= "$DOC_OBSERVATIONS"
    end
    ret =  "    $M\n\n"
    ret *= "A measure type"
    isempty(_instances) ||
        (ret  *= ", which includes the instance(s) $_instances")
    ret *= ".\n\n"
    ret *= "    $(name(M))()(ŷ, y)\n"
    supports_weights(M) &&
        (ret *= "    $(name(M))()(ŷ, y, w)\n")
    supports_class_weights(M) &&
        (ret *= "    $(name(M))()(ŷ, y, class_w)\n")
    ret *= "\n$leader"
    supports_weights(M) &&
        (ret *= DOC_WEIGHTS)
    supports_class_weights(M) &&
        (ret *= DOC_CLASS_WEIGHTS)
    ret *= "\n\n"
    isempty(body) || (ret *= "$body\n\n")
    ret *= "For more information, run `info($(name(M)))`. "
    isempty(footer) ||
        (ret *= "\n\n$footer")
    return ret
end


# TODO: I wonder why this is not a macro?

"""
    metadata_measure(T; kw...)

Helper function to write the metadata for a single measure.
"""
function metadata_measure(T; name::String="",
                          instances::Vector{String}=String[],
                          target_scitype=Unknown,
                          prediction_type::Symbol=:unknown,
                          orientation::Symbol=:unknown,
                          reports_each_observation::Bool=true,
                          aggregation=Mean(),
                          is_feature_dependent::Bool=false,
                          supports_weights::Bool=false,
                          supports_class_weights::Bool=false,
                          docstring::String="",
                          distribution_type=missing)
    pred_str        = "$prediction_type"
    orientation_str = "$orientation"
    dist = ifelse(ismissing(distribution_type), missing, "$distribution_type")
    ex = quote
        if !isempty($name)
            MMI.name(::Type{<:$T}) = $name
        end
        if !isempty($instances)
            instances(::Type{<:$T}) = $instances
        end
        if !isempty($docstring)
            MMI.docstring(::Type{<:$T}) = $docstring
        end
        # traits common with models
        MMI.target_scitype(::Type{<:$T}) = $target_scitype
        MMI.prediction_type(::Type{<:$T}) = Symbol($pred_str)
        MMI.supports_weights(::Type{<:$T}) = $supports_weights
        # traits specific to measures
        orientation(::Type{<:$T}) = Symbol($orientation_str)
        reports_each_observation(::Type{<:$T}) = $reports_each_observation
        aggregation(::Type{<:$T}) = $aggregation
        is_feature_dependent(::Type{<:$T}) = $is_feature_dependent
        supports_class_weights(::Type{<:$T}) = $supports_class_weights
        distribution_type(::Type{<:$T}) = $dist
    end
    parentmodule(T).eval(ex)
end
