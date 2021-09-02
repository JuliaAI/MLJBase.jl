const DOC_OBSERVATIONS =
    "on predictions `ŷ`, "*
    "given ground truth observations `y`. "
const DOC_WEIGHTS =
    "Optionally specify per-sample weights, `w`. "
const DOC_CLASS_WEIGHTS =
    "An optional `AbstractDict`, denoted `class_w` above, "*
    "keyed on `levels(y)`, specifies class weights. "

macro create_aliases(M_ex)
    esc(quote
        M = $M_ex
        for alias in Symbol.(instances(M))
        # isdefined(parentmodule(M), alias) || eval(:(const $alias = $M()))
        eval(:(const $alias = $M()))
        end
        end)
end

function detailed_doc_string(M; typename="", body="", footer="", scitype="")

    _instances = _decorate(instances(M))
    human_name = MLJBase.human_name(M)
    if isempty(scitype)
        scitype = target_scitype(M) |> string
    end

    if isempty(typename)
        ret = "    $M\n\n"
    else
        ret = "    MLJBase.$typename\n\n"
    end

    ret *= "A measure type for $(human_name)"
    isempty(_instances) ||
        (ret  *= ", which includes the instance(s): "*
         "$_instances")
    ret *= ".\n\n"
    ret *= "    $(name(M))()(ŷ, y)\n"
    supports_weights(M) &&
        (ret *= "    $(name(M))()(ŷ, y, w)\n")
    supports_class_weights(M) &&
        (ret *= "    $(name(M))()(ŷ, y, class_w)\n")
    ret *= "\n"
    if isempty(fieldnames(M))
            ret *= "Evaluate the $(human_name) "
    else
        ret *= "Evaluate the default instance of $(name(M)) "
    end
    ret *= "$DOC_OBSERVATIONS"
    supports_weights(M) &&
        (ret *= DOC_WEIGHTS)
    supports_class_weights(M) &&
        (ret *= DOC_CLASS_WEIGHTS)
    ret *= "\n\n"
    isempty(body) || (ret *= "$body\n\n")
    ret *= "Requires `scitype(y)` to be a subtype of $scitype; "
    ret *= "`ŷ` must be an array of $(prediction_type(M)) predictions. "
    isempty(footer) ||(ret *= "\n\n$footer")
    ret *= "\n\n"
    ret *= "For more information, run `info($(name(M)))`. "
    return ret
end


_err_create_docs() = error(
    "@create_docs syntax error. Usage: \n"*
    "@create_docs(MeasureType, typename=..., body=..., scitype=..., footer=...")
macro create_docs(M_ex, exs...)
    M_ex isa Symbol || _err_create_docs()
    t = ""
    b = ""
    s = ""
    f = ""
    for ex in exs
        ex.head == :(=) || _err_create_docs()
        ex.args[1] == :typename && (t = ex.args[2])
        ex.args[1] == :body &&     (b = ex.args[2])
        ex.args[1] == :scitype &&  (s = ex.args[2])
        ex.args[1] == :footer &&   (f = ex.args[2])
    end
    esc(quote
        "$(detailed_doc_string($M_ex, typename=$t, body=$b, scitype=$s, footer=$f))"
        function $M_ex end
        end)
end

# TODO: I wonder why this is not a macro?

"""
    metadata_measure(T; kw...)

Helper function to write the metadata for a single measure.
"""
function metadata_measure(T; name::String="",
                          human_name="",
                          instances::Vector{String}=String[],
                          target_scitype=Unknown,
                          prediction_type::Symbol=:unknown,
                          orientation::Symbol=:unknown,
                          aggregation=Mean(),
                          is_feature_dependent::Bool=false,
                          supports_weights::Bool=true,
                          supports_class_weights::Bool=false,
                          docstring::String="",
                          distribution_type=Unknown)
    pred_str        = "$prediction_type"
    orientation_str = "$orientation"
#    dist = ifelse(ismissing(distribution_type), missing, "$distribution_type")
    ex = quote

        # traits common with models:
        if !isempty($name)
            StatisticalTraits.name(::Type{<:$T}) = $name
        end
        if !isempty($docstring)
            StatisticalTraits.docstring(::Type{<:$T}) = $docstring
        end
        StatisticalTraits.target_scitype(::Type{<:$T}) = $target_scitype
        StatisticalTraits.prediction_type(::Type{<:$T}) = Symbol($pred_str)
        StatisticalTraits.supports_weights(::Type{<:$T}) = $supports_weights

        # traits specific to measures:
        if !isempty($instances)
            StatisticalTraits.instances(::Type{<:$T}) = $instances
        end
        if !isempty($human_name)
            StatisticalTraits.human_name(::Type{<:$T}) = $human_name
        end
        StatisticalTraits.orientation(::Type{<:$T}) = Symbol($orientation_str)
        StatisticalTraits.aggregation(::Type{<:$T}) = $aggregation
        StatisticalTraits.is_feature_dependent(::Type{<:$T}) =
            $is_feature_dependent
        StatisticalTraits.supports_class_weights(::Type{<:$T}) =
            $supports_class_weights
        StatisticalTraits.distribution_type(::Type{<:$T}) = $distribution_type

    end
    parentmodule(T).eval(ex)
end
