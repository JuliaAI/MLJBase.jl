## LINEAR LEARNING NETWORKS (FOR INTERNAL USE ONLY)

# Here `model` is a function or `Unsupervised` model (possibly
# `Static`).  This function wraps `model` in an appropriate `Node`,
# `node`, and returns `(node, mode.machine)`.
function node_and_machine(model, X)
    if model isa Unsupervised
        if model isa Static
            mach = machine(model)
        else
            mach = machine(model, X)
        end
        return transform(mach, X), mach
    else
        n = node(model, X)
        return n, n.machine
    end
end

# `models_and_functions` can include both functions (static
# operatations) and bona fide models (including `Static`
# transformers). If `ys == nothing` the learning network is assumed to
# be unsupervised. Otherwise: If `target === nothing`, then no target
# transform is applied; if `target !== nothing` and `inverse ===
# nothing`, then corresponding `transform` and `inverse_transform` are
# applied; if neither `target` nor `inverse` are `nothing`, then both
# `target` and `inverse` are assumed to be StaticTransformations, to
# be applied respectively to `ys` and to the output of the
# `models_and_functions` pipeline. Note that target inversion is
# applied to the output of the *last* nodal machine in the pipeline,
# corresponding to the last element of `models_and_functions`.

# returns a learning network machine

# No checks whatsoever are performed. Returns a learning network.
function linear_learning_network_machine(super_type,
                                         Xs,
                                         ys,
                                         ws,
                                         target,
                                         inverse,
                                         invert_last,
                                         operation,
                                         models_and_functions...)

    n = length(models_and_functions)

    if ys !== nothing && target !== nothing
        yt, target_machine = node_and_machine(target, ys)
    else
        yt  = ys
    end

    function tip_after_inversion(tip)
        if target !== nothing
            if inverse === nothing
                return inverse_transform(target_machine, tip)
            else
                return node_and_machine(inverse, tip) |> first
            end
        end
        return tip
    end

    if ws !== nothing
        tail_args = (yt, ws)
    else
        tail_args = (yt,)
    end

    # initialize the tip (terminal node) of network:
    tip = Xs  # last node added to network

    for m in models_and_functions
        if m isa Supervised
            supervised_machine = machine(m, tip, tail_args...)
            tip = operation(supervised_machine, tip)
            invert_last ||
                (tip = tip_after_inversion(tip))
        else
            tip = node_and_machine(m, tip) |> first
        end
    end

    invert_last && (tip = tip_after_inversion(tip))

    if super_type <: Unsupervised
        return machine(super_type(), Xs; transform=tip)
    elseif ws == nothing
        return machine(super_type(), Xs, ys; predict=tip)
    else
        return machine(super_type(), Xs, ys, ws; predict=tip)
    end

end


## PREPROCESSING

function is_uppercase(char::Char)
    i = Int(char)
    i > 64 && i < 91
end

function snakecase(str::AbstractString)
    snake = Char[]
    n = length(str)
    for i in eachindex(str)
        char = str[i]
        if is_uppercase(char)
            if i != 1 && i < n &&
                !(is_uppercase(str[i + 1]) && is_uppercase(str[i - 1]))
                push!(snake, '_')
            end
            push!(snake, lowercase(char))
        else
            push!(snake, char)
        end
    end
    return join(snake)
end

# `M` is a model type and the return value a `Symbol`. The
# `existing_names` gets updated.
function generate_name!(M::DataType, existing_names)
    str = split(string(M), '{') |> first
    candidate = split(str, '.') |> last |> snakecase |> Symbol
    candidate in existing_names ||
        (push!(existing_names, candidate); return candidate)
    n = 2
    new_candidate = candidate
    while true
        new_candidate = string(candidate, n) |> Symbol
        new_candidate in existing_names || break
        n += 1
    end
    push!(existing_names, new_candidate)
    return new_candidate
end

generate_name!(model::Model, existing_names) =
    generate_name!(typeof(model), existing_names)

pipe_alert(message) = throw(ArgumentError("@pipeline error.\n"*
                                     string(message)))
pipe_alert(k::Int) = throw(ArgumentError("@pipeline error $k. "))

not_unsupervised_alert(v) =
    pipe_alert("The value, `$v`, for `target` is not allowed. "*
               "Expecting an "*
               "`Unsupervised` model, subtype of "*
               "`Unsupervised`, or function. ")
pipe_argument_error(v) =
    pipe_alert("Encountered `$v` where a "*
               "model instance, model type, function, "*
               "or key-word assignement was expected. ")

function super_type(prediction_type::Symbol)
    if prediction_type == :deterministic
        return Deterministic
    elseif prediction_type == :probabilistic
        return Probabilistic
    elseif prediction_type == :interval
        return Interval
    else
        return Unsupervised
    end
end

function super_type(M::Type{<:Model})
    options = [Deterministic, Probabilistic,
               Interval, Unsupervised]
    idx = findfirst(super -> M <: super, options)
    return options[idx]
end

super_type(model::Model) = super_type(typeof(model))

# does expression processing, syntax and semantic
# checks. is_probabilistic is `true`, `false` or `missing`.
#function pipeline_preprocess(modl, ex, is_probabilistic::Union{Missing,Bool})
function pipeline_preprocess(modl, exs...)

    length(exs) > 0 || pipe_alert("Cannot create an empty pipeline. ")

    options = [:deterministic, :probabilistic,
               :interval]
    options_pretty =
        join([string(':',opt) for opt in options], ", ")

    fieldnames_ = Symbol[]
    models_ = Symbol[]
    model_supertypes_ = []
    models_and_functions_ = []
    models_and_functions  = []
    trait_ex_given_name_ = Dict{Symbol,Any}()

    pipetype_ = nothing
    pred_type = nothing
    target = nothing
    target_ = :nothing
    inverse = nothing
    inverse_ = :nothing
    invert_last = false
    operation = predict

    function add_model(ex)
        model_, model = eval_and_reassign(modl, ex)
        fieldname_ = generate_name!(model, fieldnames_)
        push!(models_and_functions_, model_)
        push!(models_and_functions, model)
        push!(models_, model_)
        push!(model_supertypes_, super_type(model))
    end

    for ex in exs
        if ex isa Expr || ex isa Symbol
            if ex isa Expr && ex.head == :(=)

                # --- key-word assignments ---

                variable_ = ex.args[1]
                variable_ isa Symbol || pipe_alert(8)
                if variable_ == :name
                    value_ = ex.args[2]
                    value_ isa Symbol || value_ isa AbstractString ||
                        value_ isa QuoteNode ||
                    pipe_alert("You cannot use `$value_` as the name "*
                               "for a pipeline type. ")
                    if value_ isa String
                        value_ = Symbol(value_)
                    end
                    if value_ isa QuoteNode
                        value_.value isa Symbol ||
                            pipe_alert("You cannot use `$value_` as the name "*
                                       "for a pipeline type. ")
                        value_ = value_.value
                    end
                    isdefined(modl, value_) &&
                        pipe_alert("`$(ex.args[2])` is the name of "*
                                   "an existing object. ")
                    pipetype_ = value_
                else
                    value_, value = eval_and_reassign(modl, ex.args[2])
                    if variable_ == :invert_last
                        value isa Bool ||
                            pipe_alert("`invert_last` must be "*
                                       "`true` or `false`. ") # untested
                        invert_last = value
                    elseif variable_ == :operation
                        value in eval.(OPERATIONS) &&
                            !(value in [transform, inverse_transform]) ||
                            pipe_alert("`operation=$value` is "*
                                       "unsupported. ") # untested
                        operation = value
                    elseif variable_ == :target
                        if value isa Function
                            value_, value =
                                eval_and_reassign(
                                    modl,
                                    :(MLJBase.WrappedFunction($value)))
                        end
                        if value isa DataType
                            value <: Unsupervised ||
                                not_unsupervised_alert(value)
                            value_, value =
                                eval_and_reassign(modl, Expr(:call, value_))
                        elseif value isa Model
                            value isa Unsupervised ||
                                not_unsupervised_alert(value)
                        else
                            not_unsupervised_alert(value)
                        end
                        target_ = value_
                        target = value
                        push!(fieldnames_, :target)
                        push!(models_, value_)
                        push!(model_supertypes_, super_type(value))
                    elseif variable_ == :inverse
                        if value isa Function
                            value_, value =
                                eval_and_reassign(
                                    modl,
                                    :(MLJBase.WrappedFunction($value)))
                        else
                            pipe_alert("`inverse` must be a function. ")
                        end
                        inverse_ = value_
                        inverse = value
                        push!(fieldnames_, :inverse)
                        push!(models_, value_)
                        push!(model_supertypes_, super_type(value))
                    elseif variable_ == :prediction_type
                        if !(value in options)
                            pipe_alert("`prediction_type` can only "*
                                       "be one of these: "*
                                       "$options_pretty. ")
                        else
                            pred_type = value
                        end
                    else
                        pipe_alert("Unrecognized key-word `$variable_`.")
                    end
                end

            else

                # --- model instances, model types and functions ---

                thing = modl.eval(ex)
                if thing isa Function
                    push!(models_and_functions_, ex)
                    push!(models_and_functions, thing)
                elseif thing isa Model
                    add_model(ex)
                elseif thing isa DataType || thing isa UnionAll
                    thing <: Model ||
                        pipe_argument_error(ex)
                    add_model(Expr(:call, ex))
                else
                    pipe_argument_error(ex)
                end
            end
        else
            pipe_argument_error(ex)
        end
    end

    pipetype_ === nothing &&
        (pipetype_ = "Pipeline"*string(gensym())[3:end] |> Symbol)

    inverse !== nothing && target === nothing &&
        pipe_alert("You have specified `inverse=...` but no `target`.")

    supervised_components = filter(models_and_functions) do m
        !(m isa Unsupervised) && !(m isa Function)
    end

    length(supervised_components) < 2 ||
        pipe_alert("More than one component of the pipeline is a "*
                   "supervised model .")

    is_supervised  = (length(supervised_components) == 1)

    non_static_components = filter(models_and_functions) do m
        m isa Unsupervised &&
            !(m isa Static) ||
            m isa Supervised
    end

    is_static = isempty(non_static_components)

    # assign traits where this can be done reliably and does not
    # overwrite user-defined ones:
    if is_supervised
        supervised_model = supervised_components[1]
        supervised_is_last =
            supervised_model == last(models_and_functions) &&
            target == nothing
        if !haskey(trait_ex_given_name_, :supports_weights)
            trait_ex_given_name_[:supports_weights] =
                supports_weights(supervised_model)
        end
        if !haskey(trait_ex_given_name_, :target_scitype) &&
            supervised_is_last
            trait_ex_given_name_[:target_scitype] =
                :(MLJBase.target_scitype($supervised_model))
        end
    elseif !haskey(trait_ex_given_name_, :output_scitype)
        last_model = last(models_and_functions)
        if last_model isa Model
            trait_ex_given_name_[:output_scitype] =
                output_scitype(last_model)
        end
    end

    # check consistency of user-defined prediction_type and infer
    # final super type:
    if is_supervised
        if pred_type != nothing
            super = super_type(pred_type)
            supervised_is_last && !(supervised_model isa super) &&
                pipe_alert("The pipeline's last component model has type "*
                           "`$(typeof(supervised_model))`, which conflicts "*
                           "the declaration "*
                           "`prediction_type=$pred_type`. ")
        elseif supervised_is_last
            if operation != predict
                super = Deterministic
            else
                super = super_type(supervised_model)
            end
        else
            @info "Treating pipeline as a `Deterministic` predictor.\n"*
            "To override, specify `prediction_type=...` "*
            "(options: $options_pretty). "
            super = Deterministic
        end
    else
        if pred_type != nothing
            @warn "Pipeline appears to have no supervised "*
            "component models. Ignoring declaration "*
            "`prediction_type=$(pred_type)`. "
        end
        super = is_static ? Static : Unsupervised
    end

    target isa MLJBase.WrappedFunction && inverse == nothing &&
        pipe_alert("It appears `target` is a function. "*
                   "You must therefore specify `inverse=...` .")

    target == nothing || is_supervised ||
        pipe_alert("`target=...` has been specified but no "*
                   "supervised components have been specified. ")

    return (pipetype_, fieldnames_, models_, model_supertypes_,
            models_and_functions_, target_, inverse_,
            invert_last, operation,
            super, trait_ex_given_name_)

end

function pipeline_(modl, exs...)

    (pipetype_, fieldnames_, models_, model_supertypes_,
     models_and_functions_,
     target_, inverse_, invert_last, operation,
     super_type, trait_ex_given_name_) =
         pipeline_preprocess(modl, exs...)

    if prediction_type === :unknown # unsupervised case
        ys_ = :nothing
    else
        ys_ = :(source())
    end

    if haskey(trait_ex_given_name_, :supports_weights) &&
        trait_ex_given_name_[:supports_weights]
        ws_ = :(source())
    else
        ws_ = nothing
    end

    mach_ = quote
        MLJBase.linear_learning_network_machine($super_type,
                                                source(nothing), $ys_, $ws_,
                                                $target_, $inverse_,
                                                $invert_last,
                                                $operation,
                                                $(models_and_functions_...))
    end

    field_declarations =
        [:($(fieldnames_[j])::$(model_supertypes_[j])=$(models_[j]))
                                for j in eachindex(models_)]

    supertype_ = _exported_type(modl.eval(mach_).model)

    struct_ = :(mutable struct $pipetype_ <: $supertype_
                        $(field_declarations...)
                    end)

    no_fields = length(models_) == 0

    from_network_(modl,
                  mach_,
                  pipetype_,
                  struct_,
                  no_fields,
                  trait_ex_given_name_)

    return pipetype_

end

"""
    @pipeline model1 model2 ... modelk

Create an instance of an automatically generated composite model type,
in which the specified models are composed in order. This means
`model1` receives inputs, whose output is passed to `model2`, and so
forth. Model types or instances may be specified.

At most one of the models may be a supervised model, but this model
can appear in any position.

The `@pipeline` macro accepts several key-word arguments discussed
further below.

Static (unlearned) transformations - that is, ordinary functions - may
also be inserted in the pipeline as shown in the following example:

    @pipeline X->coerce(X, :age=>Continuous) OneHotEncoder ConstantClassifier,


### Target transformation and inverse transformation

A learned target transformation (such as standardization) can also be
specified, using the key-word `target`, provided the transformer
provides an `inverse_transform` method:

    @pipeline OneHotEncoder KNNRegressor target=UnivariateTransformer

A static transformation can be specified instead, but then
an `inverse` must also be given:

    @pipeline(OneHotEncoder, KNNRegressor,
              target = v -> log.(v),
              inverse = v -> exp.(v))

*Important.* By default, the target inversion is applied *immediately
 following* the (unique) supervised model in the pipeline. To apply
 at the end of the pipeline, specify `invert_last=true`.


### Optional key-word arguments

- `target=...` - any `Unsupervised` model or `Function`

- `inverse=...` - any `Function` (unspecified if `target` is `Unsupervised`)

- `invert_last` (default=`true`) - set to `true` to delay target
  inversion to end of pipeline

- `prediction_type` (default=`:deterministic` if not inferable) -
  prediction type of the pipeline; possible values: `:deterministic`,
  `:probabilistic`, `:interval`

- `operation` (default=`predict`) - operation applied to the
  supervised component model, when present; possible values:
  `predict`, `predict_mean`, `predict_median`, `predict_mode`

- `name` (defaults to auto-generated) - new composite model type name;
  can be any name not already in current global namespace

See also: [`@from_network`](@ref)

"""
macro pipeline(exs...)

    if !isempty(exs) &&
        exs[1] isa Expr && exs[1].head == :call && length(exs[1].args) > 2
        pipetype_ = pipeline_deprecated_(__module__, exs...)
    else
        pipetype_ = pipeline_(__module__, exs...)
    end

    esc(quote
        $pipetype_()
        end)

end
