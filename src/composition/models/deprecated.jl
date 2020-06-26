## LINEAR LEARNING NETWORKS (FOR INTERNAL USE ONLY)

# returns Model supertype - or `missing` if arguments are incompatible
function kind_(is_supervised, is_probabilistic)
    if is_supervised
        if ismissing(is_probabilistic) || !is_probabilistic
            return :DeterministicComposite
        else
            return :ProbabilisticComposite
        end
    else
        if ismissing(is_probabilistic) || !is_probabilistic
            return :UnsupervisedComposite
        else
            return missing
        end
    end
end

# Here `model` is a function or Unsupervised model (possibly
# `Static`).  This function wraps `model` in an appropriate `Node`,
# `node`, and returns `(node, mode.machine)`.
function node_(model, X)
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

# No checks whatsoever are performed. Returns a learning network.
function linear_learning_network(Xs,
                                 ys,
                                 ws,
                                 target,
                                 inverse,
                                 models_and_functions...)

    n = length(models_and_functions)

    if ys !== nothing && target !== nothing
        yt, target_machine = node_(target, ys)
    else
        yt  = ys
    end

    if ws !== nothing
        tail_args = (yt, ws)
    else
        tail_args = (yt,)
    end

    nodes = Vector(undef, n + 1)
    nodes[1] = Xs

    for i = 2:(n + 1)
        m = models_and_functions[i-1]
        if m isa Supervised
            supervised_machine = machine(m, nodes[i-1], tail_args...)
            nodes[i] = predict(supervised_machine, nodes[i-1])
       else
            nodes[i] = node_(m, nodes[i-1]) |> first
        end
    end

    if target === nothing
        terminal_node=nodes[end]
    else
        if inverse === nothing
            terminal_node = inverse_transform(target_machine, nodes[end])
        else
            terminal_node = node_(inverse, nodes[end]) |> first
        end
    end

    return terminal_node

end


## PREPROCESSING

function eval_and_reassign_deprecated(modl, ex)
    s = gensym()
    evaluated = modl.eval(ex)
    modl.eval(:($s = $evaluated))
    return s, evaluated
end

# does expression processing, syntax and semantic
# checks. is_probabilistic is `true`, `false` or `missing`.
function pipeline_preprocess_deprecated(
    modl, ex, is_probabilistic::Union{Missing,Bool})

    ex isa Expr || pipe_alert(2)
    length(ex.args) > 1 || pipe_alert(3)
    ex.head == :call || pipe_alert(4)
    pipetype_ = ex.args[1]
    pipetype_ isa Symbol || pipe_alert(5)

    fieldnames_ = []               # for `from_network`
    models_ = []                   # for `from_network`
    models_and_functions_ = []     # for `linear_learning_network`
    models_and_functions  = []     # for `linear_learning_network`
    trait_value_given_name_ = Dict{Symbol,Any}()
    for ex in ex.args[2:end]
        if ex isa Expr
            if ex.head == :kw
                variable_ = ex.args[1]
                variable_ isa Symbol || pipe_alert(8)
                value_, value = eval_and_reassign_deprecated(modl, ex.args[2])
                if variable_ == :target
                    if value isa Function
                        value_, value =
                            eval_and_reassign_deprecated(modl,
                                              :(MLJBase.WrappedFunction($value)))
                    end
                    value isa Unsupervised ||
                        pipe_alert("Got $value where a function or "*
                                   "Unsupervised model instance was "*
                                   "expected. ")
                    target_ = value_
                    target = value
                    push!(fieldnames_, :target)
                elseif variable_ == :inverse
                    if value isa Function
                        value_, value =
                            eval_and_reassign_deprecated(modl,
                                              :(MLJBase.WrappedFunction($value)))
                    else
                        pipe_alert(10)
                    end
                    inverse_ = value_
                    inverse = value
                    push!(fieldnames_, :inverse)
                else
                    value isa Model ||
                        throw(ArgumentError("$value given where `Model` "*
                                            "instance expected. "))
                    push!(models_and_functions_, value_)
                    push!(models_and_functions, value)
                    push!(fieldnames_, variable_)
                end
                push!(models_, value_)
            else
                f = modl.eval(ex)
                f isa Function ||
                    pipe_alert("Perhaps a missing name, as `name=$f'?")
                push!(models_and_functions_, ex)
                push!(models_and_functions, f)
            end
        else
            f = modl.eval(ex)
            f isa Function || pipe_alert(7)
            push!(models_and_functions_, ex)
            push!(models_and_functions, f)
        end
    end

    (@isdefined target)  || (target = nothing; target_ = :nothing)
    (@isdefined inverse) || (inverse = nothing; inverse_ = :nothing)
    inverse !== nothing && target === nothing &&
        pipe_alert("You have specified `inverse=...` but no `target`.")

    supervised(m) = !(m isa Unsupervised) && !(m isa Function)

    supervised_components = filter(supervised, models_and_functions)

    length(supervised_components) < 2 ||
        pipe_alert("More than one component of the pipeline is a "*
                   "supervised model .")

    if length(supervised_components) == 1
        model = supervised_components[1]
        trait_value_given_name_[:supports_weights] =
            supports_weights(model)
    end

    is_supervised  =
        length(supervised_components) == 1

    # `kind_` is defined in /src/composition/models/from_network.jl
    kind = kind_(is_supervised, is_probabilistic)

    ismissing(kind) &&
        pipe_alert("Composite has no supervised components and so "*
                  "`prediction_type=:probablistic` (or "*
                   "`is_probabilistic=true`) "*
                  "declaration is not allowed. ")

    target isa MLJBase.WrappedFunction && inverse == nothing &&
        pipe_alert("It appears `target` is a function. "*
                   "You must therefore specify `inverse=...` .")

    target == nothing || is_supervised ||
        pipe_alert("`target=...` has been specified but no "*
                   "supervised components have been specified. ")

    return (pipetype_, fieldnames_, models_,
            models_and_functions_, target_, inverse_,
            kind, trait_value_given_name_)

end

function pipeline_preprocess_deprecated(modl, ex, kw_ex)
    kw_ex isa Expr || pipe_alert(10)
    kw_ex.head == :(=) || pipe_alert(11)
    if kw_ex.args[1] == :is_probabilistic
        value_ = kw_ex.args[2]
        if value_ == :missing
            value = missing
        else
            value = value_
            value isa Bool ||
                pipe_alert("`is_probabilistic` can only be `true` or `false`.")
        end
    elseif kw_ex.args[1] == :prediction_type
        value_ = kw_ex.args[2].value
        if value_ == :probabilistic
            value = true
        elseif value_ == :deterministic
            value =false
        else
            pipe_alert("`prediction_type` can only be `:probabilistic` "*
                      "or `:deterministic`.")
        end
    else
        pipe_alert("Unrecognized keywork `$(kw_ex.args[1])`.")
    end
    return pipeline_preprocess_deprecated(modl, ex, value)
end

function pipeline_deprecated_(modl, ex, kw_ex)

    Base.depwarn("Using deprecated @pipeline syntax. "*
                 "New syntax:\n   @pipeline model1 model2 ... \n "*
                 "Query the docstring or see manual for options and "*
                 "further details. ", Symbol("@pipeline"))

    (pipetype_, fieldnames_, models_, models_and_functions_,
     target_, inverse_, kind, trait_value_given_name_) =
         pipeline_preprocess_deprecated(modl, ex, kw_ex)

    if kind === :UnsupervisedComposite
        ys_ = :nothing
    else
        ys_ = :(source(kind=:target))
    end

    if haskey(trait_value_given_name_, :supports_weights) &&
        trait_value_given_name_[:supports_weights]
        ws_ = :(source(kind=:weights))
    else
        ws_ = nothing
    end

    if kind == :DeterministicComposite
        super_type = Deterministic
    elseif kind == :ProbabilisticComposite
        super_type = Probabilistic
    else
        super_type = Unsupervised
    end

    mach_ = quote
        MLJBase.linear_learning_network_machine($super_type,
                                                source(nothing), $ys_, $ws_,
                                                $target_, $inverse_,
                                                true,
                                                predict,
                                                $(models_and_functions_...))
    end

    field_declarations =
        [:($(fieldnames_[j])=$(models_[j])) for j in eachindex(models_)]

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
                  trait_value_given_name_)

    # N_ex = quote

    #     MLJBase.linear_learning_network(source(nothing), $ys_, $ws_,
    #                                 $target_, $inverse_,
    #                                 $(models_and_functions_...))
    # end

    # from_network_(modl, pipetype_, fieldnames_, models_, N_ex,
    #               kind, trait_value_given_name_)

    return pipetype_

end

pipeline_deprecated_(modl, ex) =
    pipeline_deprecated_(modl, ex, :(is_probabilistic=missing))

macro pipeline_deprecated(exs...)

    pipetype_ = pipeline_deprecated_(__module__, exs...)

    esc(quote
        $pipetype_()
        end)

end
