## EXPORTING LEARNING NETWORKS AS MODELS WITH @from_network

"""
    replace(W::Node, a1=>b1, a2=>b2, ...; empty_unspecified_sources=false)

Create a deep copy of a node `W`, and thereby replicate the learning
network terminating at `W`, but replacing any specified sources and
models `a1, a2, ...` of the original network with `b1, b2, ...`.

If `empty_unspecified_sources=ture` then any source nodes not
specified are replaced with empty version of the same kind.

"""
function Base.replace(W::Node, pairs::Pair...; empty_unspecified_sources=false)

    # Note: We construct nodes of the new network as values of a
    # dictionary keyed on the nodes of the old network. Additionally,
    # there are dictionaries of models keyed on old models and
    # machines keyed on old machines. The node and machine
    # dictionaries must be built simultaneously.

    # build model dict:
    model_pairs = filter(collect(pairs)) do pair
        first(pair) isa Model
    end
    models_ = models(W)
    models_to_copy = setdiff(models_, first.(model_pairs))
    model_copy_pairs = [model=>deepcopy(model) for model in models_to_copy]
    newmodel_given_old = IdDict(vcat(model_pairs, model_copy_pairs))

    # build complete source replacement pairs:
    sources_ = sources(W)
    specified_source_pairs = filter(collect(pairs)) do pair
        first(pair) isa Source
    end
    unspecified_sources = setdiff(sources_, first.(specified_source_pairs))
    unspecified_sources_wrapping_something =
        filter(s -> !isempty(s), unspecified_sources)
    if !isempty(unspecified_sources_wrapping_something) &&
        !empty_unspecified_sources
        @warn "No replacement specified for one or more non-empty source "*
        "nodes. Contents will be duplicated. "
    end
    if empty_unspecified_sources
        unspecified_source_pairs = [s => source(kind=MLJBase.kind(s)) for
                                    s in unspecified_sources]
    else
        unspecified_source_pairs = [s => deepcopy(s) for
                                    s in unspecified_sources]
    end

    all_source_pairs = vcat(specified_source_pairs, unspecified_source_pairs)

    # drop source nodes from all nodes of network terminating at W:
    nodes_ = filter(nodes(W)) do N
        !(N isa Source)
    end
    isempty(nodes_) && error("All nodes in network are source nodes. ")
    # instantiate node and machine dictionaries:
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}(all_source_pairs)
    newmach_given_old = IdDict{NodalMachine,NodalMachine}()

    # build the new network:
    for N in nodes_
       args = [newnode_given_old[arg] for arg in N.args]
         if N.machine === nothing
             newnode_given_old[N] = node(N.operation, args...)
         else
             if N.machine in keys(newmach_given_old)
                 mach = newmach_given_old[N.machine]
             else
                 train_args = [newnode_given_old[arg] for arg in N.machine.args]
                 mach = NodalMachine(newmodel_given_old[N.machine.model],
                                train_args...)
                 newmach_given_old[N.machine] = mach
             end
             newnode_given_old[N] = N.operation(mach, args...)
        end
    end

    return newnode_given_old[nodes_[end]]

 end

# closure for later:
function fit_method(network, models...)

    network_Xs = sources(network, kind=:input)[1]

    function _fit(model::M, verb::Integer, args...) where M <: Supervised
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [models[j] => replacement_models[j]
                              for j in eachindex(models)]
        network_ys = sources(network, kind=:target)[1]
        Xs = source(args[1])
        ys = source(args[2], kind=:target)
        source_replacements = [network_Xs => Xs, network_ys => ys]
        if length(args) == 3
            ws = source(args[3], kind=:weights)
            network_ws = sources(network, kind=:weights)[1]
            push!(source_replacements, network_ws => ws)
        end
        replacements = vcat(model_replacements, source_replacements)
        yhat = replace(network, replacements...; empty_unspecified_sources=true)

        if length(args) == 2
            issubset([Xs, ys], sources(yhat)) ||
                error("Failed to replace learning network "*
                      ":input or :target source")
        elseif length(args) == 3
            issubset([Xs, ys, ws], sources(yhat)) ||
                error("Failed to replace learning network "*
                      ":input or :target source")
        else
            throw(ArgumentError)
        end

        fit!(yhat, verbosity=verb)

        return fitresults(yhat)
    end

    function _fit(model::M, verb::Integer, X) where M <:Unsupervised
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [models[j] => replacement_models[j]
                          for j in eachindex(models)]
        Xs = source(X)
        source_replacements = [network_Xs => Xs,]
        replacements = vcat(model_replacements, source_replacements)
        Xout = replace(network, replacements...)
        Set([Xs]) == Set(sources(Xout)) ||
            error("Failed to replace learning network :input source ")

        fit!(Xout, verbosity=verb)

        return fitresults(Xout)
    end

    return _fit
end

net_alert(message) = throw(ArgumentError("Learning network export error.\n"*
                                     string(message)))
net_alert(k::Int) = throw(ArgumentError("Learning network export error $k. "))

# returns Model supertype - or `missing` if arguments are incompatible
function kind_(is_supervised, is_probabilistic)
    if is_supervised
        if ismissing(is_probabilistic) || !is_probabilistic
            return :DeterministicNetwork
        else
            return :ProbabilisticNetwork
        end
    else
        if ismissing(is_probabilistic) || !is_probabilistic
            return :UnsupervisedNetwork
        else
            return missing
        end
    end
end

function from_network_preprocess(modl, ex,
                                 is_probabilistic::Union{Missing,Bool})

    trait_ex_given_name_ex = Dict{Symbol,Any}()

    ex isa Expr || net_alert(1)
    ex.head == :call || net_alert(2)
    ex.args[1] == :(<=) || net_alert(3)
    ex.args[2] isa Expr || net_alert(4)
    ex.args[2].head == :call || net_alert(5)
    modeltype_ex = ex.args[2].args[1]
    modeltype_ex isa Symbol || net_alert(6)
    if length(ex.args[2].args) == 1
        kw_exs = []
    else
        kw_exs = ex.args[2].args[2:end]
    end
    fieldname_exs = []
    model_exs = []
    for ex in kw_exs
        ex isa Expr || net_alert(7)
        ex.head == :kw || net_alert(8)
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        variable_ex isa Symbol || net_alert(9)
        push!(fieldname_exs, variable_ex)
        value = modl.eval(value_ex)
        value isa Model ||
            net_alert("Got $value but expected something of type `Model`.")
        push!(model_exs, value_ex)
    end
    N_ex = ex.args[3]
    N = modl.eval(N_ex)
    N isa AbstractNode ||
        net_alert("Got $N but expected something of type `AbstractNode`. ")

    inputs = sources(N, kind=:input)
    targets = sources(N, kind=:target)
    weights = sources(N, kind=:weights)

    length(inputs) == 0 &&
        net_alert("Network has no source with `kind=:input`.")
    length(inputs) > 1  &&
        net_alert("Network has multiple sources with `kind=:input`.")
    length(targets) > 1 &&
        net_alert("Network has multiple sources with `kind=:target`.")
    length(weights) > 1 &&
        net_alert("Network has multiple sources with `kind=:weights`.")

    if length(weights) == 1
        trait_ex_given_name_ex[:supports_weights] = true
    end

    is_supervised = length(targets) == 1

    kind = kind_(is_supervised, is_probabilistic)
    ismissing(kind) &&
        net_alert("Network appears unsupervised (has no source with "*
                  "`kind=:target`) and so `is_probabilistic=true` "*
                  "declaration is not allowed. ")

    models_ = [modl.eval(e) for e in model_exs]
    issubset(models_, models(N)) ||
        net_alert("One or more specified models are not in the learning network "*
                  "terminating at $N_ex.\n Use models($N_ex) to inspect models. ")

    nodes_  = nodes(N)

    return modeltype_ex, fieldname_exs, model_exs, N_ex,
           kind, trait_ex_given_name_ex

end

from_network_preprocess(modl, ex) = from_network_preprocess(modl, ex, missing)

function from_network_preprocess(modl, ex, kw_ex)
    kw_ex isa Expr || net_alert(10)
    kw_ex.head == :(=) || net_alert(11)
    if kw_ex.args[1] == :is_probabilistic
        value = kw_ex.args[2]
        if value isa Bool
            return from_network_preprocess(modl, ex, value)
        else
            net_alert("`is_probabilistic` can only be `true` or `false`.")
        end
    elseif kw_ex.args[1] == :prediction_type
        value = kw_ex.args[2].value
        if value == :probabilistic
            return from_network_preprocess(modl, ex, true)
        elseif value == :deterministic
            return from_network_preprocess(modl, ex, false)
        else
            net_alert("`prediction_type` can only be `:probabilistic` "*
                      "or `:deterministic`.")
        end
    else
        net_alert("Unrecognized keywork `$(kw_ex.args[1])`.")
    end
end

function from_network_(modl, modeltype_ex, fieldname_exs, model_exs,
                          N_ex, kind, trait_value_ex_given_name_ex)

    args = gensym(:args)

    # code defining the composite model struct and fit method:
    program1 = quote

        import MLJBase
        import MLJModelInterface

        mutable struct $modeltype_ex <: MLJBase.$kind
            $(fieldname_exs...)
        end

        MLJModelInterface.fit(model::$modeltype_ex, verb::Integer, $args...) =
            MLJBase.fit_method($N_ex, $(model_exs...))(model, verb, $args...)

    end

    program2 = quote
        # defined keyword constructor for composite model:
        defaults =
            MLJBase.@set_defaults $modeltype_ex deepcopy.([$(model_exs...)])
    end

    modl.eval(program1)

    # define composite model traits:
    for (name_ex, value_ex) in trait_value_ex_given_name_ex
        program = quote
            MLJBase.$name_ex(::Type{<:$modeltype_ex}) = $value_ex
        end
        modl.eval(program)
    end

    modl.eval(program2)

end

"""
    @from_network(NewCompositeModel(fld1=model1, fld2=model2, ...) <= N
    @from_network(NewCompositeModel(fld1=model1, fld2=model2, ...) <= N is_probabilistic=false

Create a new stand-alone model type called `NewCompositeModel`, using
a learning network as a blueprint. Here `N` refers to the terminal
node of the learning network (from which final predictions or
transformations are fetched).

*Important.* If the learning network is supervised (has a source with
`kind=:target`) and makes probabilistic predictions, then one must
declare `is_probabilistic=true`. In the deterministic case the keyword
argument can be omitted.

The model type `NewCompositeModel` is equipped with fields named
`:fld1`, `:fld2`, ..., which correspond to component models `model1`,
`model2`, ...,  appearing in the network (which must therefore be elements of
`models(N)`).  Deep copies of the specified component models are used
as default values in an automatically generated keyword constructor
for `NewCompositeModel`.

### Return value

 A new `NewCompositeModel` instance, with default field values.

For details and examples refer to the "Learning Networks" section of
the documentation.

"""
macro from_network(exs...)

    args = from_network_preprocess(__module__, exs...)
    modeltype_ex = args[1]

    from_network_(__module__, args...)

    esc(quote
        $modeltype_ex()
        end)

end
