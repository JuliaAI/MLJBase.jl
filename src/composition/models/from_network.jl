## EXPORTING LEARNING NETWORKS AS MODELS WITH @from_network

# closure to generate the fit methods for exported composite. Here
# `mach` is a learning network machine.
function fit_method(mach, models...)

    signature = mach.fitresult
    mach_args = mach.args

    function _fit(model, verbosity::Integer, args...)
        length(args) > length(mach_args) &&
            throw(ArgumentError("$M does not support more than "*
                                "$(length(mach_args)) training arguments"))
        replacement_models = [getproperty(model, fld)
                              for fld in propertynames(model)]
        model_replacements = [models[j] => replacement_models[j]
                              for j in eachindex(models)]
        source_replacements = [mach_args[i] => source(args[i])
                               for i in eachindex(args)]
        replacements = vcat(model_replacements, source_replacements)

        new_mach =
            replace(mach, replacements...; empty_unspecified_sources=true)

        return!(new_mach, model, verbosity)
    end

    return _fit
end

net_error(message) = throw(ArgumentError("Learning network export error.\n"*
                                     string(message)))
net_error(k::Int) = throw(ArgumentError("Learning network export error $k. "))

_insert_subtyping(ex, subtype_ex) =
    Expr(:(<:), ex, subtype_ex)

# create the exported type symbol, e.g. abstract_type(T) == Unsupervised
# would result in :UnsupervisedComposite
_exported_type(T::Model) = Symbol(nameof(abstract_type(T)), :Composite)

function eval_and_reassign(modl, ex)
    s = gensym()
    evaluated = modl.eval(ex)
    if evaluated isa Symbol
        hack = String(evaluated)
        modl.eval(:($s = Symbol($hack)))
    else
        modl.eval(:($s = $evaluated))
    end
    return s, evaluated
end

function without_line_numbers(block_ex)
    block_ex.head == :block || throw(ArgumentError)
    args = filter(block_ex.args) do arg
        !(arg isa LineNumberNode)
    end
    return Expr(:block, args...)
end

function from_network_preprocess(modl, mach_ex, block_ex)

    mach_ex, mach  = eval_and_reassign(modl, mach_ex)
    mach isa Machine{<:Surrogate} ||
        net_error("$mach is not a learning network machine. ")
    if block_ex.head == :block
        block_ex = without_line_numbers(block_ex)
        struct_ex = block_ex.args[1]
        trait_declaration_exs = block_ex.args[2:end]
    elseif block_ex.head == :struct
        struct_ex = block_ex
        trait_declaration_exs = []
    else
        net_error("Expected `struct`, `mutable struct` or "*
                  "`begin ... end` block, but got `$block_ex` ")
    end

    # if necessary add or modify struct subtyping:
    if struct_ex.args[2] isa Symbol
        struct_ex.args[2] = _insert_subtyping(struct_ex.args[2],
                                              _exported_type(mach.model))
        modeltype_ex = struct_ex.args[2].args[1]
    elseif struct_ex.args[2] isa Expr
        struct_ex.args[2].head == :(<:) ||
                    net_error("Badly formed `struct` subtying. ")
        modeltype_ex = struct_ex.args[2].args[1]
        super = eval(struct_ex.args[2].args[2])
        inferred_super_ex = _exported_type(mach.model)
        if !(super <: Composite)
            @warn "New composite type must subtype `Composite` but "*
            "`$super` does not. Instead declaring "*
            "`$modeltype_ex <: $inferred_super_ex`. "
            struct_ex.args[2].args[2] = inferred_super_ex
        end
    else
        net_error(41)
    end

    # test if there are no fields:
    field_exs = without_line_numbers(struct_ex.args[3]).args
    no_fields = isempty(field_exs)

    # extract trait definitions:
    trait_ex_given_name_ex = Dict{Symbol,Any}()

    ne() = net_error("Bad trait declaration. ")
    for ex in trait_declaration_exs
        ex isa Expr           || ne()
        ex.head == :(=)       || ne()
        ex.args[1] isa Symbol || ne()
        ex.args[1] in MLJModelInterface.MODEL_TRAITS ||
            net_error("Expected a model trait as keywork but "*
                      "got $(ex.args[2]). Options are:\n"*
                      "$MLJModelInterface.MODEL_TRAIES. ")
        length(ex.args) == 2  || ne()
        trait_ex_given_name_ex[ex.args[1]] = ex.args[2]
    end

    return mach_ex, modeltype_ex, struct_ex, no_fields, trait_ex_given_name_ex

end

function from_network_(modl,
                       mach_ex,
                       modeltype_ex,
                       struct_ex,
                       no_fields,
                       trait_ex_given_name_ex)

    args = gensym(:args)
    models = gensym(:models)
    instance = gensym(:instance)

    # Define the new model type with keyword constructor:
    if no_fields
        modl.eval(struct_ex)
    else
        modl.eval(MLJBase.Parameters.with_kw(struct_ex, modl, false))
    end

    # Test that an instance can be created:
    try
        modl.eval(:($modeltype_ex()))
    catch e
        @error "Problem instantiating a default instance of the "*
        "new composite type. Each field name in the struct expression "*
        "must have a corresponding model instance (that also appears "*
        "somewhere in the network). "*
        "Perhaps you forgot to specify one of these?"
        throw(e)
    end

    # code defining fit method:
    program1 = quote

        $(isdefined(modl, :MLJ) ? :(import MLJ.MLJBase) : :(import MLJBase))
        $(isdefined(modl, :MLJ) ? :(import MLJ.MLJBase.MLJModelInterface) :
          :(import MLJBase.MLJModelInterface))

        $instance = $modeltype_ex()
        $models = [getproperty($instance, name)
                   for name in fieldnames($modeltype_ex)]

        MLJModelInterface.fit(model::$modeltype_ex, verb::Integer, $args...) =
            MLJBase.fit_method($mach_ex, $models...)(model, verb, $args...)

    end

    modl.eval(program1)

    # define composite model traits:
    for (name_ex, value_ex) in trait_ex_given_name_ex
        program = quote
            MLJBase.$name_ex(::Type{<:$modeltype_ex}) = $value_ex
        end
        modl.eval(program)
    end

    return nothing

end


"""

    @from_network mach [mutable] struct NewCompositeModel
           ...
    end

or

    @from_network mach begin
        [mutable] struct NewCompositeModel
           ...
        end
        <optional trait declarations>
    end

Create a new stand-alone model type called `NewCompositeModel`, using
the specified learning network machine `mach` as a blueprint.

For more on learning network machines, see [`machine`](@ref).


### Example

Consider the following simple learning network for training a decision
tree after one-hot encoding the inputs, and forcing the predictions to
be point-predictions (rather than probabilistic):

```julia
Xs = source()
ys = source()

hot = OneHotEncoder()
tree = DecisionTreeClassifier()

W = transform(machine(hot, Xs), Xs)
yhat = predict_mode(machine(tree, W, ys), W)
```

A learning network machine is defined by

```julia
mach = machine(Deterministic(), Xs, ys; predict=yhat)
```

To specify a new `Deterministic` composite model type `WrappedTree` we
specify the model instances appearing in the network as "default"
values in the following decorated struct definition:

```julia
@from_network mach struct WrappedTree
    encoder=hot
    decision_tree=tree
end
```
and create a new instance with `WrappedTree()`.

To allow the second model component to be replaced by any other
probabilistic model we instead make a mutable struct declaration and,
if desired, annotate types appropriately.  In the following code
illustration some model trait declarations have also been added:

```julia
@from_network mach begin
    mutable struct WrappedTree
        encoder::OneHotEncoder=hot
        classifier::Probabilistic=tree
    end
    input_scitype = Table(Continuous, Finite)
    is_pure_julia = true
end
```

"""
macro from_network(exs...)
    args = from_network_preprocess(__module__, exs...)
    modeltype_ex = args[2]
    from_network_(__module__, args...)
end
