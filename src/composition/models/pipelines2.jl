# Code to construct pipelines without macros

# ## Note on mutability.

# The components in a pipeline, as defined here, can be replaced so
# long as their "abstract supertype" (eg, `Probabilistic`) remains the
# same. This is the type returned by `abstract_type()`; in the present
# code it will always be one of the types listed in
# `SUPPORTED_TYPES_FOR_PIPELINES` below, or `Any`, if `component` is
# not a model (which, by assumption, means it is callable).


# # HELPERS

# modify collection of symbols to guarantee uniqueness. For example,
# individuate([:x, :y, :x, :x]) = [:x, :y, :x2, :x3])
function individuate(v)
    isempty(v) && return v
    ret = Symbol[first(v),]
    for s in v[2:end]
        s in ret || (push!(ret, s); continue)
        n = 2
        candidate = s
        while true
            candidate = string(s, n) |> Symbol
            candidate in ret || break
            n += 1
        end
        push!(ret, candidate)
    end
    return ret
end

function as_type(prediction_type::Symbol)
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

_instance(x) = x
_instance(T::Type{<:Model}) = T()


# # TYPES

const SUPPORTED_TYPES_FOR_PIPELINES = [
    :Deterministic,
    :Probabilistic,
    :Interval,
    :Unsupervised,
    :Static]

const PIPELINE_TYPE_GIVEN_TYPE = Dict(
    :Deterministic => :DeterministicPipeline,
    :Probabilistic => :ProbabilisticPipeline,
    :Interval      => :IntervalPipeline,
    :Unsupervised  => :UnsupervisedPipeline,
    :Static        => :StaticPipeline)

const COMPOSITE_TYPE_GIVEN_TYPE = Dict(
    :Deterministic => :DeterministicComposite,
    :Probabilistic => :ProbabilisticComposite,
    :Interval      => :IntervalComposite,
    :Unsupervised  => :UnsupervisedComposite,
    :Static        => :StaticComposite)

const PREDICTION_TYPE_OPTIONS = [:deterministic,
                                 :probabilistic,
                                 :interval]

for T_ex in SUPPORTED_TYPES_FOR_PIPELINES
    P_ex = PIPELINE_TYPE_GIVEN_TYPE[T_ex]
    C_ex = COMPOSITE_TYPE_GIVEN_TYPE[T_ex]
    quote
        mutable struct $P_ex{N<:NamedTuple,operation} <: $C_ex
            named_components::N
            cache::Bool
            $P_ex(operation, named_components::N, cache) where N =
                new{N,operation}(named_components, cache)
        end
    end |> eval
end

# hack an alias for the union type, `SomePipeline{N,operation}` (not
# exported):
const _TYPE_EXS = map(values(PIPELINE_TYPE_GIVEN_TYPE)) do P_ex
    :($P_ex{N,operation})
end
quote
    const SomePipeline{N,operation} =
        Union{$(_TYPE_EXS...)}
end |> eval

# not exported:
const SupervisedPipeline{N,operation} =
    Union{DeterministicPipeline{N,operation},
          ProbabilisticPipeline{N,operation},
          IntervalPipeline{N,operation}}

components(p::SomePipeline) = values(getfield(p, :named_components))
names(p::SomePipeline) = keys(getfield(p, :named_components))
operation(p::SomePipeline{N,O}) where {N,O} = O


# # GENERIC CONSTRUCTOR

const PRETTY_PREDICTION_OPTIONS =
    join([string("`:", opt, "`") for opt in PREDICTION_TYPE_OPTIONS],
         ", ",
         ", and ")
const ERR_TOO_MANY_SUPERVISED = ArgumentError(
    "More than one supervised model in a pipeline is not permitted")
const ERR_EMPTY_PIPELINE = ArgumentError(
    "Cannot create an empty pipeline. ")
err_prediction_type_conflict(supervised_model, prediction_type) =
    ArgumentError("The pipeline's last component model has type "*
                  "`$(typeof(supervised_model))`, which conflicts "*
                  "with the declaration "*
                  "`prediction_type=$prediction_type`. ")
const INFO_TREATING_AS_DETERMINISTIC =
    "Treating pipeline as a `Deterministic` predictor.\n"*
    "To override, use `Pipeline` constructor with `prediction_type=...`. "*
    "Options are $PRETTY_PREDICTION_OPTIONS. "
const ERR_INVALID_PREDICTION_TYPE = ArgumentError(
    "Invalid `prediction_type`. Options are $PRETTY_PREDICTION_OPTIONS. ")
const WARN_IGNORING_PREDICTION_TYPE =
    "Pipeline appears to have no supervised "*
    "component models. Ignoring declaration "*
    "`prediction_type=$(prediction_type)`. "
const ERR_MIXED_PIPELINE_SPEC = ArgumentError(
    "Either specify all pipeline components without names, as in "*
    "`Pipeline(model1, model2)` or specify names for all "*
    "components, as in `Pipeline(myfirstmodel=model1, mysecondmodel=model2)`. ")


# The following combines its arguments into a named tuple, performing
# a number of checks and modifications. Specifically, it checks
# `components` is a valid sequence, modifies `names` to make them
# unique, and replaces the types appearing in the named tuple type
# parameters with their abstract supertypes. See the "Note on
# mutability" above.
function pipe_named_tuple(names, components)

    isempty(names) && throw(ERR_EMPTY_PIPELINE)

    # make keys unique:
    names = names |> individuate |> Tuple

    # check sequence:
    supervised_components = filter(components) do c
        c isa Supervised
    end
    length(supervised_components) < 2 ||
        throw(ERR_TOO_MANY_SUPERVISED)

    # return the named tuple:
    types = abstract_type.(components)
    NamedTuple{names,Tuple{types...}}(components)

end

"""
    Pipeline(component1, component2, ... , componentk; options...)
    Pipeline(name1=component1, name2=component2, ..., namek=componentk; options...)
    component1 |> component2 |> ... |> componentk

Create an instance of a composite model type which sequentially composes
the specified components in order. This means `component1` receives
inputs, whose output is passed to `component2`, and so forth. A
"component" is either a `Model` instance, a model type (converted
immediately to its default instance) or any callable object. Here the
"output" of a model is what `predict` returns if it is `Supervised`,
or what `transform` returns if it is `Unsupervised`.

Names for the component fields are automatically generated unless
explicitly specified, as in

```
Pipeline(endoder=ContinuousEncoder(drop_last=false),
         stand=Standardizer())
```

The `Pipeline` constructor accepts keyword `options` discussed further
below.

Ordinary functions (and other callables) may be inserted in the
pipeline as shown in the following example:

    Pipeline(X->coerce(X, :age=>Continuous), OneHotEncoder, ConstantClassifier)

### Syntactic sugar

The `|>` operator is overloaded to construct pipelines out of models,
callables, and existing pipelines:

```julia
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels add=true
PCA = @load PCA pkg=MultivariateStats add=true

pipe1 = MLJBase.table |> ContinuousEncoder |> Standardizer
pipe2 = PCA |> LinearRegressor
pipe1 |> pipe2
```

At most one of the components may be a supervised model, but this
model can appear in any position. A pipeline with a `Supervised`
component is itself `Supervised` and implements the `predict`
operation.  It is otherwise `Unsupervised` (possibly `Static`) and
implements `transform`.

### Special operations

If all the `components` are invertible unsupervised models (ie,
implement `inverse_transform`) then `inverse_transform` is implemented
for the pipeline. If there are no supervised models, then `predict` is
nevertheless implemented, assuming the last component is a model that
implements it (some clustering models). Similarly, calling `transform`
on a supervised pipeline calls `transform` on the supervised
component.

### Optional key-word arguments

- `prediction_type`  -
  prediction type of the pipeline; possible values: `:deterministic`,
  `:probabilistic`, `:interval` (default=`:deterministic` if not inferable)

- `operation` - operation applied to the supervised component model,
  when present; possible values: `predict`, `predict_mean`,
  `predict_median`, `predict_mode` (default=`predict`)

- `cache` - whether the internal machines created for component models
  should cache model-specific representations of data (see
  [`machine`](@ref)) (default=`true`)

!!! warning

    Set `cache=false` to guarantee data anonymization.

To build more complicated non-branching pipelines, refer to the MLJ
manual sections on composing models.

"""
function Pipeline(args...; prediction_type=nothing,
                  operation=predict,
                  cache=true,
                  kwargs...)

    # Components appear either as `args` (with names to be
    # automatically generated) or in `kwargs`, but not both.

    # This public constructor does checks and constructs a valid named
    # tuple, `named_components`, to be passed onto a secondary
    # constructor.

    isempty(args) || isempty(kwargs) ||
        throw(ERR_MIXED_PIPELINE_SPEC)

    operation in eval.(PREDICT_OPERATIONS) ||
        throw(ERR_INVALID_OPERATION)

    prediction_type in PREDICTION_TYPE_OPTIONS || prediction_type === nothing ||
        throw(ERR_INVALID_PREDICTION_TYPE)

    # construct the named tuple of components:
    if isempty(args)
        _names = keys(kwargs)
        _components = values(values(kwargs))
    else
        _names = Symbol[]
        for c in args
            generate_name!(c, _names, only=Model)
        end
        _components = args
    end

    # in case some components are specified as model *types* instead
    # of instances:
    components = _instance.(_components)

    named_components = pipe_named_tuple(_names, components)

    _pipeline(named_components, prediction_type, operation, cache)
end

function _pipeline(named_components::NamedTuple,
                   prediction_type,
                   operation,
                   cache)

    # This method assumes all arguments are valid and includes the
    # logic that determines which concrete pipeline's constructor
    # needs calling.

    components = values(named_components)

    # Is this a supervised pipeline?
    idx = findfirst(components) do c
        c isa Supervised
    end
    is_supervised = idx !== nothing
    is_supervised && @inbounds supervised_model = components[idx]

    # Is this a static pipeline? A component is *static* if it is an
    # instance of `Static <: Unsupervised` *or* a callable (anything
    # that is not a model, by assumption). When all the components are
    # static, the pipeline will be a `StaticPipeline`.
    static_components = filter(components) do m
        !(m isa Model) || m isa Static
    end

    is_static = length(static_components) == length(components)

    # To make final pipeline type determination, we need to determine
    # the corresonding abstract type (eg, `Probablistic`) here called
    # `super_type`:
    if is_supervised
        supervised_is_last = last(components) === supervised_model
        if prediction_type !== nothing
            super_type = as_type(prediction_type)
            supervised_is_last && !(supervised_model isa super_type) &&
                throw(err_prediction_type_conflict(supervised_model,
                                                   prediction_type))
        elseif supervised_is_last
            if operation != predict
                super_type = Deterministic
            else
                super_type = abstract_type(supervised_model)
            end
        else
            A = abstract_type(supervised_model)
            A == Deterministic || operation !== predict ||
                @info INFO_TREATING_AS_DETERMINISTIC
            super_type = Deterministic
        end
    else
        prediction_type === nothing ||
            @warn WARN_IGNORING_PREDICTION_TYPE
        super_type = is_static ? Static : Unsupervised
    end

    # dispatch on `super_type` to construct the appropriate type:
    _pipeline(super_type, operation, named_components, cache)

end

# where the method called in the last line will be one of these:
for T_ex in SUPPORTED_TYPES_FOR_PIPELINES
    P_ex = PIPELINE_TYPE_GIVEN_TYPE[T_ex]
    quote
        _pipeline(::Type{<:$T_ex}, args...) =
            $P_ex(args...)
    end |> eval
end


# # PROPERTY ACCESS

err_pipeline_bad_property(p, name) = ErrorException(
    "pipeline has no component `$name`")

Base.propertynames(p::SomePipeline{<:NamedTuple{names}}) where names =
    (names..., :cache)

function Base.getproperty(p::SomePipeline{<:NamedTuple{names}},
                          name::Symbol) where names
    name === :cache && return getfield(p, :cache)
    name in names && return getproperty(getfield(p, :named_components), name)
    throw(err_pipeline_bad_property(p, name))
end

function Base.setproperty!(p::SomePipeline{<:NamedTuple{names,types}},
                           name::Symbol, value) where {names,types}
    name === :cache && return setfield!(p, :cache, value)
    idx = findfirst(==(name), names)
    idx === nothing && throw(err_pipeline_bad_property(p, name))
    components = getfield(p, :named_components) |> values |> collect
    @inbounds components[idx] = value
    named_components = NamedTuple{names,types}(Tuple(components))
    setfield!(p, :named_components, named_components)
end


# # LEARNING NETWORK MACHINES FOR PIPELINES

# https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Learning-network-machines


# ## Methods to extend a pipeline learning network

# The "front" of a pipeline network, as we grow it, consists of a
# "predict" and a "transform" node. Once the pipeline is complete
# (after a series of `extend` operations - see below) the "transform"
# node is what is used to deliver the output of `transform(pipe)` in
# the exported model, and the "predict" node is what will be used to
# deliver the output of `predict(pipe). Both nodes can be changed by
# `extend` but only the "active" node is propagated.  Initially
# "transform" is active and "predict" only becomes active when a
# supervised model is encountered; this change is permanent.
# https://github.com/JuliaAI/MLJClusteringInterface.jl/issues/10

# `A == true` means `transform` is active
struct Front{A,P<:AbstractNode,N<:AbstractNode}
    predict::P
    transform::N
    Front(p::P, t::N, A) where {P,N} = new{A,P,N}(p, t)
end
active(f::Front{true})  = f.transform
active(f::Front{false}) = f.predict

function extend(front::Front{true},
                component::Supervised,
                cache,
                op,
                sources...)
    a = active(front)
    mach = machine(component, a, sources...; cache=cache)
    Front(op(mach, a), transform(mach, a), false)
end

function extend(front::Front{true}, component::Static, cache, args...)
    mach = machine(component; cache=cache)
    Front(front.predict, transform(mach, active(front)), true)
end

function extend(front::Front{false}, component::Static, cache, args...)
    mach = machine(component; cache=cache)
    Front(transform(mach, active(front)), front.transform, false)
end

function extend(front::Front{true}, component::Unsupervised, cache, args...)
    a = active(front)
    mach = machine(component, a; cache=cache)
    Front(predict(mach, a), transform(mach, a), true)
end

function extend(front::Front{false}, component::Unsupervised, cache, args...)
    a = active(front)
    mach = machine(component, a; cache=cache)
    Front(transform(mach, a), front.transform, false)
end

# fallback assumes `component` is a callable object:
extend(front::Front{true}, component, args...) =
    Front(front.predict, node(component, active(front)), true)
extend(front::Front{false}, component, args...) =
    Front(node(component, active(front)), front.transform, false)


# ## The learning network machine

const ERR_INVERSION_NOT_SUPPORTED = ErrorException(
    "Applying `inverse_transform` to a "*
    "pipeline that does not support it")

function pipeline_network_machine(super_type,
                                  cache,
                                  operation,
                                  components,
                                  source0,
                                  sources...)

    # initialize the network front:
    front = Front(source0, source0, true)

    # closure to use in reduction:
    _extend(front, component) =
        extend(front, component, cache, operation, sources...)

    # reduce to get the `predict` and `transform` nodes:
    final_front = foldl(_extend, components, init=front)
    pnode, tnode = final_front.predict, final_front.transform

    # backwards pass to get `inverse_transform` node:
    if all(c -> c isa Unsupervised, components)
        inode = source0
        node = tnode
        for i in eachindex(components)
            mach = node.machine
            inode = inverse_transform(mach, inode)
            node =  first(mach.args)
        end
    else
        inode = ErrorNode(ERR_INVERSION_NOT_SUPPORTED)
    end

    machine(super_type(), source0, sources...;
            predict=pnode, transform=tnode, inverse_transform=inode)

end


# # FIT METHOD

function MMI.fit(pipe::SomePipeline{N,operation},
                 verbosity::Integer,
                 arg0=source(),
                 args...) where {N,operation}

    source0 = source(arg0)
    sources = source.(args)

    _components = components(pipe)

    mach = pipeline_network_machine(abstract_type(pipe),
                                    pipe.cache,
                                    operation,
                                    _components,
                                    source0,
                                    sources...)
    return!(mach, pipe, verbosity)
end


# # SYNTACTIC SUGAR

const INFO_AMBIGUOUS_CACHE =
    "Joining pipelines with conflicting `cache` values. Using `cache=false`. "

import Base.(|>)
const compose = (|>)

Pipeline(p::SomePipeline) = p

const FuzzyModel = Union{Model,Type{<:Model}}

function compose(m1::FuzzyModel, m2::FuzzyModel)

    # no-ops for pipelines:
    p1 = Pipeline(m1)
    p2 = Pipeline(m2)

    _components = (components(p1)..., components(p2)...)
    _names = (names(p1)..., names(p2)...)

    named_components = pipe_named_tuple(_names, _components)

    # `cache` is only `true` if `true` for both pipelines:
    cache = false
    if p1.cache && p2.cache
        cache = true
    elseif p1.cache ⊻ p2.cache
        @info INFO_AMBIGUOUS_CACHE
    end

    _pipeline(named_components, nothing, operation(p2), cache)
end

compose(p1, p2::FuzzyModel) = compose(Pipeline(p1), p2)
compose(p1::FuzzyModel, p2) = compose(p1, Pipeline(p2))


# # TRAINING LOSSES

# ## Helpers

function supervised_component_name(pipe::SupervisedPipeline)
    idx = findfirst(model -> model isa Supervised, components(pipe))
    return names(pipe)[idx]
end

function supervised_component(pipe::SupervisedPipeline)
    name = supervised_component_name(pipe)
    named_components = getfield(pipe, :named_components)
    return getproperty(named_components, name)
end

model_type(::Machine{M}) where M = M
function supervised(machines)
    model_types = model_type.(machines)
    idx = findfirst(M -> M <: Supervised, model_types)
    return machines[idx]
end


# ## Traits

# We cannot provide the following trait at the level of types because
# the pipeline type does not know the precise type of the supervised
# component, only its `abstract_type`. See comment at top of page.
supports_training_losses(pipe::SupervisedPipeline) =
    supports_training_losses(getproperty(pipe, supervised_component_name(pipe)))

function training_losses(pipe::SupervisedPipeline, pipe_report)
    mach = supervised(pipe_report.machines)
    _report = report(mach)
    return training_losses(mach.model, _report)
end

# This trait cannot be defined at the level of types (see previous comment):
function iteration_parameter(pipe::SupervisedPipeline)
    model = supervised_component(pipe)
    name =  supervised_component_name(pipe)
    MLJBase.prepend(name, iteration_parameter(model))
end
