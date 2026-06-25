const FREEZABLE_SUPPORTED_ATOMS = (
    :Deterministic,
    :Probabilistic,
    :Interval,
    :Unsupervised,
)

# Each supported atomic type gets its own wrapper which must have appropriate supertype:
const FREEZABLE_TYPE_GIVEN_ATOM =
    Dict(atom =>
         Symbol("Freezable$atom") for atom in FREEZABLE_SUPPORTED_ATOMS)
const FREEZABLE_SUPER_GIVEN_ATOM =
    Dict(atom =>
         Symbol("$(atom)NetworkComposite") for atom in FREEZABLE_SUPPORTED_ATOMS)

# Type definitions:
for From in FREEZABLE_SUPPORTED_ATOMS
    New = FREEZABLE_TYPE_GIVEN_ATOM[From]
    To  = FREEZABLE_SUPER_GIVEN_ATOM[From]
    ex = quote
        mutable struct $New{M <: $From} <: $To
            model::M
            frozen::Bool
            cache::Bool
        end
    end
    eval(ex)
end

# dict whose keys and values are now types instead of symbols:
const freezable_type_given_atom = Dict()
for atom in FREEZABLE_SUPPORTED_ATOMS
    atom_str = string(atom)
    type = FREEZABLE_TYPE_GIVEN_ATOM[atom]
    @eval(freezable_type_given_atom[$atom] = $type)
end

# not exported:
const FREEZABLE_TYPES = values(freezable_type_given_atom)
const FREEZABLE_TYPE_EXS = values(FREEZABLE_TYPE_GIVEN_ATOM)
const SomeFreezable = Union{FREEZABLE_TYPES...}
const SupervisedFreezable = Union{
    freezable_type_given_atom[Deterministic],
    freezable_type_given_atom[Probabilistic],
    freezable_type_given_atom[Interval],
}
const FreezableSupported = Union{keys(freezable_type_given_atom)...}

const ERR_FREEZABLE_MODEL_UNSPECIFIED = ArgumentError(
    "Expecting atomic model as argument. None specified."
)
const ERR_FREEZABLE_TOO_MANY_ARGUMENTS = ArgumentError(
    "At most one non-keyword argument, a model, allowed."
)
const PRETTY_FREEZABLE_SUPPORT_OPTIONS =
    join([string("`", opt, "`") for opt in FREEZABLE_SUPPORTED_ATOMS],
         ", ",
         ", and ")
const err_freezable_unsupported(model) = ArgumentError(
    "Only these model supertypes support `Freezable` wrapping: "*
    "$PRETTY_FREEZABLE_SUPPORT_OPTIONS. "*
    "Model provided has type `$(typeof(model))`."
)

"""
    Freezable(model; frozen=true, cache=true)

Wrap `model` so `fit!` is a no-op after the first training pass. Place the wrapper inside
a `Pipeline`, `Stack`, `TunedModel`, or any other `NetworkComposite` model, and the
inner component skips retraining even when the parent rebuilds its learning network on a
row change.

Set `frozen=false` to allow normal retraining. Use [`freeze!`](@ref) and [`thaw!`](@ref)
to toggle after construction. Set `cache=false` to prioritize memory over speed.

### Example 1: Freezing a single model

This example and the next assume you have `MLJDecisionTreeInterface` in your environment.

```julia
using MLJ    # or `using MLJBase, MLJModels`

X, y = make_regression(100)
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree

model = Freezable(DecisionTreeRegressor())  # frozen=true by default
mach  = machine(model, X, y)

fit!(mach)                    # first fit trains
fit!(mach, rows=1:50)         # no-op while frozen
thaw!(model)
fit!(mach, rows=1:50)         # retrains
```

### Example 2: Freezing a component inside a pipeline

```julia
using MLJ    # or `using MLJBase, MLJModels, MLJTransforms`

X, y = make_blobs(200)
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

pipe = Pipeline(
    scaler = Freezable(Standardizer()),
    clf    = DecisionTreeClassifier(),
)
mach = machine(pipe, X, y)
fit!(mach, rows=1:100)        # both components train
fit!(mach, rows=101:200)      # only clf retrains; scaler is frozen
```

"""
function Freezable(
    args...;
    model=nothing,
    frozen::Bool=true,
    cache::Bool=true,
)
    length(args) < 2 || throw(ERR_FREEZABLE_TOO_MANY_ARGUMENTS)

    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification `model=$model`. "
    else
        model === nothing && throw(ERR_FREEZABLE_MODEL_UNSPECIFIED)
        atom = model
    end
    atom isa FreezableSupported || throw(err_freezable_unsupported(atom))

    abstract_atom = MMI.abstract_type(atom)
    haskey(freezable_type_given_atom, abstract_atom) ||
        throw(err_freezable_unsupported(atom))

    metamodel =
        freezable_type_given_atom[abstract_atom](atom,
                                                 frozen,
                                                 cache)
    message = clean!(metamodel)
    isempty(message) || @warn message
    return metamodel
end

function clean!(model::SomeFreezable)
    message = ""
    return message
end

"""
    freeze!(model::SomeFreezable)

Set `model.frozen = true`. After the first training pass, subsequent `fit!` calls on a
machine wrapping this model become no-ops.

See also [`thaw!`](@ref).
"""
freeze!(model::SomeFreezable) = (model.frozen = true; model)

"""
    thaw!(model::SomeFreezable)

Set `model.frozen = false`. The next `fit!` call on a machine wrapping this model
retrains.

See also [`freeze!`](@ref).
"""
thaw!(model::SomeFreezable) = (model.frozen = false; model)


# Prefit methods
function prefit(model::SupervisedFreezable, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    mach = machine(:model, Xs, ys; cache=model.cache)
    (predict=predict(mach, Xs), transform=transform(mach, Xs))
end

function prefit(model::FreezableUnsupervised, verbosity, X)
    Xs = source(X)
    mach = machine(:model, Xs; cache=model.cache)
    (transform=transform(mach, Xs), inverse_transform=inverse_transform(mach, Xs))
end

# Forwarded methods: the wrapper should be transparent to per-fit operations
# the inner model supports.

const ERR_FREEZABLE_MISSING_REPORT =
    "Cannot find report for the atomic model wrapped by `Freezable`. "

function MMI.training_losses(composite::SomeFreezable, freezable_report)
    hasproperty(freezable_report, :model) || throw(ERR_FREEZABLE_MISSING_REPORT)
    atomic_report = getproperty(freezable_report, :model)
    return training_losses(composite.model, atomic_report)
end

function MMI.feature_importances(composite::SupervisedFreezable, fitresult, report)
    predict_node = fitresult.interface.predict
    mach = only(MLJBase.machines_given_model(predict_node)[:model])
    return feature_importances(composite.model, mach.fitresult, mach.report[:fit])
end


# Model traits
MMI.package_name(::Type{<:SomeFreezable}) = "MLJBase"
MMI.is_wrapper(::Type{<:SomeFreezable}) = true
MMI.load_path(::Type{<:SomeFreezable}) = "MLJBase.Freezable"
MMI.constructor(::Type{<:SomeFreezable}) = Freezable

for New in FREEZABLE_TYPE_EXS
    quote
        MMI.iteration_parameter(::Type{<:$New{M}}) where M =
            MLJBase.prepend(:model, iteration_parameter(M))
    end |> eval
    for trait in [
        :input_scitype,
        :output_scitype,
        :target_scitype,
        :fit_data_scitype,
        :predict_scitype,
        :transform_scitype,
        :inverse_transform_scitype,
        :is_pure_julia,
        :supports_weights,
        :supports_class_weights,
        :supports_online,
        :supports_training_losses,
        :reports_feature_importances,
        :is_supervised,
        :prediction_type,
        ]
        quote
            MMI.$trait(::Type{<:$New{M}}) where M = MMI.$trait(M)
        end |> eval
    end
end
