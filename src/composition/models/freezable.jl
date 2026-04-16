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

Wrap the atomic `model` in a `Freezable` wrapper. When `frozen=true`,
training is skipped after initial fit, even if training rows change.
This is useful for avoiding expensive recomputation during
cross-validation or hyperparameter tuning, at the cost of data
hygiene.

Unlike `freeze!(mach)`, which operates on an already-constructed
machine, `Freezable` operates at the model level. This means the
freeze semantics compose: a `Freezable`-wrapped model can be placed
inside a `Pipeline`, `Stack`, or `TunedModel`, and the inner
component will automatically skip retraining without the user needing
access to the internal machines that the composite creates.

Set `frozen=false` to allow normal retraining. The `frozen` field can
be toggled after construction.

Specify `cache=false` to prioritize memory over speed, or to guarantee
data anonymity.

### Example 1: Freezing a single model

```julia
using MLJBase

X, y = make_regression(100)

model = Freezable(DecisionTreeRegressor())  # frozen=true by default
mach = machine(model, X, y)

fit!(mach)                    # initial training always proceeds
predict(mach, X)              # works normally

fit!(mach, rows=1:50)         # no-op: frozen, so retraining is skipped

thaw!(model)                  # or equivalently: model.frozen = false
fit!(mach, rows=1:50)         # retrains on the new rows
```

### Example 2: Freezing a component inside a pipeline

The main use case for `Freezable` is inside composites. Here a
`Standardizer` is frozen so it is trained once on the first fold and
then reused across all subsequent folds, while the classifier
retrains normally on each fold:

```julia
using MLJBase

X, y = make_blobs(200)

pipe = Pipeline(
    scaler = Freezable(Standardizer()),   # trained once, then frozen
    clf    = DecisionTreeClassifier(),     # retrains on every fold
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

Set `model.frozen = true`. Subsequent `fit!` calls on a machine
wrapping this model will be no-ops (after initial training).

See also [`thaw!`](@ref).
"""
freeze!(model::SomeFreezable) = (model.frozen = true; model)

"""
    thaw!(model::SomeFreezable)

Set `model.frozen = false`. The next `fit!` call on a machine
wrapping this model will retrain normally.

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

function MLJModelInterface.fit(composite::SomeFreezable, verbosity, data...)
    # Build the learning network (inner machine starts unfrozen):
    fitresult = prefit(composite, verbosity, data...) |> MLJBase.Signature

    # Train the network (initial training always proceeds):
    greatest_lower_bound = MLJBase.glb(fitresult)
    acceleration = MLJBase.acceleration(fitresult)
    fit!(greatest_lower_bound; verbosity, composite, acceleration)

    # After initial training, freeze the inner machine if frozen=true:
    if composite.frozen
        d = MLJBase.machines_given_model(greatest_lower_bound)
        if haskey(d, :model)
            for mach in d[:model]
                freeze!(mach)
            end
        end
    end

    report = MLJBase.report(fitresult)

    # for passing to `update` so changes in `composite` can be detected:
    cache = deepcopy(composite)

    return fitresult, cache, report
end

function MLJModelInterface.update(
    composite::SomeFreezable,
    verbosity,
    fitresult,
    old_composite,
    data...,
)
    greatest_lower_bound = MLJBase.glb(fitresult)

    # Synchronize frozen state on the inner machine(s):
    d = MLJBase.machines_given_model(greatest_lower_bound)
    if haskey(d, :model)
        for mach in d[:model]
            composite.frozen ? freeze!(mach) : thaw!(mach)
        end
    end

    # Check if non-model, non-frozen hyperparameters changed (e.g., `cache`).
    # Changes to `frozen` are handled above via freeze!/thaw! and should not
    # trigger a full refit:
    non_frozen_changed = any(propertynames(composite)) do field
        field in MLJBase.models(greatest_lower_bound) && return false
        field === :frozen && return false
        old_value = getproperty(old_composite, field)
        value = getproperty(composite, field)
        value != old_value
    end
    non_frozen_changed && return MLJModelInterface.fit(composite, verbosity, data...)

    # retrain the network:
    fit!(greatest_lower_bound; verbosity, composite)

    report = MLJBase.report(fitresult)

    # for passing to `update` so changes in `composite` can be detected:
    cache = deepcopy(composite)

    return fitresult, cache, report
end


# When the Freezable model has frozen=true and the outer machine has already been
# trained (state > 0), skip retraining entirely. This prevents the NetworkComposite
# from rebuilding the learning network when training rows change.
#
# We synchronize the outer machine's `frozen` flag with the model's `frozen` field
# before each fit. The standard `fit_only!` checks `mach.frozen` at the top and
# returns immediately if true. Initial training (state == 0) always proceeds.
function MLJBase.fit!(mach::Machine{<:SomeFreezable}; kwargs...)
    # Synchronize outer machine frozen flag:
    if mach.model.frozen && mach.state > 0
        mach.frozen = true
    else
        mach.frozen = false
    end

    # Delegate to the standard fit! logic:
    glb_node = MLJBase.glb(mach.args...)
    fit!(glb_node; kwargs...)
    MLJBase.fit_only!(mach; kwargs...)
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
