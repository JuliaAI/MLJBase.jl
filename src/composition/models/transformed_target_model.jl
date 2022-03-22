# wrapper `TransformedTargetModel`
# "TT" is shorthand for "TransformedTargetModel"

# # TYPES

const TT_SUPPORTED_ATOMS = (
    :Probabilistic,
    :ProbabilisticUnsupervisedDetector,
    :ProbabilisticSupervisedDetector,
    :Deterministic,
    :DeterministicUnsupervisedDetector,
    :DeterministicSupervisedDetector,
    :Interval)

# Each supported atomic type gets its own wrapper:

const TT_TYPE_GIVEN_ATOM =
    Dict(atom =>
         Symbol("TransformedTargetModel$atom") for atom in TT_SUPPORTED_ATOMS)

# ...which must have appropriate supertype:

const TT_SUPER_GIVEN_ATOM =
    Dict(atom =>
         Symbol("$(atom)Composite") for atom in TT_SUPPORTED_ATOMS)

# The type definitions:

for TransformedAbstract in TT_SUPPORTED_ATOMS
    New = TT_TYPE_GIVEN_ATOM[TransformedAbstract]
    To  = TT_SUPER_GIVEN_ATOM[TransformedAbstract]
    ex = quote
        mutable struct $New{M, target_scitype, predict_scitype, prediction_type} <: $To
            model::M
            target   # Unsupervised or callable
            inverse  # callable or `nothing`
            cache
        end
    end
    eval(ex)
end

# dict whose keys and values are now types instead of symbols:
const tt_type_given_atom = Dict()
for atom in TT_SUPPORTED_ATOMS
    atom_str = string(atom)
    type = TT_TYPE_GIVEN_ATOM[atom]
    @eval(tt_type_given_atom[$atom] = $type)
end

# not exported:
const TT_TYPES = values(tt_type_given_atom)
const TT_TYPE_EXS = values(TT_TYPE_GIVEN_ATOM)
const SomeTT = Union{TT_TYPES...}
const TTSupported = Union{keys(tt_type_given_atom)...}


# # CONSTRUCTOR

const ERR_MODEL_UNSPECIFIED = ArgumentError(
    "Expecting atomic model as argument. None specified. ")
const ERR_TOO_MANY_ARGUMENTS = ArgumentError(
    "At most one non-keyword argument, a model, allowed. ")
const PRETTY_TT_SUPPORT_OPTIONS =
    join([string("`", opt, "`") for opt in TT_SUPPORTED_ATOMS],
         ", ",
         ", and ")
const err_tt_unsupported(model) = ArgumentError(
    "Only these model supertypes support wrapping as in "*
    "`TransformedTarget`: $PRETTY_TT_SUPPORT_OPTIONS.\n"*
    "Model provided has type `$(typeof(model))`. ")
const WARN_IDENTITY_INVERSE =
    "Model being wrapped is not a deterministic predictor. "*
    "Setting `inverse=identity` to suppress inverse transformations "*
    "of predictions. "
const WARN_MISSING_INVERSE =
    "Specified `target` is not a model instance or type "*
    "and so is assumed callable (eg, is a function). "*
    "I am setting `inverse=identity` as no `inverse` specified. This means "*
    "predictions of the (semi)supervised model will be "*
    "returned on a scale different from the training target. "

"""
    TransformedTargetModel(
        model;
        target = nothing,
        inverse = nothing,
        cache = true,
        traits = nothing,
    )

Wrap the supervised or semi-supervised `model` in a transformation of
the target variable.

Here `target` one of the following:

- The `Unsupervised` model that is to transform the training target.
  By default (`inverse=nothing`) the parameters learned by this
  transformer are also used to inverse-transform the predictions of
  `model`, which means `target` must implement the `inverse_transform`
  method. If this is not the case, specify `inverse=identity` to
  suppress inversion.

- A callable object for transforming the target, such as `y ->
  log.(y)`. In this case a callable `inverse`, such as `z -> exp.(z)`,
  should be specified.

Specify `cache=false` to prioritize memory over speed, or to guarantee data
anonymity.

Specify `inverse=identity` if `model` is a probabilistic predictor, as
inverse-transforming sample spaces is not supported. Alternatively,
replace `model` with a deterministic model, such as `Pipeline(model,
y -> mode.(y))`.


### Examples

A model that normalizes the target before applying ridge regression,
with predictions returned on the original scale:

```
@load RidgeRegressor pkg=MLJLinearModels
model = RidgeRegressor()
tmodel = TransformedTargetModel(model, target=Standardizer())
```

A model that applies a static `log` transformation to the data, again
returning predictions to the original scale:

```
tmodel2 = TransformedTargetModel(model, target=y->log.(y), inverse=z->exp.(y))
```

"""
function TransformedTargetModel(
        args...;
        model = nothing,
        target = nothing,
        inverse = nothing,
        cache = true,
        traits::Union{AbstractDict, Nothing} = nothing,
    )
    length(args) < 2 || throw(ERR_TOO_MANY_ARGUMENTS)
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification `model=$model`. "
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
        atom = model
    end
    atom isa TTSupported || throw(err_tt_unsupported(atom))
    target === nothing && throw(ERR_TARGET_NOT_SPECIFIED)
    M = typeof(atom)
    OriginalAbstract = MMI.abstract_type(atom)
    if traits === nothing
        target_scitype  = MMI.target_scitype(M)
        predict_scitype = MMI.predict_scitype(M)
        prediction_type = MMI.prediction_type(M)
        TransformedAbstract = OriginalAbstract

    else
        target_scitype  = traits[:target_scitype]
        predict_scitype = traits[:predict_scitype]
        prediction_type = traits[:prediction_type]
        if prediction_type === :deterministic
            TransformedAbstract = MMI.Deterministic
        elseif prediction_type === :probabilistic
            TransformedAbstract = MMI.Probabilistic
        else
            msg = string(
                "Could not automatically infer the new abstract type from the ",
                "given trait values",
            )
            throw(ArgumentError(msg))
        end
    end
    tt_type = tt_type_given_atom[TransformedAbstract]
    tt_type_with_params = tt_type{M, target_scitype, predict_scitype, prediction_type}
    metamodel = tt_type_with_params(
        atom,
        target,
        inverse,
        cache,
    )
    message = clean!(metamodel)
    isempty(message) || @warn message
    return metamodel
end

_is_model_type(m) = m isa Type && m <: Model

function clean!(model::SomeTT)
    message = ""
    if prediction_type(model.model) !== :deterministic &&
        model.inverse != identity
        model.inverse = identity
        message *= WARN_IDENTITY_INVERSE
    end
    if !(model.target isa Model) &&
        !_is_model_type(model.target) && model.inverse === nothing
        model.inverse = identity
        message *= WARN_MISSING_INVERSE
    end
    return message
end


# # FIT METHOD

function MMI.fit(model::SomeTT, verbosity, X, y, other...)

    _target = model.target
    inverse = model.inverse
    atom = model.model
    cache = model.cache

    Xs = source(X)
    ys = source(y)
    others = source.(other)

    target = _is_model_type(_target) ? _target() : _target

    if target isa Model
        if target isa Static
            unsupervised_mach = machine(target, cache=cache)
        else
            unsupervised_mach = machine(target, ys, cache=cache)
        end
        z = transform(unsupervised_mach, ys)
    else
        z = node(target, ys)
    end

    supervised_mach = machine(atom, Xs, z, cache=cache)
    zhat = predict(supervised_mach, Xs)

    yhat = if target isa Model && inverse != identity
        inverse_transform(unsupervised_mach, zhat)
    else
        node(inverse, zhat)
    end

    # in case the atomic mode implements `transform`:
    W = transform(supervised_mach, Xs)

    network_mach =  machine(MMI.abstract_type(atom)(),
                            Xs,
                            ys,
                            others...;
                            predict=yhat,
                            transform=W)

    return!(network_mach, model, verbosity)
end


# # TRAINING LOSSES

function training_losses(model::SomeTT, tt_report)
    mach = first(tt_report.machines)
    return training_losses(mach)
end


## MODEL TRAITS

MMI.package_name(::Type{<:SomeTT}) = "MLJBase"
MMI.package_license(::Type{<:SomeTT}) = "MIT"
MMI.package_uuid(::Type{<:SomeTT}) = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
MMI.is_wrapper(::Type{<:SomeTT}) = true
MMI.package_url(::Type{<:SomeTT}) =
    "https://github.com/JuliaAI/MLJBase.jl"

for New in TT_TYPE_EXS
    quote
        MMI.iteration_parameter(::Type{<:$New{M}}) where M =
            MLJBase.prepend(:model, iteration_parameter(M))
    end |> eval
    for trait in [
        :input_scitype,
        # :output_scitype,
        :fit_data_scitype,
        # :transform_scitype,
        # :inverse_transform_scitype,
        :is_pure_julia,
        :supports_weights,
        :supports_class_weights,
        :supports_online,
        :supports_training_losses,
        :is_supervised,
        ]
        quote
            MMI.$trait(::Type{<:$New{M}}) where {M} = MMI.$trait(M)
        end |> eval
    end

    # TODO: figure out how to make the for loop work
    for trait in [
        :target_scitype,
        :predict_scitype,
        :prediction_type,
        ]
        quote
            MMI.$trait(
                ::Type{<:$New{M, target_scitype, predict_scitype, prediction_type}}
            ) where {M, target_scitype, predict_scitype, prediction_type} = $trait
        end |> eval
    end
end
