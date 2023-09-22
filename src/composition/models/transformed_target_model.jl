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
         Symbol("$(atom)NetworkComposite") for atom in TT_SUPPORTED_ATOMS)

# The type definitions:

for From in TT_SUPPORTED_ATOMS
    New = TT_TYPE_GIVEN_ATOM[From]
    To  = TT_SUPER_GIVEN_ATOM[From]
    ex = quote
        mutable struct $New{M <: $From} <: $To
            model::M
            transformer   # Unsupervised or callable
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
    "Expecting atomic model as argument. None specified. "
)
const ERR_TRANSFORMER_UNSPECIFIED = ArgumentError(
"You must specify `transformer=...`. ."
)
const ERR_TOO_MANY_ARGUMENTS = ArgumentError(
    "At most one non-keyword argument, a model, allowed. "
)
const PRETTY_TT_SUPPORT_OPTIONS =
    join([string("`", opt, "`") for opt in TT_SUPPORTED_ATOMS],
         ", ",
         ", and ")
const err_tt_unsupported(model) = ArgumentError(
    "Only these model supertypes support wrapping as in "*
    "`TransformedTarget`: $PRETTY_TT_SUPPORT_OPTIONS.\n"*
    "Model provided has type `$(typeof(model))`. "
)
const WARN_IDENTITY_INVERSE =
    "Model being wrapped is not a deterministic predictor. "*
    "Setting `inverse=identity` to suppress inverse transformations "*
    "of predictions. "
const WARN_MISSING_INVERSE =
    "Specified `transformer` is not a model instance or type "*
    "and so is assumed callable (eg, is a function). "*
    "I am setting `inverse=identity` as no `inverse` specified. This means "*
    "predictions of the (semi)supervised model will be "*
    "returned on a scale different from the training target. "

const WARN_TARGET_DEPRECATED =
    "`TransformedTargetModel(target=...)` is deprecated in favor of "*
    "`TransformedTargetModel(transformer=...)`. "

"""
    TransformedTargetModel(model; transformer=nothing, inverse=nothing, cache=true)

Wrap the supervised or semi-supervised `model` in a transformation of
the target variable.

Here `transformer` one of the following:

- The `Unsupervised` model that is to transform the training target.
  By default (`inverse=nothing`) the parameters learned by this
  transformer are also used to inverse-transform the predictions of
  `model`, which means `transformer` must implement the `inverse_transform`
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
tmodel = TransformedTargetModel(model, transformer=Standardizer())
```

A model that applies a static `log` transformation to the data, again
returning predictions to the original scale:

```
tmodel2 = TransformedTargetModel(model, transformer=y->log.(y), inverse=z->exp.(y))
```

"""
function TransformedTargetModel(
    args...;
    model=nothing,
    transformer=nothing,
    inverse=nothing,
    cache=true,
)

    isnothing(target) ||
        Base.depwarn(WARN_TARGET_DEPRECATED, :TransformedTargetModel, force=true)
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
    transformer === nothing && throw(ERR_TRANSFORMER_UNSPECIFIED)

    metamodel =
        tt_type_given_atom[MMI.abstract_type(atom)](atom,
                                                    transformer,
                                                    inverse,
                                                    cache)
    message = clean!(metamodel)
    isempty(message) || @warn message
    return metamodel
end

_is_model_type(m) = m isa Type && m <: Model

function clean!(model::SomeTT)
    message = ""
    if _is_model_type(model.transformer)
        model.transformer = model.transformer()
    end
    if prediction_type(model.model) !== :deterministic &&
        model.inverse != identity
        model.inverse = identity
        message *= WARN_IDENTITY_INVERSE
    end
    if !(model.transformer isa Model) &&
        !_is_model_type(model.transformer) && model.inverse === nothing
        model.inverse = identity
        message *= WARN_MISSING_INVERSE
    end
    return message
end


# # PREFIT METHOD

function prefit(model::SomeTT, verbosity, X, y, other...)

    transformer = model.transformer
    inverse = model.inverse
    atom = model.model
    cache = model.cache

    Xs = source(X)
    ys = source(y)
    others = source.(other)

    if transformer isa Model
        if transformer isa Static
            unsupervised_mach = machine(:transformer, cache=cache)
        else
            unsupervised_mach = machine(:transformer, ys, cache=cache)
        end
        z = transform(unsupervised_mach, ys)
    else
        z = node(transformer, ys)
    end

    supervised_mach = machine(:model, Xs, z, cache=cache)
    zhat = predict(supervised_mach, Xs)

    yhat = if transformer isa Model && inverse != identity
        inverse_transform(unsupervised_mach, zhat)
    else
        node(inverse, zhat)
    end

    # in case the atomic model implements `transform`:
    W = transform(supervised_mach, Xs)

    # learning network interface:
    (predict=yhat, transform=W)

end


# # TRAINING LOSSES

const ERR_TT_MISSING_REPORT =
    "Cannot find report for `TransformedTargetModel` atomic model, from which "*
    "to extract training losses. "

function training_losses(composite::SomeTT, tt_report)
    hasproperty(tt_report, :model) || throw(ERR_TT_MISSING_REPORT)
    atomic_report = getproperty(tt_report, :model)
    return training_losses(composite.model, atomic_report)
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
    for trait in [:input_scitype,
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
        :is_supervised,
        :prediction_type]
        quote
            MMI.$trait(::Type{<:$New{M}}) where M = MMI.$trait(M)
        end |> eval
    end
end
