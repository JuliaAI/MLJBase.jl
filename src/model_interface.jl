abstract type MLJType end


## MODEL API - TYPES

# for storing hyperparameters:
abstract type Model <: MLJType end

abstract type Supervised <: Model end
abstract type Unsupervised <: Model end

# supervised models that `predict` probability distributions are of:
abstract type Probabilistic <: Supervised end

# supervised models that `predict` point-values are of:
abstract type Deterministic <: Supervised end

# supervised models that `predict` intervals:
abstract type Interval <: Supervised end

# for static operations dependent on user-specified parameters:
abstract type Static <: Unsupervised end

# for models that are "exported" learning networks (return a Node as
# their fit-result; see MLJ docs:
abstract type ProbabilisticNetwork <: Probabilistic end
abstract type DeterministicNetwork <: Deterministic end
abstract type UnsupervisedNetwork <: Unsupervised end


## MODEL API - METHODS

# every model interface must implement a `fit` method of the form
# `fit(model, verbosity::Integer, training_args...) -> fitresult, cache, report`
# or, one the simplified versions
# `fit(model, training_args...) -> fitresult`
fit(model::Model, verbosity::Integer, args...) =
    fit(model, args...), nothing, nothing

# fallback for static transformations:
fit(model::Static, verbosity::Integer, args...) = nothing, nothing, nothing

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)

# fallbacks for supervised models that don't support sample weights:
fit(model::Supervised, verbosity::Integer, X, y, w) =
    fit(model, verbosity, X, y)
update(model::Supervised, verbosity, fitresult, cache, X, y, w) =
    update(model, verbosity, fitresult, cache, X, y)

# stub for online learning method update method
function update_data end

# methods dispatched on a model and fit-result are called
# *operations*.  Supervised models must implement a `predict`
# operation (extending the `predict` method of StatsBase).

# unsupervised methods must implement this operation:
function transform end

# unsupervised methods may implement this operation:
function inverse_transform end

# this operation can be optionally overloaded to provide access to
# fitted parameters (eg, coeficients of linear model):
fitted_params(::Model, fitresult) = (fitresult=fitresult,)

# probabilistic supervised models may also overload one or more of
# `predict_mode`, `predict_median` and `predict_mean` defined below.

# mode:
predict_mode(model::Probabilistic, fitresult, Xnew) =
    predict_mode(model, fitresult, Xnew, Val(target_scitype(model)))
predict_mode(model, fitresult, Xnew, ::Any) =
    mode.(predict(model, fitresult, Xnew))
const BadModeTypes = Union{AbstractArray{Continuous},Table(Continuous)}
predict_mode(model, fitresult, Xnew, ::Val{<:BadModeTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Continuous` targets. "))

# mean:
predict_mean(model::Probabilistic, fitresult, Xnew) =
    predict_mean(model, fitresult, Xnew, Val(target_scitype(model)))
predict_mean(model, fitresult, Xnew, ::Any) =
    mean.(predict(model, fitresult, Xnew))
const BadMeanTypes = Union{AbstractArray{<:Finite},Table(Finite)}
predict_mean(model, fitresult, Xnew, ::Val{<:BadMeanTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Finite` targets. "))

# median:
predict_median(model::Probabilistic, fitresult, Xnew) =
    predict_median(model, fitresult, Xnew, Val(target_scitype(model)))
predict_median(model, fitresult, Xnew, ::Any) =
    median.(predict(model, fitresult, Xnew))
const BadMedianTypes = Union{AbstractArray{<:Finite},Table(Finite)}
predict_median(model, fitresult, Xnew, ::Val{<:BadMedianTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Finite` targets. "))

# operations implemented by some meta-models:
function evaluate end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(model::Model) = ""


## STUB FOR @load (extended by MLJModels)

macro load end
