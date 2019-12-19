## THE CONSTANT REGRESSOR

export ConstantClassifier, ConstantRegressor,
    DeterministicConstantClassifier,
    ProbabilisticConstantClassifer

import MLJBase
import MLJBase: metadata_pkg, metadata_model
import Distributions

"""
ConstantRegressor(; distribution_type=Distributions.Normal)

A regressor that, for any new input pattern, predicts the univariate
probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.
"""
struct ConstantRegressor{D} <: MLJBase.Probabilistic
    distribution_type::Type{D}
end

function ConstantRegressor(; distribution_type=Distributions.Normal)
    model   = ConstantRegressor(distribution_type)
    message = clean!(model)
    isempty(message) || @warn message
    return model
end

function clean!(model::ConstantRegressor)
    message = ""
    MLJBase.isdistribution(model.distribution_type) ||
        error("$model.distribution_type is not a valid distribution_type.")
    return message
end

function MLJBase.fit(::ConstantRegressor{D}, verbosity::Int, X, y) where D
    fitresult = Distributions.fit(D, y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJBase.fitted_params(::ConstantRegressor, fitresult) = (target_distribution=fitresult,)

MLJBase.predict(::ConstantRegressor, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

##
## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)
##

struct DeterministicConstantRegressor <: MLJBase.Deterministic end

function MLJBase.fit(::DeterministicConstantRegressor, verbosity::Int, X, y)
    fitresult = mean(y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJBase.predict(::DeterministicConstantRegressor, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

##
## THE CONSTANT CLASSIFIER
##

"""
ConstantClassifier()

A classifier that, for any new input pattern, `predict`s the
`UnivariateFinite` probability distribution `d` best fitting the
training target data. So, `pdf(d, level)` is the proportion of levels
in the training data coinciding with `level`. Use `predict_mode` to
obtain the training target mode instead.
"""
struct ConstantClassifier <: MLJBase.Probabilistic end

# here `args` is `y` or `y, w`:
function MLJBase.fit(::ConstantClassifier, verbosity::Int, X, y, w=nothing)
    fitresult = Distributions.fit(MLJBase.UnivariateFinite, y, w)
    cache     = nothing
    report    = NamedTuple
    return fitresult, cache, report
end

MLJBase.fitted_params(::ConstantClassifier, fitresult) = (target_distribution=fitresult,)

MLJBase.predict(::ConstantClassifier, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

##
## DETERMINISTIC CONSTANT CLASSIFIER (FOR TESTING)
##

struct DeterministicConstantClassifier <: MLJBase.Deterministic end

function MLJBase.fit(::DeterministicConstantClassifier, verbosity::Int, X, y)
    # dump missing target values and make into a regular array:
    fitresult = mode(skipmissing(y) |> collect) # a CategoricalValue or CategoricalString
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MLJBase.predict(::DeterministicConstantClassifier, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

##
## METADATA
##

metadata_pkg.((ConstantRegressor, ConstantClassifier,
               DeterministicConstantRegressor, DeterministicConstantClassifier),
              name="MLJModels",
              uuid="d491faf4-2d78-11e9-2867-c94bc002c0b7",
              url="https://github.com/alan-turing-institute/MLJModels.jl",
              julia=true,
              license="MIT",
              is_wrapper=false)

metadata_model(ConstantRegressor,
               input=MLJBase.Table(MLJBase.Scientific),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr="Constant regressor (Probabilistic).",
               path="MLJModels.ConstantRegressor")

metadata_model(DeterministicConstantRegressor,
               input=MLJBase.Table(MLJBase.Scientific),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr="Constant regressor (Deterministic).",
               path="MLJModels.DeterministicConstantRegressor")

metadata_model(ConstantClassifier,
               input=MLJBase.Table(MLJBase.Scientific),
               target=AbstractVector{<:MLJBase.Finite},
               weights=true,
               descr="Constant classifier (Probabilistic).",
               path="MLJModels.ConstantClassifier")

metadata_model(DeterministicConstantClassifier,
               input=MLJBase.Table(MLJBase.Scientific),
               target=AbstractVector{<:MLJBase.Finite},
               weights=false,
               descr="Constant classifier (Deterministic).",
               path="MLJModels.DeterministicConstantClassifier")
