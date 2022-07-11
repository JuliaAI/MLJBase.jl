## THE CONSTANT REGRESSOR

export ConstantClassifier, ConstantRegressor,
       DeterministicConstantClassifier,
       ProbabilisticConstantClassifer,
       DeterministicConstantRegressor

import Distributions

"""
ConstantRegressor(; distribution_type=Distributions.Normal)

A regressor that, for any new input pattern, predicts the univariate
probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.
"""
struct ConstantRegressor{D} <: MMI.Probabilistic end

function ConstantRegressor(; distribution_type=Distributions.Normal)
    model   = ConstantRegressor{distribution_type}()
    message = clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::ConstantRegressor{D}) where D
    message = ""
    D <: Distributions.Sampleable ||
        error("$model.distribution_type is not a valid distribution_type.")
    return message
end

MMI.reformat(::ConstantRegressor, X) = (MMI.matrix(X),)
MMI.reformat(::ConstantRegressor, X, y) = (MMI.matrix(X), y)
MMI.selectrows(::ConstantRegressor, I, A) = (view(A, I, :),)
MMI.selectrows(::ConstantRegressor, I, A, y) = (view(A, I, :), y[I])

function MMI.fit(::ConstantRegressor{D}, verbosity::Int, A, y) where D
    fitresult = Distributions.fit(D, y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MMI.fitted_params(::ConstantRegressor, fitresult) =
    (target_distribution=fitresult,)

MMI.predict(::ConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

##
## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)
##

struct DeterministicConstantRegressor <: MMI.Deterministic end

function MMI.fit(::DeterministicConstantRegressor, verbosity::Int, X, y)
    fitresult = mean(y)
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MMI.reformat(::DeterministicConstantRegressor, X) = (MMI.matrix(X),)
MMI.reformat(::DeterministicConstantRegressor, X, y) = (MMI.matrix(X), y)
MMI.selectrows(::DeterministicConstantRegressor, I, A) = (view(A, I, :),)
MMI.selectrows(::DeterministicConstantRegressor, I, A, y) =
    (view(A, I, :), y[I])

MMI.predict(::DeterministicConstantRegressor, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

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
mutable struct ConstantClassifier <: MMI.Probabilistic
    testing::Bool
    bogus::Int
end

ConstantClassifier(; testing=false, bogus=0) =
    ConstantClassifier(testing, bogus)

function MMI.reformat(model::ConstantClassifier, X)
    model.testing && @info "reformatting X"
    return (MMI.matrix(X),)
end

function MMI.reformat(model::ConstantClassifier, X, y)
    model.testing && @info "reformatting X, y"
    return (MMI.matrix(X), y)
end

function MMI.reformat(model::ConstantClassifier, X, y, w)
    model.testing && @info "reformatting X, y, w"
    return (MMI.matrix(X), y, w)
end

function MMI.selectrows(model::ConstantClassifier, I, A)
    model.testing && @info "resampling X"
    return (view(A, I, :),)
end

function MMI.selectrows(model::ConstantClassifier, I, A, y)
    model.testing && @info "resampling X, y"
    return (view(A, I, :), y[I])
end

function MMI.selectrows(model::ConstantClassifier, I, A, y, ::Nothing)
    model.testing && @info "resampling X, y, nothing"
    return (view(A, I, :), y[I], nothing)
end

function MMI.selectrows(model::ConstantClassifier, I, A, y, w)
    model.testing && @info "resampling X, y, nothing"
    return (view(A, I, :), y[I], w[I])
end

# here `args` is `y` or `y, w`:
function MMI.fit(::ConstantClassifier, verbosity::Int, A, y, w=nothing)
    fitresult = Distributions.fit(MLJBase.UnivariateFinite, y, w)
    cache     = nothing
    report    = NamedTuple
    return fitresult, cache, report
end

MMI.fitted_params(::ConstantClassifier, fitresult) =
    (target_distribution=fitresult,)

MMI.predict(::ConstantClassifier, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

##
## DETERMINISTIC CONSTANT CLASSIFIER (FOR TESTING)
##

struct DeterministicConstantClassifier <: MMI.Deterministic end

function MMI.fit(::DeterministicConstantClassifier, verbosity::Int, X, y)
    # dump missing target values and make into a regular array:
    fitresult = mode(skipmissing(y) |> collect) # a CategoricalValue
    cache     = nothing
    report    = NamedTuple()
    return fitresult, cache, report
end

MMI.reformat(::DeterministicConstantClassifier, X) = (MMI.matrix(X),)
MMI.reformat(::DeterministicConstantClassifier, X, y) = (MMI.matrix(X), y)
MMI.selectrows(::DeterministicConstantClassifier, I, A) = (view(A, I, :),)
MMI.selectrows(::DeterministicConstantClassifier, I, A, y) =
    (view(A, I, :), y[I])

MMI.predict(::DeterministicConstantClassifier, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

#
# METADATA
#

metadata_pkg.((ConstantRegressor, ConstantClassifier,
               DeterministicConstantRegressor, DeterministicConstantClassifier),
              name="MLJModels",
              uuid="d491faf4-2d78-11e9-2867-c94bc002c0b7",
              url="https://github.com/alan-turing-institute/MLJModels.jl",
              julia=true,
              license="MIT",
              is_wrapper=false)

metadata_model(ConstantRegressor,
               input=MMI.Table,
               target=AbstractVector{MMI.Continuous},
               weights=false,
               path="MLJModels.ConstantRegressor")

metadata_model(DeterministicConstantRegressor,
               input=MMI.Table,
               target=AbstractVector{MMI.Continuous},
               weights=false,
               path="MLJModels.DeterministicConstantRegressor")

metadata_model(ConstantClassifier,
               input=MMI.Table,
               target=AbstractVector{<:MMI.Finite},
               weights=true,
               path="MLJModels.ConstantClassifier")

metadata_model(DeterministicConstantClassifier,
               input=MMI.Table,
               target=AbstractVector{<:MMI.Finite},
               weights=false,
               path="MLJModels.DeterministicConstantClassifier")
