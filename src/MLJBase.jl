# Users of this module should first read the document
# https://github.com/alan-turing-institute/MLJ.jl/blob/master/doc/adding_new_models.md 

module MLJBase

export MLJType, Model, Supervised, Unsupervised, Deterministic, Probabilistic
export selectrows, selectcols, schema, table, levels_seen
export fit, update, clean!
export predict, predict_mean, predict_mode 
export transform, inverse_transform, se, evaluate, best
export target_scitype, target_quantity, input_scitypes, input_is_multivariate, is_pure_julia
export load_path, package_url, package_name, package_uuid
export fitresult_type

export partition                                     # utilities.jl
export Found, Continuous, Discrete, OrderedFactor    # scitypes.jl
export FiniteOrderedFactor                           # scitypes.jl
export Count, Multiclass, Binary                     # scitypes.jl
export scitype
export HANDLE_GIVEN_ID, @more, @constant             # show.jl
export UnivariateNominal, average                    # distributions.jl
export SupervisedTask, UnsupervisedTask, MLJTask        # tasks.jl
export X_and_y, X_, y_                               # tasks.jl
export load_boston, load_ames, load_iris             # datasets.jl
export load_reduced_ames                             # datasets.jl
export load_crabs, datanow                           # datasets.jl
export info                                          # info.jl

# methods from other packages to be rexported:
export pdf, mean, mode

import Base.==

using Tables
import Distributions
import Distributions: pdf, mode
using CategoricalArrays
import CategoricalArrays
import CSV
using DataFrames # remove ultimately

# from Standard Library:
using Statistics
using Random
using InteractiveUtils


## CONSTANTS

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 1

include("utilities.jl")
include("scitypes.jl")


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# for storing hyperparameters:
abstract type Model <: MLJType end

abstract type Supervised{R} <: Model end # parameterized by fit-result type `R`
abstract type Unsupervised <: Model  end

# supervised models that `predict` probability distributions are of:
abstract type Probabilistic{R} <: Supervised{R} end

# supervised models that `predict` point-values are of:
abstract type Deterministic{R} <: Supervised{R} end


## THE MODEL INTERFACE

# every model interface must implement a `fit` method of the form
# `fit(model, verbosity, X, y) -> fitresult, cache, report` or
# `fit(model, verbosity, X, ys...) -> fitresult, cache, report` (multivariate case)
# or, one the simplified versions
# `fit(model, X, y) -> fitresult`
# `fit(model, X, ys...) -> fitresult`
fit(model::Model, verbosity::Int, args...) = fit(model, args...), nothing, nothing

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)

# methods dispatched on a model and fit-result are called *operations*.
# supervised models must implement this operation:
function predict end

# unsupervised methods must implement this operation:
function transform end

# unsupervised methods may implement this operation:
function inverse_transform end

# optional operation providing access to fitted parameters (eg,
# coeficients of linear model):
function fitted_params end

# operations implemented by some meta-models:
function se end
function evaluate end
function best end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(model::Model) = ""

# fallback trait declarations:
target_scitype(::Type{<:Supervised}) = Other    # a Tuple type in multivariate case
output_scitypes(::Type{<:Unsupervised}) = Other # never a Tuple type
output_is_multivariate(::Type{<:Unsupervised}) = true
input_scitypes(::Type{<:Model}) = Other 
input_is_multivariate(::Type{<:Model}) = true 
is_pure_julia(::Type{<:Model}) = false
package_name(::Type{<:Model}) = "unknown"
load_path(M::Type{<:Model}) = "unknown"
package_uuid(::Type{<:Model}) = "unknown"
package_url(::Type{<:Model}) = "unknown"

target_scitype(model::Model) = target_scitype(typeof(model))
input_scitypes(model::Model) = input_scitypes(typeof(model))
input_is_multivariate(model::Model) = input_is_multivariate(typeof(model))
is_pure_julia(model::Model) = is_pure_julia(typeof(model))
package_name(model::Model) = package_name(typeof(model))
load_path(model::Model) = load_path(typeof(model))
package_uuid(model::Model) = package_uuid(typeof(model))
package_url(model::Model) = package_url(typeof(model))

# probabilistic supervised models may also overload one or more of
# `predict_mode`, `predict_median` and `predict_mean` defined below.

# mode:
predict_mode(model::Probabilistic, fitresult, Xnew) =
    predict_mode(model, fitresult, target_scitype(model), Xnew)
function predict_mode(model::Probabilistic, fitresult, ::Type{<:Union{Multiclass,OrderedFactor}}, Xnew)
    prob_predictions = predict(model, fitresult, Xnew)
    null = categorical(levels(prob_predictions[1]))[1:0] # empty cat vector with all levels
    modes = categorical(mode.(prob_predictions))
    return vcat(null, modes)
end
predict_mode(model::Probabilistic, fitresult, ::Type{<:Count}, Xnew) =
    mode.(predict(model, fitresult, Xnew))

# mean:
predict_mean(model::Probabilistic, fitresult, Xnew) =
    mean.(predict(model, fitresult, Xnew))

# median:
predict_median(model::Probabilistic, fitresult, Xnew) =
    predict_median(model, fitresult, target_scitype(model), Xnew)
function predict_median(model::Probabilistic, fitresult, ::Type{<:OrderedFactor}, Xnew) 
    prob_predictions = predict(model, fitresult, Xnew)
    null = categorical(levels(prob_predictions[1]))[1:0] # empty cat vector with all levels
    medians = categorical(median.(prob_predictions))
    return vcat(null, medians)
end
predict_median(model::Probabilistic, fitresult, ::Type{<:Union{Continuous, Count}}, Xnew) =
    median.(predict(model, fitresult, Xnew))

# returns the fit-result type declared by a supervised model
# declaration (to test actual fit-results have the declared type):
"""
    MLJBase.fitresult_type(m)

Returns the fitresult type of any supervised model (or model type)
`m`, as declared in the model `mutable struct` declaration.

"""
fitresult_type(M::Type{<:Supervised}) = supertype(M).parameters[1]
fitresult_type(m::Supervised) = fitresult_type(typeof(m))

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end

# for displaying objects of `MLJType`:
include("show.jl") 

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

# convenience methods for manipulating categorical and tabular data
include("data.jl")

include("info.jl")
include("tasks.jl")
include("datasets.jl")

end # module

