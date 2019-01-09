module MLJInterface

export MLJType, Model, Supervised, Unsupervised, Deterministic, Probabilistic
export fit, update, clean!, info, coerce
export predict, predict_mean, predict_mode 
export transform, inverse_transform, se, evaluate, best

export @show, @constant                # from show.jl
export UnivariateNominal               # from distributions.jl

# methods from other packages to be rexported:
export pdf, mean, mode

import Base.==
using Query
import TableTraits
import DataFrames                # TODO: get rid of this dependency
import Distributions
import Distributions: pdf, mode
using CategoricalArrays

# from Standard Library:
using Statistics


## CONSTANTS

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 1


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# for storing hyperparameters:
abstract type Model <: MLJType end

abstract type Supervised{R} <: Model end # parameterized by fit-result type `R`
abstract type Unsupervised <: Model  end

# supervised models that `predict` probability distributions are of:
abstract type Probabilistic{R} <: Supervised{R} end

# supervsied models that `predict` point-values are of:
abstract type Deterministic{R} <: Supervised{R} end

# for displaying objects of `MLJType`:
include("show.jl") 

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

# convenience methods for manipulating categorical and tabular data
include("data.jl")


## THE MODEL INTERFACE

# every model interface must implement this method, used to generate
# fit-results:
function fit end

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)

# methods dispatched on a model and fit-result are called *operations*.
# supervised models must implement this operation:
function predict end

# probabilistic supervised models may also overload one or more of
# these operations delevering point predictions:
predict_mode(model::Supervised, fitresult, Xnew) =
    mode.(predict(model, fitresult, Xnew))
predict_mean(model::Supervised, fitresult, Xnew) =
    mean.(predict(model, fitresult, Xnew))
predict_median(model::Supervised, fitresult, Xnew) =
    median.(predict(model, fitresult, Xnew))

# unsupervised methods must implement this operation:
function transform end

# unsupervised methods may implement this operation:
function inverse_transform end

# operations implemented by some meta-models:
function se end
function evaluate end
function best end

# models buying into introspection should
# implement the following method, dispatched on model
# *type*:
function info end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(fitresult::Model) = ""

# supervised models may need to overload the following method to
# ensure iterable tables compliant input data supplied by user is coerced
# into the form required by its `fit` method and operations:
coerce(model::Model, Xtable) = Xtable

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end

end # module
