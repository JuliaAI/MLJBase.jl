# Users of this module should first read the document
# https://github.com/alan-turing-institute/MLJ.jl/blob/master/doc/adding_new_models.md 

module MLJBase

export MLJType, Model, Supervised, Unsupervised, Deterministic, Probabilistic
export selectrows, selectcols, schema, table, levels_seen
export fit, update, clean!, info
export predict, predict_mean, predict_mode 
export transform, inverse_transform, se, evaluate, best
export output_kind, output_quantity, input_kinds, input_quantity, is_pure_julia
export load_path, package_url, package_name, package_uuid
export fitresult_type

export HANDLE_GIVEN_ID, @show, @constant  # from show.jl
export UnivariateNominal, average         # from distributions.jl

# methods from other packages to be rexported:
export pdf, mean, mode

import Base.==
using Tables
import Distributions
import Distributions: pdf, mode
using CategoricalArrays
import CategoricalArrays

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

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(model::Model) = ""

# fallback trait declarations:
output_kind(::Type{<:Model}) = :unknown
output_quantity(::Type{<:Model}) = :univariate
input_kinds(::Type{<:Model}) = Symbol[]
input_quantity(::Type{<:Model}) = :multivariate
is_pure_julia(::Type{<:Model}) = :unknown
package_name(::Type{<:Model}) = "unknown"
load_path(M::Type{<:Model}) = "unknown"
package_uuid(::Type{<:Model}) = "unknown"
package_url(::Type{<:Model}) = "unknown"

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

include("introspection.jl")

# returns the fit-result type declared by a supervised model
# declaration (to test actual fit-results have the declared type):
"""
    MLJBase.fitresult_type(m)

Returns the fitresult type of any supervised model (or model type)
`m`, as declared in the model `mutable struct` declaration.

"""
fitresult_type(M::Type{<:Supervised}) = supertype(M).parameters[1]
fitresult_type(m::Supervised) = fitresult_type(typeof(m)) end # module
