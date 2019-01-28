# Users of this module should first read the document
# https://github.com/alan-turing-institute/MLJ.jl/blob/master/doc/adding_new_models.md 

module MLJBase

export MLJType, Model, Supervised, Unsupervised, Deterministic, Probabilistic
export Rows, Cols, Schema, selectrows, selectcols, schema, table, getrows
export fit, update, clean!, info, coerce
export predict, predict_mean, predict_mode 
export transform, inverse_transform, se, evaluate, best
export target_kind, target_quantity, inputs_can_be, is_pure_julia
export package_url, package_name, package_uuid

export HANDLE_GIVEN_ID, @show, @constant  # from show.jl
export UnivariateNominal, average         # from distributions.jl

# methods from other packages to be rexported:
export pdf, mean, mode

import Base.==
using Tables
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

# supervised models may need to overload the following method to
# ensure the input data supplied by user (something implementing the
# Queryverse iterable table API) is coerced into the form required by
# its `fit` method and operations:
coerce(model::Model, Xtable) = Xtable

# if the return type `TABLE` of `coerce` is not a Tables.jl-compatible
# table or an `AbstractMatrix` (with rows corresponding to patterns and columns
# corresponding to features) then users will not be able to use MLJ's
# performant `EnsembleModel` on `model` unless one overloads the
# following method for type `TABLE`:
getrows(model::Model, X, r) = selectrows(X, r)   
# here `r` is any integer, unitrange or colon `:`, and the right-hand
# side is defined in `data.jl`.

# fallback trait declarations:
target_kind(::Type{<:Supervised}) = :unknown
target_quantity(::Type{<:Supervised}) = :univariate
inputs_can_be(::Type{<:Model}) = Symbol[]
is_pure_julia(::Type{<:Model}) = :unknown
package_name(::Type{<:Model}) = "unknown"
package_uuid(::Type{<:Model}) = "unknown"
package_url(::Type{<:Model}) = "unknown"

_response(::Type{<:Supervised}) = :unknown
_response(::Type{<:Deterministic}) = :deterministic
_response(::Type{<:Probabilistic}) = :probabilistic

target_is(modeltype::Type{<:Supervised}) =
    [_response(modeltype), target_kind(modeltype), target_quantity(modeltype)]


if VERSION < v"1.0.0"
    import Base.info
end

function info(modeltype::Type{<:Model})

    message = "$modeltype has a bad trait declaration."

    if modeltype <: Supervised
        target_kind(modeltype) in [:numeric, :binary, :multiclass, :unknown] ||
            error(message*"target_kind must return :numeric, :binary, :multiclass (or :unknown).")
        target_quantity(modeltype) in [:univariate, :multivariate] ||
            error(message*"target_quantity must return :univariate or :multivariate")
    end
    
    issubset(Set(inputs_can_be(modeltype)), Set([:numeric, :nominal, :missing])) ||
        error(message*"inputs_can_be must return a vector with entries from [:numeric, :nominal, :missing]")
    is_pure_julia(modeltype) in [:yes, :no, :unknown] ||
        error(message*"is_pure_julia must return :yes, :no (or :unknown).")

    # modelnamesplit = split(string(modeltype.name), '.')
    # if length(modelnamesplit) > 1
    #     modelnamesplit = modelnamesplit[2:end]
    # end
    # modelname = string(reduce((a, b)->"$a.$b", modelnamesplit))

    d = Dict{Symbol,Union{Symbol,Vector{Symbol},String}}()
    d[:model_name] = string(modeltype)
    d[:inputs_can_be] = inputs_can_be(modeltype)
    d[:is_pure_julia] = is_pure_julia(modeltype)
    d[:package_name] = package_name(modeltype)
    d[:package_uuid] = package_uuid(modeltype)
    d[:package_url] = package_url(modeltype)
    if modeltype <: Supervised
        d[:target_is] = target_is(modeltype)
    end
    
    return d
end
info(model::Supervised) = info(typeof(model))

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end






end # module
