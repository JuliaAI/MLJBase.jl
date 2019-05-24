# Users of this module should first read the document
# https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/

module MLJBase

export MLJType, Model, Supervised, Unsupervised, Deterministic, Probabilistic
export DeterministicNetwork, ProbabilisticNetwork
export fit, update, clean!
export predict, predict_mean, predict_mode, fitted_params
export transform, inverse_transform, se, evaluate, best
export load_path, package_url, package_name, package_uuid
export input_scitype_union, input_is_multivariate       
export target_scitype_union, target_quantity            
export is_pure_julia, is_wrapper                                 

export params                                        # parameters.jl
export reconstruct, int, decoder, classes            # data.jl
export selectrows, selectcols, select, nrows, schema # data.jl
export table, levels_seen, matrix, container_type    # data.jl
export partition,StratifiedKFold                     # utilities.jl
export Found, Continuous, Finite, Infinite           # scitypes.jl
export OrderedFactor, Unknown                        # scitypes.jl
export Count, Multiclass, Binary                     # scitypes.jl
export scitype, scitype_union, scitypes              # scitypes.jl
export HANDLE_GIVEN_ID, @more, @constant             # show.jl
export color_on, color_off                           # show.jl
export UnivariateFinite, average                    # distributions.jl
export SupervisedTask, UnsupervisedTask, MLJTask     # tasks.jl
export X_and_y, X_, y_, nrows, nfeatures             # tasks.jl
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

# to be extended:
import StatsBase: fit, predict, fit!

# from Standard Library:
using Statistics
using Random
using InteractiveUtils
using SparseArrays

## CONSTANTS

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 0

include("utilities.jl")
include("scitypes.jl")


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# for storing hyperparameters:
abstract type Model <: MLJType end

abstract type Supervised <: Model end 
abstract type Unsupervised <: Model end

# supervised models that `predict` probability distributions are of:
abstract type Probabilistic <: Supervised end

# supervised models that `predict` point-values are of:
abstract type Deterministic <: Supervised end

# for models that are "exported" learning networks (return a Node as
# their fit-result; see MLJ/networks.jl):
abstract type ProbabilisticNetwork <: Probabilistic end
abstract type DeterministicNetwork <: Deterministic end

# by default, MLJType objects are `==` if: (i) they have a common
# supertype AND (ii) they have the same set of defined fields AND
# (iii) their defined field values are `==`:
function ==(m1::M1, m2::M2) where {M1<:MLJType,M2<:MLJType}
    if M1 != M1
        return false
    end
    defined1 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m1, fld)
    end
    defined2 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m2, fld)
    end
    if defined1 != defined2
        return false
    end
    same_values = true
    for fld in defined1
        same_values = same_values && getfield(m1, fld) == getfield(m2, fld)
    end
    return same_values
end


## THE MODEL INTERFACE

# every model interface must implement a `fit` method of the form
# `fit(model, verbosity, X, y) -> fitresult, cache, report` or
# `fit(model, verbosity, X, ys...) -> fitresult, cache, report` (multivariate case)
# or, one the simplified versions
# `fit(model, X, y) -> fitresult`
# `fit(model, X, ys...) -> fitresult`
fit(model::Model, verbosity::Integer, args...) = fit(model, args...), nothing, nothing

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)

# methods dispatched on a model and fit-result are called
# *operations*.  supervised models must implement a `predict`
# operation (extending the `predict` method of StatsBase).

# unsupervised methods must implement this operation:
function transform end

# unsupervised methods may implement this operation:
function inverse_transform end

# this operation can be optionally overloaded to provide access to
# fitted parameters (eg, coeficients of linear model):
fitted_params(::Model, fitresult) = (fitresult=fitresult,)

# operations implemented by some meta-models:
function se end
function evaluate end
function best end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(model::Model) = ""

# fallback trait declarations:
target_scitype_union(::Type{<:Supervised}) =
    Union{Found,NTuple{N,Found}} where N # a Tuple type in multivariate case
output_scitype_union(::Type{<:Unsupervised}) =
    Union{Missing,Found}                 
output_is_multivariate(::Type{<:Unsupervised}) = true
input_scitype_union(::Type{<:Model}) = Union{Missing,Found}
input_is_multivariate(::Type{<:Model}) = true 
is_pure_julia(::Type{<:Model}) = false
package_name(::Type{<:Model}) = "unknown"
load_path(M::Type{<:Model}) = "unknown"
package_uuid(::Type{<:Model}) = "unknown"
package_url(::Type{<:Model}) = "unknown"
is_wrapper(::Type{<:Model}) = false
is_wrapper(m::Model) = is_wrapper(typeof(m))

target_scitype_union(model::Model) = target_scitype_union(typeof(model))
input_scitype_union(model::Model) = input_scitype_union(typeof(model))
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
    mode.(predict(model, fitresult, Xnew))

# mean:
predict_mean(model::Probabilistic, fitresult, Xnew) =
    mean.(predict(model, fitresult, Xnew))

# median:
predict_median(model::Probabilistic, fitresult, Xnew) =
    median.(predict(model, fitresult, Xnew))

# for unpacking the fields of MLJ objects:
include("parameters.jl")

# for displaying objects of `MLJType`:
include("show.jl") 

# convenience methods for manipulating categorical and tabular data
include("data.jl")

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

include("info.jl")
include("tasks.jl")
include("datasets.jl")

end # module

