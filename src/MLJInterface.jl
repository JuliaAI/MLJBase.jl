module MLJInterface

export Supervised, Unsupervised
export Names

export predict, predict_mean, predict_mode
export transform, inverse_transform, se, evaluate, best

# methods from other packages to be rexported:
export pdf, mean, mode

import Base.==
using Query
import TableTraits
import DataFrames
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


# probability distributions not provided by Distributions.jl package:
include("distributions.jl")

# for displaying objects of `MLJType`:
include("show.jl") 

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

# supervised models that predict probabilities by default may also
# implement one of these operations to do "point" predictions:
function predict_mode end # classifiers
function predict_mean end # regressors
# TODO: provide fallbacks for these

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


## LOAD VARIOUS INTERFACE COMPONENTS

include("machines.jl")
include("networks.jl")
include("composites.jl")
include("operations.jl")
include("resampling.jl")
include("parameters.jl")
include("tuning.jl")


## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")


## SETUP LAZY PKG INTERFACE LOADING

# note: presently an MLJ interface to a package, eg `DecisionTree`,
# is not loaded by `using MLJ` alone; one must additionally call
# `import DecisionTree`.

# files containing a pkg interface must have same name as pkg plus ".jl"

macro load_interface(pkgname, uuid::String, load_instr)
    (load_instr.head == :(=) && load_instr.args[1] == :lazy) ||
        throw(error("Invalid load instruction"))
    lazy = load_instr.args[2]
    filename = joinpath("interfaces", string(pkgname, ".jl"))

    if lazy
        quote
            @require $pkgname=$uuid include($filename)
        end
    else
        quote
            @eval include(joinpath($srcdir, $filename))
        end
    end
end

function __init__()
    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
    @load_interface  MultivariateStats "6f286f6a-111f-5878-ab1e-185364afe411" lazy=true
end

#@load_interface XGBoost "009559a3-9522-5dbb-924b-0b6ed2b22bb9" lazy=false

end # module
