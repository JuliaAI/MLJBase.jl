# Users of this module should first read the document
# https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/
module MLJBase

export MLJType, Model, Supervised, Unsupervised, Static
export Deterministic, Probabilistic, Interval
export DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork
export fit, update, update_data, clean!
export predict, predict_mean, predict_mode, predict_median, fitted_params
export transform, inverse_transform, se, evaluate, best
export info, info_dict
export is_same_except

export load_path, package_url, package_name, package_uuid  # model_traits.jl
export input_scitype, supports_weights                     # model_traits.jl
export target_scitype, output_scitype                      # model_traits.jl
export is_pure_julia, is_wrapper, prediction_type          # model_traits.jl
export params                                        # parameters.jl
export reconstruct, int, decoder, classes            # data.jl
export selectrows, selectcols, select, nrows         # data.jl
export table, levels_seen, matrix, container_type    # data.jl
export partition, unpack                             # data.jl
export complement, restrict, corestrict              # data.jl
export @set_defaults                                 # utilities.jl
export @mlj_model                                    # mlj_model_macro.jl
export metadata_model, metadata_pkg                  # metadata_utilities
export HANDLE_GIVEN_ID, @more, @constant             # show.jl
export color_on, color_off                           # show.jl
export UnivariateFinite, average                     # distributions.jl
export SupervisedTask, UnsupervisedTask, MLJTask     # tasks.jl
export X_and_y, X_, y_, nrows, nfeatures             # tasks.jl
export info                                          # info.jl
export load_boston, load_ames, load_iris,
       load_reduced_ames, load_crabs,
       @load_boston, @load_ames, @load_iris,
       @load_reduced_ames, @load_crabs               # datasets.jl
export @load

# MEASURES
export measures # measures/registry.jl
export orientation, reports_each_observation
export is_feature_dependent, aggregation
export aggregate
export default_measure, value
# -- continuous
export mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2
# -- confmat (measures/confusion_matrix)
export confusion_matrix, confmat
# -- finite (measures/finite)
export cross_entropy, BrierScore,
       misclassification_rate, mcr, accuracy,
       balanced_accuracy, bacc, bac,
       matthews_correlation, mcc
# -- -- binary // order independent
export auc, roc_curve, roc
# -- -- binary // order dependent
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
       TruePositiveRate, TrueNegativeRate, FalsePositiveRate, FalseNegativeRate,
       FalseDiscoveryRate, Precision, NPV, FScore,
       # standard synonyms
       TPR, TNR, FPR, FNR,
       FDR, PPV,
       Recall, Specificity, BACC,
       # defaults and their synonyms
       truepositive, truenegative, falsepositive, falsenegative,
       truepositive_rate, truenegative_rate, falsepositive_rate,
       falsenegative_rate, negativepredicitive_value,
       positivepredictive_value,
       tp, tn, fp, fn, tpr, tnr, fpr, fnr,
       falsediscovery_rate, fdr, npv, ppv,
       recall, sensitivity, hit_rate, miss_rate,
       specificity, selectivity, f1score, f1, fallout

# methods from other packages to be rexported:
export pdf, mean, mode

# re-export of ScientificTypes (`Table` not exported):
export trait
export Scientific, Found, Unknown, Finite, Infinite
export OrderedFactor, Multiclass, Count, Continuous
export Binary, ColorImage, GrayImage, Image
export scitype, scitype_union, coerce, schema, elscitype

# rexport from Random, Statistics, Distributions, CategoricalArrays,
# InvertedIndices:
export pdf, mode, median, mean, shuffle!, categorical, shuffle, levels, levels!
export std, Not

import Base.==, Base.precision, Base.getindex
import Base: @__doc__

using Tables, DelimitedFiles
using OrderedCollections # already a dependency of StatsBase
using CategoricalArrays
import InvertedIndices: Not

# to be extended:
import StatsBase: fit, predict, fit!
import Missings.levels
import Distributions
import Distributions: pdf, mode

using ScientificTypes
using LossFunctions

# from Standard Library:

using Statistics, LinearAlgebra, Random, InteractiveUtils

## CONSTANTS

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 0

include("utilities.jl")

## BASE TYPES

abstract type MLJType end
include("equality.jl") # equality for MLJType objects

## ABSTRACT MODEL TYPES

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


## THE MODEL INTERFACE

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
function se end
function evaluate end
function best end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(model::Model) = ""


## STUB FOR @load (extended by MLJModels)

macro load end


## TRAITS

"""

    info(object)

List the traits of an object, such as a model or a performance measure.

"""
info(object) = info(object, Val(ScientificTypes.trait(object)))


include("model_traits.jl")

# for unpacking the fields of MLJ objects:
include("parameters.jl")

# for displaying objects of `MLJType`:
include("show.jl")

# convenience methods for manipulating categorical and tabular data
include("data.jl")

# metadata utils
include("metadata_utilities.jl")

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

include("info.jl")
include("datasets.jl")
include("tasks.jl")
include("measures/measures.jl")
include("measures/registry.jl")

# mlj model macro to help define models
include("mlj_model_macro.jl")

function __init__()
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:supervised_model] =
        x-> x isa Supervised
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:unsupervised_model] =
        x-> x isa Unsupervised
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure] = is_measure
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type
end

end # module
