module MLJBase


## METHOD EXPORT

# defined in this file:
export MLJType, Model, Supervised, Unsupervised, Static,
    Deterministic, Probabilistic, Interval,
    DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork,
    fit, update, update_data, clean!,
    predict, predict_mean, predict_mode, predict_median, fitted_params,
    transform, inverse_transform, se, evaluate, best, @load

# computational_resources.jl:
export default_resource

# equality.jl:
export is_same_except

# model_traits.jl:
export load_path, package_url, package_name, package_uuid,
    input_scitype, supports_weights,
    target_scitype, output_scitype,
    is_pure_julia, is_wrapper, prediction_type

# parameters.jl:
export params # note this is *not* an extension of StatsBase.params

# data.jl:
export reconstruct, int, decoder, classes,
    selectrows, selectcols, select, nrows,
    table, levels_seen, matrix, container_type,
    partition, unpack,
    complement, restrict, corestrict

# utilities.jl:
export @set_defaults, flat_values,
    recursive_setproperty!, recursive_getproperty, pretty

# mlj_model_macro.jl
export @mlj_model

# metadata_utilities:
export metadata_model, metadata_pkg

# show.jl
export HANDLE_GIVEN_ID, @more, @constant, color_on, color_off

# distributions.jl:
export UnivariateFinite, average

# tasks.jl:
export SupervisedTask, UnsupervisedTask, MLJTask,
    X_and_y, X_, y_, nfeatures

# info_dict.jl:
export info_dict

# datasets.jl:
export load_boston, load_ames, load_iris,
    load_reduced_ames, load_crabs,
    @load_boston, @load_ames, @load_iris,
    @load_reduced_ames, @load_crabs

# machines.jl:
export machine, Machine, AbstractMachine, fit!, report

# networks.jl:
export NodalMachine,  machines, source, node,sources, origins,
    rebind!, nodes, freeze!, thaw!, models, Node, AbstractNode, Source

# datasets_synthetics.jl
export make_blobs, make_moons, make_circles,
       make_regression

# composites.jl:
export machines, sources, anonymize!, @from_network, fitresults

# pipelines.jl:
export @pipeline

# resampling.jl:
export ResamplingStrategy, Holdout, CV, StratifiedCV,
    evaluate!, Resampler

# measures/registry.jl:
export measures

# measures.jl:
export orientation, reports_each_observation,
    is_feature_dependent, aggregation,
    aggregate,
    default_measure, value

# measures/continuous.jl:
export mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2

# measures/confusion_matrix.jl:
export confusion_matrix, confmat

# measures/finite.jl
export cross_entropy, BrierScore,
    misclassification_rate, mcr, accuracy,
    balanced_accuracy, bacc, bac,
    matthews_correlation, mcc

# measures/finite.jl -- binary order independent:
export auc, roc_curve, roc

# measures/finite.jl -- binary order dependent:
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
    TruePositiveRate, TrueNegativeRate, FalsePositiveRate, FalseNegativeRate,
    FalseDiscoveryRate, Precision, NPV, FScore,
    # standard synonyms
    TPR, TNR, FPR, FNR,
    FDR, PPV,
    Recall, Specificity, BACC,
    # instances and their synonyms
    truepositive, truenegative, falsepositive, falsenegative,
    truepositive_rate, truenegative_rate, falsepositive_rate,
    falsenegative_rate, negativepredicitive_value,
    positivepredictive_value,
    tp, tn, fp, fn, tpr, tnr, fpr, fnr,
    falsediscovery_rate, fdr, npv, ppv,
    recall, sensitivity, hit_rate, miss_rate,
    specificity, selectivity, f1score, f1, fallout


## METHOD RE-EXPORT

# re-export from ScientificTypes (`Table` not exported):
export trait, Scientific, Found, Unknown, Finite, Infinite,
    OrderedFactor, Multiclass, Count, Continuous,
    Binary, ColorImage, GrayImage, Image,
    scitype, scitype_union, coerce, coerce!,
    schema, elscitype, info

# re-xport from Random, StatsBase, Statistics, Distributions, CategoricalArrays,
# InvertedIndices:
export pdf, mode, median, mean, shuffle!, categorical, shuffle,
    levels, levels!, std, Not


## METHOD IMPORT

import Base: ==, precision, getindex, setindex!, @__doc__

using Tables
using DelimitedFiles
using OrderedCollections
using CategoricalArrays
using ScientificTypes
using LossFunctions
import InvertedIndices: Not
using Distributed
using ComputationalResources
using ComputationalResources: CPUProcesses
using ProgressMeter
using PrettyTables

# to be extended:
import StatsBase
import StatsBase: fit, predict, fit!, mode, countmap
import Missings.levels
import Distributions
import Distributions: pdf
import ScientificTypes.info

# from Standard Library:
using Statistics, LinearAlgebra, Random, InteractiveUtils


## CONSTANTS

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 0
const DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

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


## THE REST

# methods to inspect/change default computational resource (mode of
# parallelizaion):
include("computational_resources.jl")

# model trait fallbacks
include("model_traits.jl")

# for unpacking the fields of MLJ objects:
include("parameters.jl")

# for displaying objects of `MLJType`:
include("show.jl")

# convenience methods for manipulating categorical and tabular data
include("data.jl")

include("metadata_utilities.jl") # metadata utils
include("mlj_model_macro.jl")    # macro to streamline model definitions

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

include("info_dict.jl")
include("datasets.jl")
include("datasets_synthetic.jl")
include("tasks.jl")
include("measures/measures.jl")
include("measures/registry.jl")
include("pipeline_static.jl")  # static transformer needed by pipeline.jl
include("machines.jl")
include("networks.jl")
include("operations.jl") # overloading predict, transform, etc
include("composites.jl") # building and exporting learning networks
include("pipelines.jl")
VERSION ≥ v"1.3.0-" && include("arrows.jl")
include("resampling.jl")

function __init__()
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure] = is_measure
    ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type
end

end # module
