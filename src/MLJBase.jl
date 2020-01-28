module MLJBase

# ===================================================================
## METHOD EXPORT

# defined in this file:
export MLJType, Model, Supervised, Unsupervised, Static,
    Deterministic, Probabilistic, Interval,
    DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork,
    fit, update, update_data, clean!,
    predict, predict_mean, predict_mode, predict_median, fitted_params,
    transform, inverse_transform, evaluate, best, @load

# computational_resources.jl:
export default_resource

# equality.jl:
export is_same_except

# one_dimensional_ranges.jl:
export ParamRange, NumericRange, NominalRange, iterator, scale

# model_traits.jl:
export load_path, package_url, package_name, package_uuid,
    input_scitype, supports_weights,
    target_scitype, output_scitype,
    is_pure_julia, is_wrapper, prediction_type

# parameter_inspection.jl:
export params # note this is *not* an extension of StatsBase.params

# data.jl:
export reconstruct, int, decoder, classes,
    selectrows, selectcols, select, nrows,
    table, levels_seen, matrix, container_type,
    partition, unpack,
    complement, restrict, corestrict

# utilities.jl:
export @set_defaults, flat_values,
    recursive_setproperty!, recursive_getproperty, pretty, unwind

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
    evaluate!, Resampler, PerformanceEvaluation

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

# ===================================================================
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

# ===================================================================
## METHOD IMPORT

import Base: ==, precision, getindex, setindex!, @__doc__

# Scitype
using MLJModelInterface
using MLJScientificTypes
using ScientificTypes

# Containers
using Tables, PrettyTables
using DelimitedFiles
using OrderedCollections
using CategoricalArrays
using LossFunctions
import InvertedIndices: Not

using Distributed
using ComputationalResources
using ComputationalResources: CPUProcesses
using ProgressMeter

# to be extended:
import StatsBase
import StatsBase: fit, predict, fit!, mode, countmap
import Missings.levels
import Distributions
import Distributions: pdf

# from Standard Library:
using Statistics, LinearAlgebra, Random, InteractiveUtils

# ===================================================================
## CONSTANTS

# aliases
const MST = MLJScientificTypes
const MMI = MLJModelInterface
const ST  = ScientificTypes

# the directory containing this file:
const srcdir = dirname(@__FILE__)
# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 0
const DEFAULT_AS_CONSTRUCTED_SHOW_DEPTH = 2
const INDENT = 4

## INCLUDE FILES

include("utilities.jl")

# what new models must implement:
include("model_interface.jl")

# for inspecting (possibly nested) fields of MLJ objects:
include("parameter_inspection.jl")

# equality for `MLJType` objects
include("equality.jl")

# for displaying objects of `MLJType`:
include("show.jl")

# methods to inspect/change default computational resource (mode of
# parallelizaion):
include("computational_resources.jl")

# hyperparameter ranges (domains):
include("one_dimensional_ranges.jl")
include("one_dimensional_range_methods.jl")

# model trait fallbacks
include("model_traits.jl")

# convenience methods for manipulating categorical and tabular data
include("data.jl")

# metadata utils:
include("metadata_utilities.jl")

# macro to streamline model definitions
include("mlj_model_macro.jl")

# probability distributions and methods not provided by
# Distributions.jl package:
include("distributions.jl")

# assembles model traits into dictionaries:
include("info_dict.jl")

# datasets:
include("datasets.jl")
include("datasets_synthetic.jl")

# to be depreciated:
include("tasks.jl")

# scores, losses, etc:
include("measures/measures.jl")
include("measures/registry.jl")

include("pipeline_static.jl")  # static transformer needed by pipeline.jl
include("machines.jl")
include("networks.jl")

# overloading predict, transform, etc, to work with machines:
include("operations.jl")

# building and exporting learning networks:
include("composites.jl")

# macro for non-branching exported networks:
include("pipelines.jl")

# arrow syntax for constructing learning networks:
VERSION ≥ v"1.3.0-" && include("arrows.jl")

# resampling (Holdout, CV, etc)
include("resampling.jl")

include("init.jl")

end # module
