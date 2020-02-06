module MLJBase

# ===================================================================
# IMPORTS

import Base: ==, precision, getindex, setindex!
import Base.+, Base.*

# Scitype
import ScientificTypes: TRAIT_FUNCTION_GIVEN_NAME
using MLJScientificTypes
using MLJModelInterface
import MLJModelInterface: fit, update, update_data, transform,
                          inverse_transform, fitted_params, predict,
                          predict_mode, predict_mean, predict_median,
                          evaluate, clean!

# Containers & data manipulation
using Tables, PrettyTables
using DelimitedFiles
using OrderedCollections
using CategoricalArrays
import InvertedIndices: Not

# Distributed computing
using Distributed
using ComputationalResources
using ComputationalResources: CPUProcesses
using ProgressMeter

# Operations & extensions
import LossFunctions
import LossFunctions: DistanceLoss, MarginLoss, SupervisedLoss
import StatsBase
import StatsBase: fit!, mode, countmap
import Missings: levels
import Distributions
import Distributions: pdf

# from Standard Library:
using Statistics, LinearAlgebra, Random, InteractiveUtils

# ===================================================================
## METHOD EXPORTS

# -------------------------------------------------------------------
# re-exports from MLJModelInterface, (MLJ)ScientificTypes
# NOTE: MLJBase does **not** re-export UnivariateFinite to avoid
# ambiguities between the raw constructor (MLJBase.UnivariateFinite)
# and the general method (MLJModelInterface.UnivariateFinite)

# MLJ model hierarchy
export MLJType, Model, Supervised, Unsupervised,
       Probabilistic, Deterministic, Interval, Static,
       UnivariateFinite

# model constructor + metadata
export @mlj_model, metadata_pkg, metadata_model

# model api
export fit, update, update_data, transform, inverse_transform,
       fitted_params, predict, predict_mode, predict_mean, predict_median,
       evaluate, clean!

# model traits
export input_scitype, output_scitype, target_scitype,
       is_pure_julia, package_name, package_license,
       load_path, package_uuid, package_url,
       is_wrapper, supports_weights, supports_online,
       docstring, name, is_supervised,
       prediction_type, implemented_methods, hyperparameters,
       hyperparameter_types, hyperparameter_ranges

# data operations
export matrix, int, classes, decoder, table,
       nrows, selectrows, selectcols, select

# re-exports from (MLJ)ScientificTypes
export Scientific, Found, Unknown, Known, Finite, Infinite,
       OrderedFactor, Multiclass, Count, Continuous, Textual,
       Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait
export coerce, coerce!, autotype, schema, info

# -------------------------------------------------------------------
# exports from MLJBase

export DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork,
       best, @load

# computational_resources.jl:
export default_resource

# equality.jl:
export is_same_except

# one_dimensional_ranges.jl:
export ParamRange, NumericRange, NominalRange, iterator, scale

# parameter_inspection.jl:
export params # note this is *not* an extension of StatsBase.params

# data.jl:
export reconstruct, levels_seen, container_type,
       partition, unpack, complement, restrict, corestrict

# utilities.jl:
export @set_defaults, flat_values, recursive_setproperty!,
       recursive_getproperty, pretty, unwind

# show.jl
export HANDLE_GIVEN_ID, @more, @constant, color_on, color_off

# distributions.jl:
export average

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
export make_blobs, make_moons, make_circles, make_regression

# composites.jl:
export machines, sources, anonymize!, @from_network, fitresults

# pipelines.jl:
export @pipeline

# resampling.jl:
export ResamplingStrategy, Holdout, CV, StratifiedCV,
       evaluate!, Resampler, PerformanceEvaluation

# -------------------------------------------------------------------
# exports from MLJBase specific to Measure (these may go in their
# specific MLJMeasureInterface package in some future)

# measures/registry.jl:
export measures, metadata_measure

# measure/measures.jl:
export orientation, reports_each_observation,
       is_feature_dependent, aggregation,
       aggregate, default_measure, value

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
       TruePositiveRate, TrueNegativeRate, FalsePositiveRate,
       FalseNegativeRate, FalseDiscoveryRate, Precision, NPV, FScore,
       # standard synonyms
       TPR, TNR, FPR, FNR, FDR, PPV,
       Recall, Specificity, BACC,
       # instances and their synonyms
       truepositive, truenegative, falsepositive, falsenegative,
       truepositive_rate, truenegative_rate, falsepositive_rate,
       falsenegative_rate, negativepredicitive_value,
       positivepredictive_value, tpr, tnr, fpr, fnr,
       falsediscovery_rate, fdr, npv, ppv,
       recall, sensitivity, hit_rate, miss_rate,
       specificity, selectivity, f1score, fallout

# -------------------------------------------------------------------
# re-export from Random, StatsBase, Statistics, Distributions,
# CategoricalArrays, InvertedIndices:
export pdf, mode, median, mean, shuffle!, categorical, shuffle,
       levels, levels!, std, Not


# ===================================================================
## CONSTANTS

# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)

# horizontal space for field names in `MLJType` object display:
const COLUMN_WIDTH = 24
# how deep to display fields of `MLJType` objects:
const DEFAULT_SHOW_DEPTH = 0
const DEFAULT_AS_CONSTRUCTED_SHOW_DEPTH = 2
const INDENT = 4

const CategoricalElement = Union{CategoricalValue,CategoricalString}

const Arr = AbstractArray
const Vec = AbstractVector

const MMI = MLJModelInterface
const FI  = MLJModelInterface.FullInterface

# ===================================================================
# Computational Resource
# default_resource allows to switch the mode of parallelization

default_resource()    = DEFAULT_RESOURCE[]
default_resource(res) = (DEFAULT_RESOURCE[] = res)

# stub for @load (extended by MLJModels)
macro load end

# ===================================================================
# Includes

include("init.jl")
include("utilities.jl")
include("parameter_inspection.jl")
include("equality.jl")
include("show.jl")
include("info_dict.jl")

include("interface/data_utils.jl")
include("interface/model_api.jl")
include("interface/univariate_finite.jl")

include("distributions.jl")

include("machines.jl")

include("composition/networks.jl")
include("composition/composites.jl")
include("composition/pipelines.jl")
include("composition/pipeline_static.jl")
VERSION ≥ v"1.3.0-" && include("composition/arrows.jl")

include("operations.jl")

include("hyperparam/one_dimensional_ranges.jl")
include("hyperparam/one_dimensional_range_methods.jl")
include("hyperparam/resampling.jl")

include("data/data.jl")
include("data/datasets.jl")
include("data/datasets_synthetic.jl")

include("measures/measures.jl")
include("measures/registry.jl")

end # module
