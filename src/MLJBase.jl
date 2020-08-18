module MLJBase

# ===================================================================
# IMPORTS

import Base: ==, precision, getindex, setindex!
import Base.+, Base.*

# Scitype
import ScientificTypes: TRAIT_FUNCTION_GIVEN_NAME
import ScientificTypes
using MLJScientificTypes
using MLJModelInterface

# Interface
import MLJModelInterface: fit, update, update_data, transform,
    inverse_transform, fitted_params, predict,
    predict_mode, predict_mean, predict_median,
    evaluate, clean!, is_same_except,
    save, restore, is_same_except, istransparent,
    params

# Macros
using Parameters

# Containers & data manipulation
using Tables
import PrettyTables
using DelimitedFiles
using OrderedCollections
using CategoricalArrays
import InvertedIndices: Not
import JLSO
import Dates

# Distributed computing
using Distributed
using ComputationalResources
using ComputationalResources: CPUProcesses
using ProgressMeter
import .Threads

# Operations & extensions
import LossFunctions
import LossFunctions: DistanceLoss, MarginLoss, SupervisedLoss
import StatsBase
import StatsBase: fit!, mode, countmap
import Missings: levels
import Distributions
import Distributions: pdf, sampler
const Dist = Distributions

# from Standard Library:
using Statistics, LinearAlgebra, Random, InteractiveUtils

# ===================================================================
## EXPORTS

# -------------------------------------------------------------------
# re-exports from MLJModelInterface, (MLJ)ScientificTypes
# NOTE: MLJBase does **not** re-export UnivariateFinite to avoid
# ambiguities between the raw constructor (MLJBase.UnivariateFinite)
# and the general method (MLJModelInterface.UnivariateFinite)

# MLJ model hierarchy
export MLJType, Model, Supervised, Unsupervised,
    Probabilistic, Deterministic, Interval, Static,
    ProbabilisticComposite, DeterministicComposite,
    IntervalComposite, UnsupervisedComposite, StaticComposite,
    ProbabilisticSurrogate, DeterministicSurrogate,
    IntervalSurrogate, UnsupervisedSurrogate, StaticSurrogate,
    Surrogate, Composite

export UnivariateFinite

# MLJType equality
export is_same_except

# model constructor + metadata
export @mlj_model, metadata_pkg, metadata_model

# model api
export fit, update, update_data, transform, inverse_transform,
       fitted_params, predict, predict_mode, predict_mean, predict_median,
       evaluate, clean!

# model/measure matching:
export Checker, matching

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
export Unknown, Known, Finite, Infinite,
       OrderedFactor, Multiclass, Count, Continuous, Textual,
       Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait
export coerce, coerce!, autotype, schema, info

# -------------------------------------------------------------------
# exports from this module, MLJBase

export DeterministicComposite,
    ProbabilisticComposite,
    UnsupervisedComposite, @load

# computational_resources.jl:
export default_resource

# one_dimensional_ranges.jl:
export ParamRange, NumericRange, NominalRange, iterator, scale

# parameter_inspection.jl:
export params # note this is *not* an extension of StatsBase.params

# data.jl:
export partition, unpack, complement, restrict, corestrict

# utilities.jl:
export @set_defaults, flat_values, recursive_setproperty!,
       recursive_getproperty, pretty, unwind

# show.jl
export HANDLE_GIVEN_ID, @more, @constant, @bind, color_on, color_off

# univariate_finite/
export average, UnivariateFiniteArray, UnivariateFiniteVector

# info_dict.jl:
export info_dict

# datasets.jl:
export load_boston, load_ames, load_iris,
       load_reduced_ames, load_crabs, load_smarket,
       @load_boston, @load_ames, @load_iris,
       @load_reduced_ames, @load_crabs, @load_smarket

# sources.jl:
export source, Source, CallableReturning

# machines.jl:
export machine, Machine, fit!, report, fit_only!

# datasets_synthetics.jl
export make_blobs, make_moons, make_circles, make_regression

# composition:
export machines, sources, anonymize!, @from_network, fitresults, @pipeline,
    glb, @tuple, node, @node, sources, origins, return!,
    nrows_at_source, machine,
    rebind!, nodes, freeze!, thaw!, models, Node, AbstractNode,
    DeterministicSurrogate, ProbabilisticSurrogate, UnsupervisedSurrogate,
    DeterministicComposite, ProbabilisticComposite, UnsupervisedComposite

# aliases to the above,  kept for backwards compatibility:
export  DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork


# resampling.jl:
export ResamplingStrategy, Holdout, CV, StratifiedCV,
       evaluate!, Resampler, PerformanceEvaluation

# openml.jl:
export OpenML

# -------------------------------------------------------------------
# exports from MLJBase specific to Measure (these may go in their
# specific MLJMeasureInterface package in some future)

# measures/registry.jl:
export measures, metadata_measure

# measure/measures.jl:
export orientation, reports_each_observation,
    is_feature_dependent, aggregation,
    aggregate, default_measure, value,
    spports_weights, prediction_type

# measures/continuous.jl:
export mav, mae, mape, rms, rmsl, rmslp1, rmsp, l1, l2

# measures/confusion_matrix.jl:
export confusion_matrix, confmat

# measures/finite.jl
export cross_entropy, BrierScore, brier_score,
       misclassification_rate, mcr, accuracy,
       balanced_accuracy, bacc, bac,
       matthews_correlation, mcc

# measures/finite.jl -- binary order independent:
export auc, area_under_curve, roc_curve, roc

# measures/finite.jl -- binary order dependent:
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
       TruePositiveRate, TrueNegativeRate, FalsePositiveRate,
       FalseNegativeRate, FalseDiscoveryRate, Precision, NPV, FScore,
       # standard synonyms
       TPR, TNR, FPR, FNR, FDR, PPV,
       Recall, Specificity, BACC,
       # instances and their synonyms
       truepositive, truenegative, falsepositive, falsenegative,
       true_positive, true_negative, false_positive, false_negative,
       truepositive_rate, truenegative_rate, falsepositive_rate,
       true_positive_rate, true_negative_rate, false_positive_rate,
       falsenegative_rate, negativepredictive_value,
       false_negative_rate, negative_predictive_value,
       positivepredictive_value, positive_predictive_value,
       tpr, tnr, fpr, fnr,
       falsediscovery_rate, false_discovery_rate, fdr, npv, ppv,
       recall, sensitivity, hit_rate, miss_rate,
       specificity, selectivity, f1score, fallout


# -------------------------------------------------------------------
# re-export from Random, StatsBase, Statistics, Distributions,
# CategoricalArrays, InvertedIndices:
export pdf, sampler, mode, median, mean, shuffle!, categorical, shuffle,
       levels, levels!, std, Not, support


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
include("show.jl")
include("info_dict.jl")
include("interface/data_utils.jl")
include("interface/model_api.jl")

include("univariate_finite/types.jl")
include("univariate_finite/methods.jl")
include("univariate_finite/arrays.jl")

include("sources.jl")
include("machines.jl")

include("composition/abstract_types.jl")
include("composition/learning_networks/nodes.jl")
include("composition/learning_networks/inspection.jl")
include("composition/learning_networks/machines.jl")
@static if VERSION ≥ v"1.3.0-"
    include("composition/learning_networks/arrows.jl")
end

include("composition/models/methods.jl")
include("composition/models/from_network.jl")
include("composition/models/inspection.jl")
include("composition/models/pipelines.jl")
include("composition/models/deprecated.jl")
include("composition/models/_wrapped_function.jl")

include("operations.jl")
include("resampling.jl")

include("hyperparam/one_dimensional_ranges.jl")
include("hyperparam/one_dimensional_range_methods.jl")

include("data/data.jl")
include("data/datasets.jl")
include("data/datasets_synthetic.jl")

include("matching.jl")
include("measures/measures.jl")
include("measures/measure_search.jl")

include("openml.jl")

end # module
