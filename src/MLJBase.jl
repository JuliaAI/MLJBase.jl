module MLJBase

# ===================================================================
# IMPORTS

import Base: ==, precision, getindex, setindex!
import Base.+, Base.*

# Scitype
using ScientificTypes

# Traits for models and measures (which are being overloaded):
using StatisticalTraits

for trait in StatisticalTraits.TRAITS
    eval(:(import StatisticalTraits.$trait))
end

import Base.instances # considered a trait for measures
import StatisticalTraits.snakecase
import StatisticalTraits.info

# Interface

# HACK: When https://github.com/JuliaAI/MLJModelInterface.jl/issues/124 and
# https://github.com/JuliaAI/MLJModelInterface.jl/issues/131 is resolved:
# Uncomment next line and delete "Hack Block"
# using MLJModelInterface
#####################
# Hack Block begins #
#####################
exported_names(m::Module) =
    filter!(x -> Base.isexported(m, x),
            Base.names(m; all=true, imported=true))
import MLJModelInterface
for name in exported_names(MLJModelInterface)
    name in [
        :UnivariateFinite,
        :augmented_transform,
        :info,
        :scitype # Needed to avoid clashing with `ScientificTypes.scitype` 
    ] && continue
    quote
        import MLJModelInterface.$name
    end |> eval
end
###################
# Hack Block ends #
###################

import MLJModelInterface: fit, update, update_data, transform,
    inverse_transform, fitted_params, predict, predict_mode,
    predict_mean, predict_median, predict_joint,
    evaluate, clean!, is_same_except,
    save, restore, is_same_except, istransparent,
    params, training_losses

# Macros
using Parameters

# Containers & data manipulation
using Tables
import PrettyTables
using DelimitedFiles
using OrderedCollections
using CategoricalArrays
import CategoricalArrays.DataAPI.unwrap
import InvertedIndices: Not
import Dates

# Distributed computing
using Distributed
using ComputationalResources
import ComputationalResources: CPU1, CPUProcesses, CPUThreads

using ProgressMeter
import .Threads

# Operations & extensions
import LossFunctions
import StatsBase
import StatsBase: fit!, mode, countmap
import Missings: levels
using Missings
import Distributions
using CategoricalDistributions
import Distributions: pdf, logpdf, sampler
const Dist = Distributions

# from Standard Library:
using Statistics, LinearAlgebra, Random, InteractiveUtils

# ===================================================================
## EXPORTS

# -------------------------------------------------------------------
# re-exports from MLJModelInterface, ScientificTypes
# NOTE: MLJBase does **not** re-export UnivariateFinite to avoid
# ambiguities between the raw constructor (MLJBase.UnivariateFinite)
# and the general method (MLJModelInterface.UnivariateFinite)

# traits for measures and models:
using StatisticalTraits

for trait in StatisticalTraits.TRAITS
    eval(:(export $trait))
end

export implemented_methods # defined here and not in StatisticalTraits

export UnivariateFinite

# MLJType equality
export is_same_except

# model constructor + metadata
export @mlj_model, metadata_pkg, metadata_model

# model api
export fit, update, update_data, transform, inverse_transform,
    fitted_params, predict, predict_mode, predict_mean,
    predict_median, predict_joint,
    evaluate, clean!, training_losses

# data operations
export matrix, int, classes, decoder, table,
    nrows, selectrows, selectcols, select

# re-export from ComputationalResources.jl:
export CPU1, CPUProcesses, CPUThreads

# re-exports from ScientificTypes
export Unknown, Known, Finite, Infinite,
    OrderedFactor, Multiclass, Count, Continuous, Textual,
    Binary, ColorImage, GrayImage, Image, Table

export scitype, scitype_union, elscitype, nonmissing
export coerce, coerce!, autotype, schema, info

# re-exports from CategoricalDistributions:
export CategoricalUnivariateFiniteArray, UnivariateFiniteVector


# -------------------------------------------------------------------
# exports from this module, MLJBase

# computational_resources.jl:
export default_resource

# one_dimensional_ranges.jl:
export ParamRange, NumericRange, NominalRange, iterator, scale

# parameter_inspection.jl:
export params # note this is *not* an extension of StatsBase.params

# data.jl:
export partition, unpack, complement, restrict, corestrict

# utilities.jl:
export flat_values, recursive_setproperty!,
    recursive_getproperty, pretty, unwind

# show.jl
export HANDLE_GIVEN_ID, @more, @constant, @bind, color_on, color_off

# datasets.jl:
export load_boston, load_ames, load_iris, load_sunspots,
    load_reduced_ames, load_crabs, load_smarket,
    @load_boston, @load_ames, @load_iris, @load_sunspots,
    @load_reduced_ames, @load_crabs, @load_smarket

# sources.jl:
export source, Source, CallableReturning

# machines.jl:
export machine, Machine, fit!, report, fit_only!

# datasets_synthetics.jl
export make_blobs, make_moons, make_circles, make_regression

# composition (surrogates and composites are exported in composition):
export machines, sources, @from_network, @pipeline, Stack,
    glb, @tuple, node, @node, sources, origins, return!,
    nrows_at_source, machine, rebind!, nodes, freeze!, thaw!,
    Node, AbstractNode, Pipeline,
    ProbabilisticPipeline, DeterministicPipeline, UnsupervisedPipeline,
    StaticPipeline, IntervalPipeline

export TransformedTargetModel

# aliases to the above,  kept for backwards compatibility:
export  DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork

# resampling.jl:
export ResamplingStrategy, Holdout, CV, StratifiedCV, TimeSeriesCV,
    evaluate!, Resampler, PerformanceEvaluation

# -------------------------------------------------------------------
# exports from MLJBase specific to Measure (these may go in their
# specific MLJMeasureInterface package in some future)

# `MLJType` and the abstract `Model` subtypes are exported from within
# src/composition/abstract_types.jl

# measures/registry.jl:
export measures, metadata_measure

# measure/measures.jl (excluding traits):
export aggregate, default_measure, value, skipinvalid

# measures/probabilistic:
export cross_entropy, BrierScore, brier_score,
    BrierLoss, brier_loss,
    LogLoss, log_loss, LogScore, log_score,
    SphericalScore, spherical_score,
    auc, area_under_curve, roc_curve, roc

# measures/continuous.jl:
export mav, mae, mape, rms, rmsl, rmslp1, rmsp, l1, l2, log_cosh,
    MAV, MAE, MeanAbsoluteError, mean_absolute_error, mean_absolute_value,
    LPLoss, RootMeanSquaredProportionalError, RMSP,
    RMS, rmse, RootMeanSquaredError, root_mean_squared_error,
    RootMeanSquaredLogError, RMSL, root_mean_squared_log_error, rmsl, rmsle,
    RootMeanSquaredLogProportionalError, rmsl1, RMSLP,
    MAPE, MeanAbsoluteProportionalError, log_cosh_loss, LogCosh, LogCoshLoss,
    RSquared, rsq, rsquared

# measures/confusion_matrix.jl:
export confusion_matrix, confmat, ConfusionMatrix

# measures/finite.jl:
export misclassification_rate, mcr, accuracy,
    balanced_accuracy, bacc, bac, BalancedAccuracy,
    matthews_correlation, mcc, MCC, AUC, AreaUnderCurve,
    MisclassificationRate, Accuracy, MCR, BACC, BAC,
    MatthewsCorrelation

# measures/finite.jl -- OrderedFactor{2} (order dependent):
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
    TruePositiveRate, TrueNegativeRate, FalsePositiveRate,
    FalseNegativeRate, FalseDiscoveryRate, Precision, NPV, FScore,
    NegativePredictiveValue,
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

# measures/finite.jl -- Finite{N} - multiclass generalizations of
# above OrderedFactor{2} measures (but order independent):
export MulticlassTruePositive, MulticlassTrueNegative, MulticlassFalsePositive,
    MulticlassFalseNegative, MulticlassTruePositiveRate,
    MulticlassTrueNegativeRate, MulticlassFalsePositiveRate,
    MulticlassFalseNegativeRate, MulticlassFalseDiscoveryRate,
    MulticlassPrecision, MulticlassNegativePredictiveValue, MulticlassFScore,
    # standard synonyms
    MTPR, MTNR, MFPR, MFNR, MFDR, MPPV,
    MulticlassRecall, MulticlassSpecificity,
    # instances and their synonyms
    multiclass_truepositive, multiclass_truenegative,
    multiclass_falsepositive,
    multiclass_falsenegative, multiclass_true_positive,
    multiclass_true_negative, multiclass_false_positive,
    multiclass_false_negative, multiclass_truepositive_rate,
    multiclass_truenegative_rate, multiclass_falsepositive_rate,
    multiclass_true_positive_rate, multiclass_true_negative_rate,
    multiclass_false_positive_rate, multiclass_falsenegative_rate,
    multiclass_negativepredictive_value, multiclass_false_negative_rate,
    multiclass_negative_predictive_value, multiclass_positivepredictive_value,
    multiclass_positive_predictive_value, multiclass_tpr, multiclass_tnr,
    multiclass_fpr, multiclass_fnr, multiclass_falsediscovery_rate,
    multiclass_false_discovery_rate, multiclass_fdr, multiclass_npv,
    multiclass_ppv, multiclass_recall, multiclass_sensitivity,
    multiclass_hit_rate, multiclass_miss_rate, multiclass_specificity,
    multiclass_selectivity, macro_f1score, micro_f1score,
    multiclass_f1score, multiclass_fallout, multiclass_precision,
    # averaging modes
    no_avg, macro_avg, micro_avg

# measures/loss_functions_interface.jl
export dwd_margin_loss, exp_loss, l1_hinge_loss, l2_hinge_loss, l2_margin_loss,
    logit_margin_loss, modified_huber_loss, perceptron_loss, sigmoid_loss,
    smoothed_l1_hinge_loss, zero_one_loss, huber_loss, l1_epsilon_ins_loss,
    l2_epsilon_ins_loss, lp_dist_loss, logit_dist_loss, periodic_loss,
    quantile_loss

# ------------------------------------------------------------------------
# re-export from LossFunctions.jl:
const MARGIN_LOSSES = [
    :DWDMarginLoss,
    :ExpLoss,
    :L1HingeLoss,
    :L2HingeLoss,
    :L2MarginLoss,
    :LogitMarginLoss,
    :ModifiedHuberLoss,
    :PerceptronLoss,
    :SigmoidLoss,
    :SmoothedL1HingeLoss,
    :ZeroOneLoss
]

const DISTANCE_LOSSES = [
    :HuberLoss,
    :L1EpsilonInsLoss,
    :L2EpsilonInsLoss,
    :LPDistLoss,
    :LogitDistLoss,
    :PeriodicLoss,
    :QuantileLoss
]

const WITH_PARAMETERS = [
    :DWDMarginLoss,
    :SmoothedL1HingeLoss,
    :HuberLoss,
    :L1EpsilonInsLoss,
    :L2EpsilonInsLoss,
    :LPDistLoss,
    :QuantileLoss,
]

const LOSS_FUNCTIONS = vcat(MARGIN_LOSSES, DISTANCE_LOSSES)

for Loss in LOSS_FUNCTIONS
    eval(:(export $Loss))
end

# -------------------------------------------------------------------
# re-export from Random, StatsBase, Statistics, Distributions,
# OrderedCollections, CategoricalArrays, InvertedIndices:
export pdf, sampler, mode, median, mean, shuffle!, categorical, shuffle,
   levels, levels!, std, Not, support, logpdf, LittleDict


# ===================================================================
## CONSTANTS

const PREDICT_OPERATIONS = (:predict,
                            :predict_mean,
                            :predict_mode,
                            :predict_median,
                            :predict_joint)

const OPERATIONS = (PREDICT_OPERATIONS..., :transform, :inverse_transform)

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

# Note the following are existential (union) types. In particular,
# ArrMissing{Integer} is not the same as Arr{Union{Missing,Integer}},
# etc.
const ArrMissing{T,N} = Arr{<:Union{Missing,T},N}
const VecMissing{T} = ArrMissing{T,1}
const CatArrMissing{T,N} = ArrMissing{CategoricalValue{T},N}

const MMI = MLJModelInterface
const FI  = MLJModelInterface.FullInterface
# ===================================================================
# Computational Resource
# default_resource allows to switch the mode of parallelization

default_resource()    = DEFAULT_RESOURCE[]
default_resource(res) = (DEFAULT_RESOURCE[] = res)

# ===================================================================
# Includes

include("init.jl")
include("utilities.jl")
include("show.jl")
include("interface/data_utils.jl")
include("interface/model_api.jl")

include("sources.jl")
include("machines.jl")

include("composition/abstract_types.jl")
include("composition/learning_networks/nodes.jl")
include("composition/learning_networks/inspection.jl")
include("composition/learning_networks/machines.jl")

include("composition/models/methods.jl")
include("composition/models/from_network.jl")
include("composition/models/inspection.jl")
include("composition/models/deprecated.jl")
include("composition/models/pipelines.jl")
include("composition/models/transformed_target_model.jl")

include("operations.jl")
include("resampling.jl")

include("hyperparam/one_dimensional_ranges.jl")
include("hyperparam/one_dimensional_range_methods.jl")

include("data/data.jl")
include("data/datasets.jl")
include("data/datasets_synthetic.jl")

include("measures/measures.jl")
include("measures/measure_search.jl")

include("composition/models/stacking.jl")

end # module
