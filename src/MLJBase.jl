module MLJBase

# ===================================================================
# IMPORTS

import Base: ==, precision, getindex, setindex!
import Base.+, Base.*, Base./

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
    params, training_losses, feature_importances

# Macros
using Parameters

# Containers & data manipulation
using Serialization
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
## CONSTANTS

# for variable global constants, see src/init.jl

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
const INDENT = 2

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

const MEASURE_TYPE_ALIASES = [
    :FPR, :FNR, :TPR, :TNR,
    :FDR, :PPV, :NPV, :Recall, :Specificity,
    :MFPR, :MFNR, :MTPR, :MTNR,
    :MFDR, :MPPV, :MNPV, :MulticlassRecall, :MulticlassSpecificity,
    :MCR,
    :MCC,
    :BAC, :BACC,
    :RMS, :RMSPV, :RMSL, :RMSLP, :RMSP,
    :MAV, :MAE, :MAPE,
    :RSQ, :LogCosh,
    :CrossEntropy,
    :AUC
]

const LOSS_FUNCTIONS = vcat(MARGIN_LOSSES, DISTANCE_LOSSES)

# ===================================================================
# Computational Resource
# default_resource allows to switch the mode of parallelization

default_resource()    = DEFAULT_RESOURCE[]
default_resource(res) = (DEFAULT_RESOURCE[] = res;)

# ===================================================================
# Includes

include("init.jl")
include("utilities.jl")
include("show.jl")
include("interface/data_utils.jl")
include("interface/model_api.jl")

include("models.jl")
include("sources.jl")
include("machines.jl")

include("composition/abstract_types.jl")
include("composition/learning_networks/nodes.jl")
include("composition/learning_networks/inspection.jl")
include("composition/learning_networks/signatures.jl")
include("composition/learning_networks/machines.jl")
include("composition/learning_networks/replace.jl")

include("composition/models/methods.jl")
include("composition/models/network_composite_types.jl")
include("composition/models/network_composite.jl")
include("composition/models/from_network.jl")
include("composition/models/inspection.jl")
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
include("measures/doc_strings.jl")

include("composition/models/stacking.jl")

# function on the right-hand side is defined in src/measures/meta_utilities.jl:
const MEASURE_TYPES_ALIASES_AND_INSTANCES = measures_for_export()

const EXTENDED_ABSTRACT_MODEL_TYPES = vcat(
    MLJBase.MLJModelInterface.ABSTRACT_MODEL_SUBTYPES,
    MLJBase.NETWORK_COMPOSITE_TYPES, # src/composition/models/network_composite_types.jl
    MLJBase.COMPOSITE_TYPES, # src/composition/abstract_types.jl
    MLJBase.SURROGATE_TYPES, # src/composition/abstract_types.jl
    [:MLJType, :Model, :NetworkComposite, :Surrogate, :Composite],
)

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
    evaluate, clean!, training_losses, feature_importances

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
export UnivariateFiniteArray, UnivariateFiniteVector

# -----------------------------------------------------------------------
# abstract model types defined in MLJModelInterface.jl and extended here:
for T in EXTENDED_ABSTRACT_MODEL_TYPES
    @eval(export $T)
end

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
export HANDLE_GIVEN_ID, @more, @constant, color_on, color_off

# datasets.jl:
export load_boston, load_ames, load_iris, load_sunspots,
    load_reduced_ames, load_crabs, load_smarket,
    @load_boston, @load_ames, @load_iris, @load_sunspots,
    @load_reduced_ames, @load_crabs, @load_smarket

# sources.jl:
export source, Source, CallableReturning

# machines.jl:
export machine, Machine, fit!, report, fit_only!, default_scitype_check_level,
    serializable, last_model

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

# resampling.jl:
export ResamplingStrategy, Holdout, CV, StratifiedCV, TimeSeriesCV,
    evaluate!, Resampler, PerformanceEvaluation

# `MLJType` and the abstract `Model` subtypes are exported from within
# src/composition/abstract_types.jl

# -------------------------------------------------------------------
# exports from MLJBase specific to measures

# measure names:
for m in MEASURE_TYPES_ALIASES_AND_INSTANCES
    :(export $m) |> eval
end

# measures/registry.jl:
export measures, metadata_measure

# measure/measures.jl (excluding traits):
export aggregate, default_measure, value, skipinvalid

# measures/probabilistic:
export roc_curve, roc

# measures/finite.jl (averaging modes for multiclass scores)
export no_avg, macro_avg, micro_avg


# -------------------------------------------------------------------
# re-export from Random, StatsBase, Statistics, Distributions,
# OrderedCollections, CategoricalArrays, InvertedIndices:
export pdf, sampler, mode, median, mean, shuffle!, categorical, shuffle,
   levels, levels!, std, Not, support, logpdf, LittleDict

end # module
