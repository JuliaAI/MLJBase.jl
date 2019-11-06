# see also the macro versions in datasets.jl

# -------------------------------------------------------
# To add a new dataset assuming it has a header and is, at path
# `data/newdataset.csv`
#
# 1. start by loading it with CSV:
#
#   fpath = joinpath("datadir", "newdataset.csv")
#   data = CSV.read(fpath, copycols=true,
#                   categorical=true)
#
# 2. load it with DelimitedFiles and Tables
#
#   data_raw, data_header = readdlm(fpath, ',', header=true)
#   data_table = Tables.table(data_raw; header=Symbol.(vec(data_header)))
#
# 3. retrieve the conversions:
#
#   for (n, st) in zip(names(data), scitype_union.(eachcol(data)))
#     println(":$n=>$st,")
#   end
#
# 4. copy and paste the result in a coerce
#
#  data_table = coerce(data_table, ...)
#
# -------------------------------------------------------

const DATA_DIR = joinpath(@__DIR__, "..", "data")

const COERCE_BOSTON = (:Chas => Count,)

const COERCE_REDUCED_AMES = (
    :target       => Continuous,
    :OverallQual  => OrderedFactor,
    :GrLivArea    => Continuous,
    :Neighborhood => Multiclass,
    :x1stFlrSF    => Continuous,
    :TotalBsmtSF  => Continuous,
    :BsmtFinSF1   => Continuous,
    :LotArea      => Continuous,
    :GarageCars   => Count,
    :MSSubClass   => Multiclass,
    :GarageArea   => Count,
    :YearRemodAdd => Continuous,
    :YearBuilt    => Continuous)

const COERCE_AMES = (
    :Id             => Count,
    :MSSubClass     => Multiclass,
    :MSZoning       => Multiclass,
    :LotFrontage    => Continuous,
    :LotArea        => Count,
    :Street         => Multiclass,
    :LotShape       => Multiclass,
    :LandContour    => Multiclass,
    :LotConfig      => Multiclass,
    :LandSlope      => Multiclass,
    :Neighborhood   => Multiclass,
    :Condition1     => Multiclass,
    :Condition2     => Multiclass,
    :BldgType       => Multiclass,
    :HouseStyle     => Multiclass,
    :OverallQual    => Count,
    :OverallCond    => Count,
    :YearBuilt      => Count,
    :YearRemodAdd   => Count,
    :RoofStyle      => Multiclass,
    :RoofMatl       => Multiclass,
    :Exterior1st    => Multiclass,
    :Exterior2nd    => Multiclass,
    :MasVnrType     => Multiclass,
    :MasVnrArea     => Count,
    :ExterQual      => Multiclass,
    :ExterCond      => Multiclass,
    :Foundation     => Multiclass,
    :BsmtQual       => Multiclass,
    :BsmtCond       => Multiclass,
    :BsmtExposure   => Multiclass,
    :BsmtFinType1   => Multiclass,
    :BsmtFinSF1     => Continuous,
    :BsmtFinType2   => Multiclass,
    :BsmtFinSF2     => Count,
    :BsmtUnfSF      => Count,
    :TotalBsmtSF    => Continuous,
    :Heating        => Multiclass,
    :HeatingQC      => Multiclass,
    :CentralAir     => Multiclass,
    :Electrical     => Multiclass,
    :x1stFlrSF      => Count,
    :x2ndFlrSF      => Count,
    :LowQualFinSF   => Count,
    :GrLivArea      => Count,
    :BsmtFullBath   => Count,
    :BsmtHalfBath   => Count,
    :FullBath       => Count,
    :HalfBath       => Count,
    :BedroomAbvGr   => Count,
    :KitchenAbvGr   => Count,
    :KitchenQual    => Multiclass,
    :TotRmsAbvGrd   => Count,
    :Functional     => Multiclass,
    :Fireplaces     => Count,
    :FireplaceQu    => Multiclass,
    :GarageType     => Multiclass,
    :GarageYrBlt    => Continuous,
    :GarageFinish   => Multiclass,
    :GarageCars     => Count,
    :GarageArea     => Count,
    :GarageQual     => Multiclass,
    :GarageCond     => Multiclass,
    :PavedDrive     => Multiclass,
    :WoodDeckSF     => Count,
    :OpenPorchSF    => Count,
    :EnclosedPorch  => Count,
    :x3SsnPorch     => Count,
    :ScreenPorch    => Count,
    :PoolArea       => Count,
    :MiscVal        => Count,
    :MoSold         => Count,
    :YrSold         => Count,
    :SaleType       => Multiclass,
    :SaleCondition  => Multiclass,
    :target         => Continuous)

const COERCE_IRIS = (
    :sepal_length => Continuous,
    :sepal_width  => Continuous,
    :petal_length => Continuous,
    :petal_width  => Continuous,
    :target       => Multiclass)

const COERCE_CRABS = (
    :sp    => Multiclass,
    :sex   => Multiclass,
    :index => Count,
    :FL    => Continuous,
    :RW    => Continuous,
    :CL    => Continuous,
    :CW    => Continuous,
    :BD    => Continuous)

typeof(COERCE_CRABS)

"""
load_dataset(fpath, coercions)

Load one of standard dataset like Boston etc assuming the file is a comma separated file with
a  header.
"""
function load_dataset(fname::String, coercions::Tuple)
    fpath = joinpath(DATA_DIR, fname)
    data_raw, data_header = readdlm(fpath, ',', header=true)
    data_table = Tables.table(data_raw; header=Symbol.(vec(data_header)))
    return coerce(data_table, coercions...)
end


load_boston()       = load_dataset("Boston.csv", COERCE_BOSTON)
load_reduced_ames() = load_dataset("reduced_ames.csv", COERCE_REDUCED_AMES)
load_ames()         = load_dataset("ames.csv", COERCE_AMES)
load_iris()         = load_dataset("iris.csv", COERCE_IRIS)
load_crabs()        = load_dataset("crabs.csv", COERCE_CRABS)


"""Load a well-known public regression dataset with `Continuous` features."""
macro load_boston()
    quote
        y, X = unpack(load_boston(), ==(:MedV), x->x != :Chas)
        (X, y)
    end
end

"""Load a reduced version of the well-known Ames Housing task"""
macro load_reduced_ames()
    quote
        y, X = unpack(load_reduced_ames(), ==(:target), x-> true)
        (X, y)
    end
end

"""Load the full version of the well-known Ames Housing task."""
macro load_ames()
    quote
        y, X = unpack(load_ames(), ==(:target), x->x != :Id)
        (X, y)
    end
end

"""Load a well-known public classification task with nominal features."""
macro load_iris()
    quote
        y, X = unpack(load_iris(), ==(:target), x-> true)
        (X, y)
    end
end

"""Load a well-known crab classification dataset with nominal features."""
macro load_crabs()
    quote
        y, X = unpack(load_crabs(), ==(:sp), x-> !(x in [:sex, :index]))
        (X, y)
    end
end
