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

const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

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
    :GarageArea   => Continuous,
    :YearRemodAdd => Count,
    :YearBuilt    => Count)

const COERCE_AMES = (
    :Id             => Count,
    :MSSubClass     => Multiclass,
    :MSZoning       => Multiclass,
    :LotFrontage    => Continuous,
    :LotArea        => Continuous,
    :Street         => Multiclass,
    :LotShape       => Multiclass,
    :LandContour    => Multiclass,
    :LotConfig      => Multiclass,
    :LandSlope      => OrderedFactor,
    :Neighborhood   => Multiclass,
    :Condition1     => Multiclass,
    :Condition2     => Multiclass,
    :BldgType       => Multiclass,
    :HouseStyle     => Multiclass,
    :OverallQual    => OrderedFactor,
    :OverallCond    => OrderedFactor,
    :YearBuilt      => Count,
    :YearRemodAdd   => Count,
    :RoofStyle      => Multiclass,
    :RoofMatl       => Multiclass,
    :Exterior1st    => Multiclass,
    :Exterior2nd    => Multiclass,
    :MasVnrType     => Multiclass,
    :MasVnrArea     => Continuous,
    :ExterQual      => OrderedFactor,
    :ExterCond      => OrderedFactor,
    :Foundation     => Multiclass,
    :BsmtQual       => OrderedFactor,
    :BsmtCond       => OrderedFactor,
    :BsmtExposure   => OrderedFactor,
    :BsmtFinType1   => Multiclass,
    :BsmtFinSF1     => Continuous,
    :BsmtFinType2   => Multiclass,
    :BsmtFinSF2     => Continuous,
    :BsmtUnfSF      => Continuous,
    :TotalBsmtSF    => Continuous,
    :Heating        => Multiclass,
    :HeatingQC      => OrderedFactor,
    :CentralAir     => Multiclass,
    :Electrical     => Multiclass,
    :x1stFlrSF      => Continuous,
    :x2ndFlrSF      => Continuous,
    :LowQualFinSF   => Continuous,
    :GrLivArea      => Continuous,
    :BsmtFullBath   => Count,
    :BsmtHalfBath   => Count,
    :FullBath       => Count,
    :HalfBath       => Count,
    :BedroomAbvGr   => Count,
    :KitchenAbvGr   => Count,
    :KitchenQual    => OrderedFactor,
    :TotRmsAbvGrd   => Count,
    :Functional     => OrderedFactor,
    :Fireplaces     => Count,
    :FireplaceQu    => OrderedFactor,
    :GarageType     => Multiclass,
    :GarageYrBlt    => Count,
    :GarageFinish   => Multiclass,
    :GarageCars     => Count,
    :GarageArea     => Continuous,
    :GarageQual     => OrderedFactor,
    :GarageCond     => OrderedFactor,
    :PavedDrive     => Multiclass,
    :WoodDeckSF     => Continuous,
    :OpenPorchSF    => Continuous,
    :EnclosedPorch  => Continuous,
    :x3SsnPorch     => Continuous,
    :ScreenPorch    => Continuous,
    :PoolArea       => Continuous,
    :MiscVal        => Continuous,
    :MoSold         => Multiclass,
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

const COERCE_SMARKET = (
    :Year=>Continuous,
    :Lag1=>Continuous,
    :Lag2=>Continuous,
    :Lag3=>Continuous,
    :Lag4=>Continuous,
    :Lag5=>Continuous,
    :Volume=>Continuous,
    :Today=>Continuous,
    :Direction=>Multiclass{2})

const COERCE_SUNSPOTS = (
    (:sunspot_number=>Continuous),)

"""
    load_dataset(fpath, coercions)

Load one of standard dataset like Boston etc assuming the file is a
comma separated file with a header.

"""
function load_dataset(fname::String, coercions::Tuple)
    fpath = joinpath(DATA_DIR, fname)
    data_raw, data_header = readdlm(fpath, ',', header=true)
    data_table = MLJBase.table(data_raw; names=Symbol.(vec(data_header)))
    return coerce(data_table, coercions...; tight=true)
end


load_boston()       = load_dataset("Boston.csv", COERCE_BOSTON)
load_reduced_ames() = load_dataset("reduced_ames.csv", COERCE_REDUCED_AMES)
function load_ames()
    data = load_dataset("ames.csv", COERCE_AMES)
    levels!(data.LandSlope, ["Gtl", "Mod", "Sev"])
    levels!(data.ExterQual, ["Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.ExterCond, ["Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.HeatingQC, ["Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.KitchenQual, ["Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.BsmtQual, ["_NA", "Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.BsmtCond, ["_NA", "Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.BsmtExposure, ["_NA", "No", "Mn", "Av", "Gd"])
    levels!(data.FireplaceQu, ["None", "Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.GarageQual, ["_NA", "Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.GarageCond, ["_NA", "Po", "Fa", "TA", "Gd", "Ex"])
    levels!(data.Functional, ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2",
                              "Sev", "Sal"])
    return data
end
load_iris()         = load_dataset("iris.csv", COERCE_IRIS)
load_crabs()        = load_dataset("crabs.csv", COERCE_CRABS)
function load_smarket()
    data1 = load_dataset("smarket.csv", COERCE_SMARKET)
    return merge(data1, (Year=Dates.Date.(data1.Year),))
end

"""Load a well-known sunspot time series (table with one column).
[https://www.sws.bom.gov.au/Educational/2/3/6]](https://www.sws.bom.gov.au/Educational/2/3/6)
"""
load_sunspots() = load_dataset("sunspots.csv", COERCE_SUNSPOTS)

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

"""Load a well-known sunspot time series (single table with one column)."""
macro load_sunspots()
    quote
        load_sunspots()
    end
end

"""Load a well-known crab classification dataset with nominal features."""
macro load_crabs()
    quote
        y, X = unpack(load_crabs(), ==(:sp), x-> !(x in [:sex, :index]))
        (X, y)
    end
end

""" Load S&P Stock Market dataset, as used in (An Introduction to
Statistical Learning with applications in
R)[https://rdrr.io/cran/ISLR/man/Smarket.html](https://rdrr.io/cran/ISLR/man/Smarket.html),
by Witten et al (2013), Springer-Verlag, New York."""
macro load_smarket()
    quote
        y, X = unpack(load_smarket(), ==(:Direction), x-> true)
        (X, y)
    end
end
