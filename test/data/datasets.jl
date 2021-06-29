module TestDatasets

# using Revise
using Test
using MLJBase

X, y = @load_boston
@test schema(X).names  == (:Crim, :Zn, :Indus, :NOx, :Rm, :Age,
                           :Dis, :Rad, :Tax, :PTRatio, :Black, :LStat)
@test scitype(y) <: AbstractVector{Continuous}

X, y = @load_ames
schema(X).names == (:MSSubClass, :MSZoning, :LotFrontage, :LotArea,
                    :Street, :LotShape, :LandContour, :LotConfig,
                    :LandSlope, :Neighborhood, :Condition1,
                    :Condition2, :BldgType, :HouseStyle,
                    :OverallQual, :OverallCond, :YearBuilt,
                    :YearRemodAdd, :RoofStyle, :RoofMatl,
                    :Exterior1st, :Exterior2nd, :MasVnrType,
                    :MasVnrArea, :ExterQual, :ExterCond,
                    :Foundation, :BsmtQual, :BsmtCond, :BsmtExposure,
                    :BsmtFinType1, :BsmtFinSF1, :BsmtFinType2,
                    :BsmtFinSF2, :BsmtUnfSF, :TotalBsmtSF, :Heating,
                    :HeatingQC, :CentralAir, :Electrical, :x1stFlrSF,
                    :x2ndFlrSF, :LowQualFinSF, :GrLivArea, :BsmtFullBath,
                    :BsmtHalfBath, :FullBath, :HalfBath,
                    :BedroomAbvGr, :KitchenAbvGr, :KitchenQual,
                    :TotRmsAbvGrd, :Functional, :Fireplaces,
                    :FireplaceQu, :GarageType, :GarageYrBlt,
                    :GarageFinish, :GarageCars, :GarageArea,
                    :GarageQual, :GarageCond, :PavedDrive,
                    :WoodDeckSF, :OpenPorchSF, :EnclosedPorch,
                    :x3SsnPorch, :ScreenPorch, :PoolArea, :MiscVal,
                    :MoSold, :YrSold, :SaleType, :SaleCondition)
@test scitype(y) <: AbstractVector{Continuous}

X, y = @load_reduced_ames
schema(X).names == (:OverallQual, :GrLivArea, :Neighborhood,
                    :x1stFlrSF, :TotalBsmtSF, :BsmtFinSF1, :LotArea,
                    :GarageCars, :MSSubClass, :GarageArea, :YearRemodAdd,
                    :YearBuilt)
@test scitype(y) <: AbstractVector{Continuous}

X, y = @load_iris
@test schema(X).names == (:sepal_length, :sepal_width, :petal_length,
                          :petal_width)
@test scitype(y) <: AbstractVector{<:Multiclass}

X, y = @load_crabs
@test schema(X).names ==  (:FL, :RW, :CL, :CW, :BD)
@test scitype(y) <: AbstractVector{<:Multiclass}

X, y = @load_smarket
@test schema(X).names == (:Year, :Lag1, :Lag2, :Lag3, :Lag4, :Lag5, :Volume, :Today)
@test scitype(y) == AbstractVector{Multiclass{2}}

X = @load_sunspots
@test schema(X).names == (:sunspot_number, )

end # module
true
