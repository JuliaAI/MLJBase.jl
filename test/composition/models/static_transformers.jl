module TestWrappedFunctions

# using Revise
using Test
using MLJBase
using ..Models
using CategoricalArrays
import Random.seed!
seed!(1234)


struct PlainTransformer <: Static
    ftr::Symbol
end

MLJBase.transform(transf::PlainTransformer, verbosity, X) =
    selectcols(X, transf.ftr)

@testset "machine constructor for static transformers" begin
    X = (x1=rand(3), x2=[1, 2, 3]);
    mach = machine(PlainTransformer(:x2))
    @test transform(mach, X) == [1, 2, 3]
end

struct YourTransformer <: Static
    ftr::Symbol
end
MLJBase.reporting_operations(::Type{<:YourTransformer}) = (:transform,)

# returns `(output, report)`:
MLJBase.transform(transf::YourTransformer, verbosity, X) =
    (selectcols(X, transf.ftr), (; nrows=nrows(X)))

MLJBase.predict(transf::YourTransformer, verbosity, X) =
    collect(1:nrows(X)) |> reverse

@testset "nodal machine constructor for static transformers" begin
    X = (x1=rand(3), x2=[1, 2, 3]);
    mach = machine(YourTransformer(:x2))
    @test transform(mach, X) == [1, 2, 3]
    @test_throws MLJBase.ERR_ROWS_NOT_ALLOWED transform(mach, rows=:)
    @test predict(mach, X) == [3, 2, 1]
    @test report(mach).nrows == 3
    transform(mach, (x2=["a", "b"],))
    @test report(mach).nrows == 2
end

x1 = rand(30)
x2 = rand(30)
x3 = rand(30)
y = exp.(x1 - x2 -2x3 + 0.1*rand(30))
X = (x1=x1, x2=x2, x3=x3)

f(X) = (a=selectcols(X, :x1), b=selectcols(X, :x2))

knn = KNNRegressor()

# 1. function in a pipeline:
knn_target = TransformedTargetModel(knn, transformer=UnivariateBoxCoxTransformer())
comp1 =  f |> Standardizer() |> knn_target
e = evaluate(comp1, X, y, measure=mae, resampling=Holdout(), verbosity=0)

# 2. function with parameters in a pipeline:
mutable struct GreatTransformer <: Static
    ftr::Symbol
end
MLJBase.transform(transf::GreatTransformer, verbosity, X) =
    (a=selectcols(X, transf.ftr), b=selectcols(X, :x2))
comp2 = GreatTransformer(:x3) |> Standardizer() |>  knn_target
comp2.great_transformer.ftr = :x1 # change the parameter
e2 =  evaluate(comp2, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e2.measurement[1] ≈ e.measurement[1]

# 3. function in an `NetworkComposite`:

mutable struct Comp3 <: DeterministicNetworkComposite
    rgs
end

f(X::AbstractNode) = node(f, X)

function MLJBase.prefit(::Comp3, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    X2 = f(Xs) # f define in global scope
    W = transform(machine(Standardizer(), X2), X2)
    box_mach = machine(UnivariateBoxCoxTransformer(), ys)
    z = transform(box_mach, ys)
    knn_mach = machine(:rgs, W, z)
    zhat = predict(knn_mach, W)
    yhat = inverse_transform(box_mach, zhat)
    return (; predict=yhat)
end

comp3 = Comp3(knn)
e3 =  evaluate(comp3, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e2.measurement[1] ≈ e.measurement[1]

# 4. function with parameters in `NetworkComposite`:

mutable struct CC <: DeterministicNetworkComposite
    transf
    rgs
end

function MLJBase.prefit(::CC, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    inserter_mach = machine(:transf)
    X2 = transform(inserter_mach, Xs)
    W = transform(machine(Standardizer(), X2), X2)
    box_mach = machine(UnivariateBoxCoxTransformer(), ys)
    z = transform(box_mach, ys)
    knn_mach = machine(:rgs, W, z)
    zhat = predict(knn_mach, W)
    yhat = inverse_transform(box_mach, zhat)
    return (; predict=yhat)
end

inserter = GreatTransformer(:x3)
comp4 = CC(inserter, knn)

comp4.transf.ftr = :x1 # change the parameter
e4 =  evaluate(comp4, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e4.measurement[1] ≈ e.measurement[1]

end

true
