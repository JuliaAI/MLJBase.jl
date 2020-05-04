module TestStaticTransformers

# using Revise
using Test
using MLJBase
using ..Models
using CategoricalArrays
import Random.seed!
seed!(1234)


struct MyTransformer1 <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer1, verbosity, X) =
    selectcols(X, transf.ftr)

@testset "nodal machine constructor for static transformers" begin
    X = (x1=rand(3), x2=[1, 2, 3]);
    mach = machine(MyTransformer1(:x2))
    fit!(mach)
    @test transform(mach, X) == [1, 2, 3]
end

x1 = rand(30)
x2 = rand(30)
x3 = rand(30)
y = exp.(x1 - x2 -2x3 + 0.1*rand(30))
X = (x1=x1, x2=x2, x3=x3)

f(X) = (a=selectcols(X, :x1), b=selectcols(X, :x2))

knn = @load KNNRegressor

# 1. function in a pipeline:
comp1 = @pipeline Comp1(f, std=Standardizer(),
                        rgs=knn,
                        target=UnivariateBoxCoxTransformer())
e = evaluate(comp1, X, y, measure=mae, resampling=Holdout(), verbosity=0)

# 2. function with parameters in a pipeline:
mutable struct MyTransformer <: Static
    ftr::Symbol
end
MLJBase.transform(transf::MyTransformer, verbosity, X) =
    (a=selectcols(X, transf.ftr), b=selectcols(X, :x2))
comp2 = @pipeline Comp2(transf=MyTransformer(:x3), std=Standardizer(),
                      rgs=knn,
                      target=UnivariateBoxCoxTransformer())

comp2.transf.ftr = :x1 # change the parameter
e2 =  evaluate(comp2, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e2.measurement[1] ≈ e.measurement[1]

# 3. function in an exported learning network:
Xs = source(X)
ys = source(y, kind=:target)
f(X::AbstractNode) = node(f, X)
# or, without arrow syntax:
# W = Xs |> f |> Standardizer()
# z = ys |> UnivariateBoxCoxTransformer()
# zhat = (W, z) |> knn
# yhat = zhat |> inverse_transform(z)
X2 = f(Xs)
W = transform(machine(Standardizer(), X2), X2)
box_mach = machine(UnivariateBoxCoxTransformer(), ys)
z = transform(box_mach, ys)
knn_mach = machine(knn, W, z)
zhat = predict(knn_mach, W)
yhat = inverse_transform(box_mach, zhat)
comp3 = @from_network Comp3(rgs=knn) <= yhat
e3 =  evaluate(comp3, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e2.measurement[1] ≈ e.measurement[1]

# 4. function with parameters in exported learning network:
inserter = MyTransformer(:x3)
# W = Xs |> inserter |> Standardizer()
# z = ys |> UnivariateBoxCoxTransformer()
# zhat = (W, z) |> knn
# yhat = zhat |> inverse_transform(z)
inserter_mach = machine(inserter)
X2 = transform(inserter_mach, Xs)
W = transform(machine(Standardizer(), X2), X2)
box_mach = machine(UnivariateBoxCoxTransformer(), ys)
z = transform(box_mach, ys)
knn_mach = machine(knn, W, z)
zhat = predict(knn_mach, W)
yhat = inverse_transform(box_mach, zhat)
comp4 = @from_network Comp4(transf=inserter, rgs=knn) <= yhat
comp4.transf.ftr = :x1 # change the parameter
e4 =  evaluate(comp4, X, y, measure=mae, resampling=Holdout(), verbosity=0)
@test e4.measurement[1] ≈ e.measurement[1]

end

true
