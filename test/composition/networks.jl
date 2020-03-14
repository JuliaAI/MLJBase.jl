module TestLearningNetworks

# using Revise
using Test
using MLJBase
using ..Models
using CategoricalArrays
import Random.seed!
seed!(1234)

@load KNNRegressor

@testset "network #1" begin

    N =100
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = 2X.x1  - X.x2 + 0.05*rand(N)

    knn_ = KNNRegressor(K=7)

    # split the rows:
    allrows = eachindex(y);
    train, valid, test = partition(allrows, 0.7, 0.15);
    @test vcat(train, valid, test) == allrows

    Xtrain = selectrows(X, train)
    ytrain = y[train]

    Xs = source(Xtrain)
    ys = source(ytrain)

    knn1 = machine(knn_, Xs, ys)
    @test_logs (:info, r"Training") fit!(knn1, verbosity=3)
    knn_.K = 5
    @test_logs((:info, r"Training"),
               fit!(knn1, rows=train[1:end-10], verbosity=2))
    @test_logs (:info, r"Training") fit!(knn1, verbosity=2)
    yhat = predict(knn1, Xs)
    yhat(selectrows(X, test))
    @test rms(yhat(selectrows(X, test)), y[test]) < 0.3
    @test MLJBase.is_stale(knn1) == false

    fit!(yhat)
    pred = yhat()

    # test serialization of NodalMachine:
    io = IOBuffer()
    MLJBase.save(io, knn1)
    seekstart(io)
    mach = fit!(machine(io, Xtrain, ytrain))
    @test predict(mach) ≈ pred

end

@testset "network #2" begin

    N =100
    X = (x1=rand(N),
         x2=rand(N),
         x3=categorical(rand("yn",N)),
         x4=categorical(rand("yn",N)))

    y = 2X.x1  - X.x2 + 0.05*rand(N)
    X = source(X)
    y = source(y)

    hot = OneHotEncoder()
    hotM = machine(hot, X)
    W = transform(hotM, X)
    knn = KNNRegressor()
    knnM = machine(knn, W, y)
    yhat = predict(knnM, W)

    # should get "Training" for both:
    @test_logs (:info, r"^Training") (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

    # should get "Not retraining" for both:
    @test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)

    # should get "Updating" for first, "Training" for second:
    hot.drop_last = true
    @test_logs (:info, r"^Updating")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

    # should get "Not retraining" for both:
    @test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)

    # should get "Not retraining" for first, "Updating for second":
    knn.K = 17
    @test_logs (:info, r"^Not retraining") (:info, r"^Updating") fit!(yhat)

    # should get "Training" for both:
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, rows=1:100)

    # should get "Training" for both"
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

    # should get "Training" for both"
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, force=true)

end

@testset "network #3" begin

    N =100
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = 2X.x1  - X.x2 + 0.05*rand(N)

    XX = source(X)
    yy = source(y)

    # construct a transformer to standardize the target:
    uscale_ = UnivariateStandardizer()
    uscale = machine(uscale_, yy)

    # get the transformed inputs, as if `uscale` were already fit:
    z = transform(uscale, yy)

    # construct a transformer to standardize the inputs:
    scale_ = Standardizer()
    scale = machine(scale_, XX) # no need to fit

    # get the transformed inputs, as if `scale` were already fit:
    Xt = transform(scale, XX)

    # do nothing to the DataFrame
    Xa = node(identity, Xt)

    # choose a learner and make it machine:
    knn_ = KNNRegressor(K=7) # just a container for hyperparameters
    knn = machine(knn_, Xa, z) # no need to fit

    # get the predictions, as if `knn` already fit:
    zhat = predict(knn, Xa)

    # inverse transform the target:
    yhat = inverse_transform(uscale, zhat)

    # fit-through training:
    @test_logs((:info, r"Training"),
               (:info, r"Features standarized: "),
               (:info, r" *:x1"),
               (:info, r" *:x2"),
               (:info, r" *:x3"),
               (:info, r"Training"),
               (:info, r"Training"),
               fit!(yhat, rows=1:50, verbosity=2))
    @test_logs(
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        fit!(yhat, rows=1:50, verbosity=1))
    @test_logs(
        (:info, r"Training"),
        (:info, r"Training"),
        (:info, r"Training"),
        fit!(yhat, verbosity=1))
    knn_.K =67
    @test_logs(
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        (:info, r"Updating"),
        fit!(yhat, verbosity=1))

end

@testset "overloading methods for AbstractNode" begin
    A  = rand(3,7)
    As = source(A)
    @test MLJBase.matrix(MLJBase.table(As))() == A

    X = (x1 = [1,2,3], x2=[10, 20, 30], x3=[100, 200, 300])
    Xs = source(X)
    @test selectrows(Xs, 1)() == selectrows(X, 1)
    @test selectrows(Xs, 2:3)() == selectrows(X, 2:3)
    @test selectcols(Xs, 1)() == selectcols(X, 1)
    @test selectcols(Xs, 2:3)() == selectcols(X, 2:3)
    @test selectcols(Xs, :x1)() == selectcols(X, :x1)
    @test selectcols(Xs, [:x1, :x3])() == selectcols(X, [:x1, :x3])

    y = rand(4)
    ys = source(y)
    @test vcat(ys, ys)() == vcat(y, y)
    @test hcat(ys, ys)() == hcat(y, y)
    @test log(ys)() == log.(y)
    @test exp(ys)() == exp.(y)

    Z = (rand(4), rand(4), rand(4))
    Zs = source(Z)
    @test mean(Zs)() == mean.(Z)
    @test mode(Zs)() == mode.(Z)
    @test median(Zs)() == median.(Z)

    a, b, λ = rand(4), rand(4), rand()
    as, bs = source(a), source(b)
    @test (as + bs)() == a + b
    @test (λ * bs)() == λ * b
end

struct MyTransformer1 <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer1, verbosity, X) =
    selectcols(X, transf.ftr)

@testset "nodal machine constructor for static transformers" begin
    X = (x1=rand(3), x2=[1, 2, 3])
    mach = machine(MyTransformer1(:x2))
    fit!(mach)
    @test transform(mach, X) == [1, 2, 3]
end


## MORE STATIC TRANSFORMER TESTS

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
