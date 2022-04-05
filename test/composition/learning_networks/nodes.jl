module TestLearningCompositesCore

# using Revise
using Test
using MLJBase
using ..Models
using ..TestUtilities
using CategoricalArrays
import Random.seed!
seed!(1234)

N =100
X = (x1=rand(N), x2=rand(N), x3=rand(N));
y = 2X.x1  - X.x2 + 0.05*rand(N);

@testset "error messages for invalid learning networks" begin

    Xs = source(X)
    ys = source(y)
    mach1 = machine(Standardizer(), Xs)
    W = transform(mach1, Xs)
    fit!(W, verbosity=0)
    @test_logs((:error, r"Failed"), @test_throws Exception W(34))

    mach2 = machine(DecisionTreeClassifier(), W, ys)
    yhat = predict(mach2, W)
    @test_logs((:error, r"Problem fitting"),
               (:info, r"Running type checks"),
               (:warn, MLJBase.warn_generic_scitype_mismatch(
                   scitype((X, y)),
                   MLJBase.fit_data_scitype(mach2.model),
                   typeof(mach2.model)
               )),
               (:info, r"^It seems an upstream"),
               (:error, r"^Problem fitting"),
               @test_throws Exception fit!(yhat, verbosity=-1))
    Xs = source()
    mach1 = machine(Standardizer(), Xs)
    W = transform(mach1, Xs)
    @test_logs((:error, r"Problem fitting"),
               (:warn, r"^Some learning network source"),
               (:info, r"Running type checks"),
               (:warn, MLJBase.warn_generic_scitype_mismatch(
                   Tuple{Nothing},
                   MLJBase.fit_data_scitype(mach1.model),
                   typeof(mach1.model)
               )),
               (:info, r"^It seems an upstream"),
               (:error, r"^Problem fitting"),
               @test_throws Exception fit!(W, verbosity=-1))
end

@testset "network #1" begin

    knn_ = KNNRegressor(K=7)

    # split the rows:
    allrows = eachindex(y);
    train, valid, test = partition(allrows, 0.7, 0.15);

    Xtrain = selectrows(X, train);
    ytrain = y[train];

    Xs = source(Xtrain);
    ys = source(ytrain);

    knn1 = machine(knn_, Xs, ys)
    @test_mach_sequence fit_only!(knn1) [(:train, knn1),]
    knn_.K = 5
    @test_logs((:info, r"Training"),
               fit_only!(knn1, rows=train[1:end-10], verbosity=2))
    @test_logs (:info, r"Training") fit_only!(knn1, verbosity=2)
    yhat = predict(knn1, Xs);
    yhat(selectrows(X, test))
    @test rms(yhat(selectrows(X, test)), y[test]) < 0.3

    @test_mach_sequence fit!(yhat) [(:skip, knn1),]
    pred = yhat();
end

@testset "network #2" begin

    N =100
    X = (x1=rand(N),
         x2=rand(N),
         x3=categorical(rand("yn",N)),
         x4=categorical(rand("yn",N)));

    y = 2X.x1  - X.x2 + 0.05*rand(N);
    X = source(X)
    y = source(y)

    hot = OneHotEncoder()
    hotM = machine(hot, X)
    W = transform(hotM, X)
    @test scitype(W) == CallableReturning{Table}

    knn = KNNRegressor()
    knnM = machine(knn, W, y)
    yhat = predict(knnM, W)
    @test scitype(yhat) ==
        MLJBase.CallableReturning{AbstractVector{Continuous}}

    @test_mach_sequence fit!(yhat) [(:train, hotM), (:train, knnM)]
    @test_mach_sequence fit!(yhat) [(:skip, hotM), (:skip, knnM)]

    hot.drop_last = true
    @test_mach_sequence fit!(yhat) [(:update, hotM), (:train, knnM)]
    @test_mach_sequence fit!(yhat) [(:skip, hotM), (:skip, knnM)]

    knn.K = 17
    @test_mach_sequence fit!(yhat) [(:skip, hotM), (:update, knnM)]

    # change rows:
    @test_mach_sequence fit!(yhat, rows=1:100) [(:train, hotM), (:train, knnM)]

    # change rows again:
    @test_mach_sequence fit!(yhat) [(:train, hotM), (:train, knnM)]

    # force:
    @test_mach_sequence fit!(yhat, force=true) [(:train, hotM), (:train, knnM)]

end

@testset "with parallel regressors and pre-processors" begin
    N =100
    X = (x1=rand(N),
         x2=rand(N),
         x3=categorical(rand("yn",N)),
         x4=categorical(rand("yn",N)));

    y = abs.(2X.x1  - X.x2 + 0.05*rand(N));
    X = source(X)
    y = source(y)

    cox = UnivariateBoxCoxTransformer()
    coxM = machine(cox, y, cache=true)
    z = transform(coxM, y)

    hot = OneHotEncoder()
    hotM = machine(hot, X, cache=true)
    W = transform(hotM, X)

    knn = KNNRegressor()
    knnM = machine(knn, W, z, cache=true)
    zhat1 = predict(knnM, W)

    @test_mach_sequence fit!(W) [(:train, hotM), ]
    @test_mach_sequence fit!(z) [(:train, coxM), ]
    evaluate!(knnM, measure=l2, verbosity=0);
    fit!(knnM, verbosity=0)

    cox.shift=true
    @test_mach_sequence(fit!(zhat1),
                   [(:update, coxM), (:skip, hotM), (:train, knnM)],
                   [(:skip, hotM), (:update, coxM), (:train, knnM)])

    # no training:
    @test_mach_sequence(fit!(zhat1),
                   [(:skip, coxM), (:skip, hotM), (:skip, knnM)],
                   [(:skip, hotM), (:skip, coxM), (:skip, knnM)])


    tree = DecisionTreeRegressor()
    treeM = machine(tree, W, z, cache=true)
    zhat2 = predict(treeM, W)

    zhat = 0.5*zhat1 + 0.5*zhat2
    @test elscitype(zhat) == Unknown

    @test_mach_sequence(fit!(zhat),
                   [(:skip, coxM), (:skip, hotM),
                     (:skip, knnM), (:train, treeM)],
                   [(:skip, hotM), (:skip, coxM),
                    (:skip, knnM), (:train, treeM)],
                   [(:skip, coxM), (:skip, hotM),
                    (:train, treeM), (:skip, knnM)],
                   [(:skip, hotM), (:skip, coxM),
                    (:train, treeM), (:skip, knnM)])

    yhat = inverse_transform(coxM, zhat)

    @test_mach_sequence(fit!(yhat),
                   [(:skip, coxM), (:skip, hotM),
                     (:skip, knnM), (:skip, treeM)],
                   [(:skip, hotM), (:skip, coxM),
                    (:skip, knnM), (:skip, treeM)],
                   [(:skip, coxM), (:skip, hotM),
                    (:skip, treeM), (:skip, knnM)],
                   [(:skip, hotM), (:skip, coxM),
                    (:skip, treeM), (:skip, knnM)])

    # error handling:
    MLJBase.rebind!(X, "junk5")
    @test_logs((:error, r""),
               (:error, r""),
               (:error, r""),
               (:error, r""),
               @test_throws Exception fit!(yhat, verbosity=-1, force=true))

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
    @test_mach_sequence(fit!(yhat, rows=1:50),
                   [(:train, uscale), (:train, scale), (:train, knn)],
                   [(:train, scale), (:train, uscale), (:train, knn)])

    @test_mach_sequence(fit!(yhat, rows=1:50),
                   [(:skip, uscale), (:skip, scale), (:skip, knn)],
                   [(:skip, scale), (:skip, uscale), (:skip, knn)])

    # change rows:
    @test_mach_sequence(fit!(yhat),
                   [(:train, uscale), (:train, scale), (:train, knn)],
                   [(:train, scale), (:train, uscale), (:train, knn)])

    knn_.K =67
    @test_mach_sequence(fit!(yhat),
                   [(:skip, uscale), (:skip, scale), (:update, knn)],
                   [(:skip, scale), (:skip, uscale), (:update, knn)])
end

@testset "network with machines sharing one model" begin

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
    xscale_ = Standardizer()
    xscale = machine(xscale_, XX) # no need to fit

    # get the transformed inputs, as if `scale` were already fit:
    Xt = transform(xscale, XX)

    # choose a learner and make two machines from it:
    knn_ = KNNRegressor(K=7) # just a container for hyperparameters
    knn1 = machine(knn_, Xt, z) # no need to fit
    knn2 = machine(knn_, Xt, z) # no need to fit

    # get the predictions, as if `knn` already fit:
    zhat1 = predict(knn1, Xt)
    zhat2 = predict(knn2, Xt)
    zhat = zhat1 + zhat2

    # inverse transform the target:
    yhat = inverse_transform(uscale, zhat)

    fit!(yhat, verbosity=0)

    θ = fitted_params(yhat)

end

mutable struct PermuteArgs <: MLJBase.Static
    permutation::NTuple{N,Int} where N
end

MLJBase.transform(p::PermuteArgs, _, args...) =
    Tuple([args[i] for i in p.permutation])
MLJBase.inverse_transform(p::PermuteArgs, _, args...) =
    Tuple([args[i] for i in sortperm(p.permutation |> collect)])

@testset "nodes wrapping Static transformers can be called with rows" begin

    y1 = source(10*ones(Int, 3))
    y2 = source(20*ones(Int, 3))
    y3 = source(30*ones(Int, 3))

    permuter = PermuteArgs((2, 3, 1))
    mach = machine(permuter)
    y = transform(mach, y1, y2, y3)

    @test y(rows=1:2) == (20*ones(Int, 2), 30*ones(Int, 2), 10*ones(Int, 2))

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
    @test nrows(Xs)() == 3

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

    X = source(1)
    y = source(7)
    @test glb(X, y)() == (1, 7)
    glb_node = @tuple X y
    @test glb_node() == (1, 7)

    X = source(1:10)
    Y = node(X->selectrows(X, 3:4), X)
    @test nrows_at_source(Y) == 10
    @test nrows(Y)() == 2

end

@testset "reformat logic - with upstream transformer learning from data" begin
    X = (x1=rand(5), x2=rand(5))
    y = categorical(collect("abaaa"))

    Xs = source(X)
    ys = source(y)

    std = Standardizer()
    mach1 = machine(std, Xs)
    W = transform(mach1, Xs)

    # a classifier with reformat front-end:
    clf = ConstantClassifier(testing=true)
    mach2 = machine(clf, W, ys; cache=true)
    yhat = predict(mach2, W)
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(yhat, verbosity=0, rows=1:3))

    # training network with new rows changes upstream state of
    # classifier and hence retriggers reformatting of data:
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(yhat, verbosity=0, rows=1:2))

    # however just changing classifier hyper-parameter avoids
    # reformatting and resampling:
    clf.bogus = 123
    @test_logs fit!(yhat, verbosity=0, rows=1:2)
end

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

function MLJBase.transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X * s.scaling
    MLJBase.table(s.scaling * MLJBase.matrix(X), prototype=X)
end

function MLJBase.inverse_transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X / s.scaling
    MLJBase.table(MLJBase.matrix(X) / s.scaling, prototype=X)
end

@testset "reformat logic - with upstream Static transformer" begin
    X = (x1=ones(5), x2=ones(5))
    y = categorical(collect("abaaa"))

    Xs = source(X)
    ys = source(y)

    scaler = Scale(2.0)
    mach1 = machine(scaler)
    W = transform(mach1, Xs)

    # a classifier with reformat front-end:
    clf = ConstantClassifier(testing=true)
    mach2 = machine(clf, W, ys, cache=true)
    yhat = predict(mach2, W)
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(yhat, verbosity=0, rows=1:3))

    # training network with new rows does not change upstream state of
    # classifier, because `scaler isa Static` and no reformatting of
    # data:
    @test_logs((:info, "resampling X, y"),
               fit!(yhat, verbosity=0, rows=1:2)
               )

    # however changing an upstream hyperparameter forces reforamatting:
    scaler.scaling = 3.0
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(yhat, verbosity=0, rows=1:2))

    # however just changing classifier hyper-parameter avoids
    # reformatting and resampling:
    clf.bogus = 123
    @test_logs fit!(yhat, verbosity=0, rows=1:2)
end

@testset "@node" begin
    X1 = source(4)
    X2 = source(5)
    Xp = source(1:10)

    add(a, b, c) = a + b + c
    N = @node add(X1, 1, X2)
    @test N() == 10

    N = @node tuple(X1, 5, X1)
    @test N() == (4, 5, 4)

    Y = @node selectrows(Xp, 3:4)
    @test Y() == 3:4
    @test Y([:one, :two, :three, :four]) == [:three, :four]
end

end

true
