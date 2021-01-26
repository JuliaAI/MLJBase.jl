module TestPipelines

using MLJBase
using ..Models
using Test
using Statistics
using MLJScientificTypes
using StableRNGs

rng = StableRNG(698790187)

struct MyTransformer <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer, verbosity, X) =
    selectcols(X, transf.ftr)

NN = 7
X = MLJBase.table(rand(rng,NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(rng,NN);
Xs = source(X); ys = source(y)

broadcast_mode(v) = mode.(v)
doubler(y) = 2*y

@testset "pipeline helpers" begin
    @test MLJBase.super_type(KNNRegressor()) == Deterministic
    @test MLJBase.super_type(ConstantClassifier()) == Probabilistic
    @test MLJBase.super_type(Standardizer()) == Unsupervised
    @test MLJBase.super_type(:deterministic) == Deterministic
    @test MLJBase.super_type(:probabilistic) == Probabilistic
    @test MLJBase.super_type(:something) == Unsupervised
end

@testset "linear_learning_network" begin
    t = MLJBase.table
    m = MLJBase.matrix
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()

    models = [f, h]
    mach = MLJBase.linear_learning_network_machine(
        Unsupervised, Xs, nothing, nothing, nothing, nothing,
        true, predict, models...)
    tree= mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedSurrogate
    @test tree.operation == transform
    @test tree.model == h
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1

    models = [f, k]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, nothing, nothing,
        true, predict, models...)
    tree= mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == predict
    @test tree.model == k
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1
    @test tree.train_arg2.source == ys

    models = [m, t]
    mach = MLJBase.linear_learning_network_machine(
        Unsupervised, Xs, nothing, nothing, nothing, nothing,
        true, predict, models...)
    tree= mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedSurrogate
    @test tree.operation == t
    @test tree.model == nothing
    @test tree.arg1.operation == m
    @test tree.arg1.model == nothing
    @test tree.arg1.arg1.source == Xs

    # with learned target transformation:
    models = [f, k]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, u, nothing,
        true, predict, models...)
    tree= mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == inverse_transform
    @test tree.model == u
    @test tree.arg1.operation == predict
    @test tree.arg1.model == k
    @test tree.arg1.arg1.operation == transform
    @test tree.arg1.arg1.model == f
    @test tree.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.train_arg1 == tree.arg1.arg1
    @test tree.arg1.train_arg2.operation == transform
    @test tree.arg1.train_arg2.model == u
    @test tree.arg1.train_arg2.arg1.source == ys
    @test tree.arg1.train_arg2.train_arg1.source == ys
    @test tree.train_arg1.source == ys

    # with static target transformation:
    models = [f, k]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, MLJBase.WrappedFunction(log),
        MLJBase.WrappedFunction(exp),
        true, predict, models...)
    tree= mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == transform
    @test tree.model.f == exp
    @test tree.arg1.operation == predict
    @test tree.arg1.model == k
    @test tree.arg1.arg1.operation == transform
    @test tree.arg1.arg1.model == f
    @test tree.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.train_arg1 == tree.arg1.arg1
    @test tree.arg1.train_arg2.operation == transform
    @test tree.arg1.train_arg2.model.f == log
    @test tree.arg1.train_arg2.arg1.source == ys

    # with supervised model not at end and static target transformation:
    models = [f, c, broadcast_mode]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, MLJBase.WrappedFunction(log),
        MLJBase.WrappedFunction(exp),
        true, predict, models...)
    tree= mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == transform
    @test tree.model.f == exp
    @test tree.arg1.operation == broadcast_mode
    @test tree.arg1.model == nothing
    @test tree.arg1.arg1.operation == predict
    @test tree.arg1.arg1.model == c
    @test tree.arg1.arg1.arg1.operation == transform
    @test tree.arg1.arg1.arg1.model == f
    @test tree.arg1.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.arg1.train_arg1 == tree.arg1.arg1.arg1
    @test tree.arg1.arg1.train_arg2.operation == transform
    @test tree.arg1.arg1.train_arg2.model.f == log
    @test tree.arg1.arg1.train_arg2.arg1.source == ys

    # with supervised model not at end and with learned target transformation:
    models = [f, c, broadcast_mode]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, u, nothing,
        true, predict, models...)
    tree = mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == inverse_transform
    @test tree.model == u
    @test tree.arg1.operation == broadcast_mode
    @test tree.arg1.model == nothing
    @test tree.arg1.arg1.operation == predict
    @test tree.arg1.arg1.model == c
    @test tree.arg1.arg1.arg1.operation == transform
    @test tree.arg1.arg1.arg1.model == f
    @test tree.arg1.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.arg1.train_arg1 == tree.arg1.arg1.arg1
    @test tree.arg1.arg1.train_arg2.operation == transform
    @test tree.arg1.arg1.train_arg2.model == u
    @test tree.arg1.arg1.train_arg2.arg1.source == ys
    @test tree.arg1.arg1.train_arg2.train_arg1.source == ys
    @test tree.train_arg1.source == ys

    # test of invert_last=false - learned target transformation:
    models = [k, doubler]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, u, nothing,
        false, predict, models...)
    tree = mach.fitresult.predict |> MLJBase.tree
    @test tree.operation == doubler
    @test tree.arg1.operation == inverse_transform
    @test tree.arg1.arg1.operation == predict
    @test tree.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.arg1.train_arg2.operation == transform
    @test tree.arg1.arg1.train_arg2.arg1.source == ys
    @test tree.arg1.arg1.train_arg2.train_arg1.source == ys
    @test tree.arg1.train_arg1.source == ys

    # test of invert_last=false - static target transformation:
    models = [k, doubler]
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, MLJBase.WrappedFunction(log),
        MLJBase.WrappedFunction(exp),
        false, predict, models...)
    tree = mach.fitresult.predict |> MLJBase.tree
    @test tree.operation == doubler
    @test tree.model == nothing
    @test tree.arg1.operation == transform
    @test tree.arg1.model.f == exp
    @test tree.arg1.arg1.operation == predict
    @test tree.arg1.arg1.arg1.source == Xs
    @test tree.arg1.arg1.train_arg1.source == Xs
    @test tree.arg1.arg1.train_arg2.operation == transform
    @test tree.arg1.arg1.train_arg2.model.f == log
    @test tree.arg1.arg1.train_arg2.arg1.source == ys

    # check a probablistic case:
    models = [f, c]
    mach = MLJBase.linear_learning_network_machine(
        Probabilistic, Xs, ys, nothing, u, nothing,
        true, predict, models...)
    @test mach.model isa ProbabilisticSurrogate

    # check a static case:
    models = [m, t]
    mach = MLJBase.linear_learning_network_machine(
        Static, Xs, ys, nothing, nothing, nothing,
        true, predict, models...)
    @test mach.model isa StaticSurrogate

    # build a linear network for training:
    mach = MLJBase.linear_learning_network_machine(
        Deterministic, Xs, ys, nothing, u, nothing,
        true, predict, f, k)

    # build the same network by hand:
    fM = machine(f, Xs)
    Xt = transform(fM, Xs)
    uM = machine(u, ys)
    yt = transform(uM, ys)
    kM = machine(k, Xt, yt)
    zhat = predict(kM, Xt)
    N2 = inverse_transform(uM, zhat)

    # compare predictions
    fit!(mach); fit!(N2)
    yhat = predict(mach, X);
    @test yhat ≈ N2()
    k.K = 3; f.features = [:x3,]
    fit!(mach); fit!(N2)
    @test !(yhat ≈ predict(mach, X))
    @test predict(mach, X) ≈ N2()
    global hand_built = predict(mach, X);

end


## PIPELINE_PREPROCESS

F = FeatureSelector()
H = OneHotEncoder()
K = KNNRegressor()
C = ConstantClassifier()
U = UnivariateStandardizer()

m = :(MLJBase.matrix)
t = :(MLJBase.table)
f = :(F)
h = :(H)
k = :(K)
c = :(C)
e = :(target=exp)
l = :(inverse=log)
u = :(target=U)
p = :(prediction_type=:deterministic)
v = :(invert_last=true)

exs = [f, m, t, h, k, u, p, v]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test string(out[1])[1:8]  == "Pipeline"
@test out[2] ==  [:feature_selector, :one_hot_encoder,
                  :knn_regressor, :target]
@test eval.(out[3]) == [F, H, K, U]
@test eval.(out[4]) ==
    [Unsupervised, Unsupervised, Deterministic, Unsupervised]
@test eval.(out[5]) == [F, MLJBase.matrix, MLJBase.table, H, K]
@test eval.(out[6]) == U
@test eval.(out[7]) == nothing
@test out[8] == true
@test out[9] == predict
@test out[10] == Deterministic
@test haskey(out[11], :supports_weights)

exs =  [f, m, t, h, k, e, l, p, v]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[2] == [:feature_selector, :one_hot_encoder, :knn_regressor,
                 :target, :inverse]
@test eval.(out[3]) == [F, H, K, MLJBase.WrappedFunction(exp),
                 MLJBase.WrappedFunction(log)]
@test eval.(out[4]) == [Unsupervised, Unsupervised,
                        Deterministic, Unsupervised, Unsupervised]
@test eval.(out[5]) == [F, MLJBase.matrix, MLJBase.table, H, K]
@test eval.(out[6]) == MLJBase.WrappedFunction(exp)
@test eval.(out[7]) == MLJBase.WrappedFunction(log)
@test out[10] == Deterministic

# invert_last = false:
exs = [k, :(doubler), u]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[8] == false

exs = [m, t, :(MyTransformer(:age))]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[10] == Static

exs = [f, m, t, h, k, u, p, :(name=MyPipe)]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[1] == :MyPipe

exs = [f, m, t, h, k, u, p, :(name="MyPipe2")]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[1] == :MyPipe2

exs = [f, m, t, h, k, u, p, :(name=:MyPipe3)]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[1] == :MyPipe3

# auto-detect probablistic predictions:
exs = [f, m, t, h, c]
out = MLJBase.pipeline_preprocess(TestPipelines, exs...)
@test out[10] == Probabilistic

# prediction_type=:probabilistic declared but no supervised models
exs = [f, m, t, h, :(prediction_type=:probabilistic)]
@test_logs (:warn, r"Pipeline appears to have no") begin
    MLJBase.pipeline_preprocess(TestPipelines, exs...)
end

exs = [f, m, t, h, k, u]
@test_logs (:info, r"Treating") begin
    MLJBase.pipeline_preprocess(TestPipelines, exs...)
end

# unsupported operation:
exs = [h, k, :(operation=transform)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))
exs = [h, k, :(operation=:predict)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# pipe_line error 3:
exs =  Symbol[]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# target is a function but no inverse=...:
exs =  [f, m, t, h, k, e, p]
@test_throws(ArgumentError,
MLJBase.pipeline_preprocess(TestPipelines, exs...))

# inverse but no target:
exs =  [f, k, l, p]
@test_throws(ArgumentError,
MLJBase.pipeline_preprocess(TestPipelines, exs...))

# target specified but no component is supervised:
exs =  [f, h, u]
@test_throws(ArgumentError,
MLJBase.pipeline_preprocess(TestPipelines, exs...))

# pipe_line error 8:
exs = [f, m, t, h, k, u, p, :("mary"=true)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# target has wrong form:
exs = [f, m, t, h, k, u, p, :(target=KNNRegressor)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))
exs = [f, m, t, h, k, u, p, :(target=KNNRegressor())]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))
exs = [f, m, t, h, k, u, p, :(target="junk")]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# inverse not a function:
exs =  [f, m, t, h, k, e, p, :(inverse=Standardizer())]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# bad name for pipeline:
exs = [f, m, t, h, k, u, p, :(name=true)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# existing name:
exs = [f, m, t, h, k, u, :(name=Int)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# bad prediction_type:
exs = [f, m, t, h, k, u, :(prediction_type=:junk)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# invalid key-word
exs = [f, m, t, h, k, u, :(junk=:junk)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# non-expression:
exs = [f, m, t, h, k, u, :("junk")]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# junk expression:
exs = [f, m, t, h, k, u, :(sin(2))]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# datatype that's not a model:
exs = [f, m, t, h, k, u, :Int]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

# More than one component of the pipeline is a supervised model:
exs = [f, m, t, h, k, u, k]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

exs = [f, m, k, :(prediction_type=:probabilistic)]
@test_throws(ArgumentError,
             MLJBase.pipeline_preprocess(TestPipelines, exs...))

exs = [h, c, u]
@test_logs((:info, r"Treating"),
           (:warn, r"Pipeline is applying"), 
           MLJBase.pipeline_preprocess(TestPipelines, exs...))


## SIMPLE SUPERVISED PIPELINE WITH TARGET TRANSFORM

# test a simple pipeline prediction agrees with prediction of
# hand-built learning network built earlier:
p = @pipeline(FeatureSelector, KNNRegressor,
              target=UnivariateStandardizer,
              prediction_type=:deterministic)
p.knn_regressor.K = 3; p.feature_selector.features = [:x3,]
mach = machine(p, X, y)
fit!(mach)
@test MLJBase.tree(mach.fitresult.predict).arg1.model.K == 3
MLJBase.tree(mach.fitresult.predict).arg1.arg1.model.features == [:x3, ]
@test predict(mach, X) ≈ hand_built

# test a simple probabilistic classifier pipeline:
X = MLJBase.table(rand(rng,7,3));
y = categorical(collect("ffmmfmf"));
Xs = source(X)
ys = source(y)
p = @pipeline OneHotEncoder ConstantClassifier
mach = machine(p, X, y)
fit!(mach)
@test p isa ProbabilisticComposite
pdf(predict(mach, X)[1], 'f') ≈ 4/7

# test invalid replacement of classifier with regressor throws
# informative error message:
p.constant_classifier = ConstantRegressor()
@test_logs((:error, r"^Problem"),
           (:info, r"^Running type"),
           (:warn, r"The scitype of"),
           (:error, r"Problem"),
           @test_throws Exception fit!(mach, verbosity=-1))

# test a simple deterministic classifier pipeline:
X = MLJBase.table(rand(rng,7,3))
y = categorical(collect("ffmmfmf"))
Xs = source(X)
ys = source(y)
p = @pipeline(OneHotEncoder, ConstantClassifier, broadcast_mode,
              prediction_type=:probabilistic)
mach = machine(p, X, y)
fit!(mach)
@test predict(mach, X) == fill('f', 7)

# test pipelines with weights:
w = map(y) do η
    η == 'm' ? 100 : 1
end
mach = machine(p, X, y, w)
fit!(mach)
@test predict(mach, X) == fill('m', 7)

# test a pipeline with static transformation of target:
NN = 100
X = (x1=rand(rng,NN), x2=rand(rng,NN), x3=categorical(rand(rng,"abc", NN)));
y = 1000*abs.(2X.x1 - X.x2 + 0.05*rand(rng,NN))
# by hand:
Xs =source(X); ys = source(y)
hot = OneHotEncoder()
hot_=machine(hot, Xs)
W = transform(hot_, Xs)
sel = FeatureSelector(features=[:x1,:x3__a])
sel_ = machine(sel, W)
Wsmall = transform(sel_, W)
z = log(ys)
knn = KNNRegressor(K=4)
knn_ = machine(knn, Wsmall, z)
zhat = predict(knn_, Wsmall)
yhat = exp(zhat)
fit!(yhat)
pred1 = yhat();
# with pipeline:
p = @pipeline(OneHotEncoder,
              FeatureSelector,
              KNNRegressor,
              target=v->log.(v),
              inverse=v->exp.(v))
p.feature_selector.features = [:x1, :x3__a]
p.knn_regressor.K = 4
p_ = machine(p, X, y)
fit!(p_)
pred2 = predict(p_, X);
@test pred1 ≈ pred2

# and another:
age = [23, 45, 34, 25, 67]
X = (age = age,
     gender = categorical(['m', 'm', 'f', 'm', 'f']))
height = [67.0, 81.5, 55.6, 90.0, 61.1]
p = @pipeline(X -> coerce(X, :age=>Continuous),
              OneHotEncoder,
              KNNRegressor(K=3),
              target = UnivariateStandardizer,
              prediction_type = :probabilistic)
fit!(machine(p, X, height))


## STATIC TRANSFORMERS IN PIPELINES

p99 = @pipeline(X -> coerce(X, :age=>Continuous),
                OneHotEncoder,
                MyTransformer(:age))

mach  = machine(p99, X) |> fit!

@test transform(mach, X) == float.(X.age)


## PURE STATIC PIPLINES

p = @pipeline(X -> coerce(X, :age=>Continuous),
                MyTransformer(:age))

# no training arguments!
mach = machine(p) |> fit!
@test transform(mach, X) == X.age

p = @pipeline exp log x-> 2*x
mach = machine(p) |> fit!
@test transform(mach, 20) ≈ 40


## OPERATION DIFFERENT FROM PREDICT

p = @pipeline(OneHotEncoder,
              ConstantRegressor,
              operation=predict_mean)

@test p isa Deterministic

mach = machine(p, X, height) |> fit!
@test scitype(predict(mach, X)) == AbstractVector{Continuous}


## USE OF INVERT_LAST

X = (x1 = float.(1:10),)
y = X.x1

p = @pipeline(KNNRegressor(K=1), doubler, target=UnivariateStandardizer)

mach = machine(p, X, y) |> fit!
@test predict(mach, X) ≈ doubler(y)

p = @pipeline(KNNRegressor(K=1), doubler, target=UnivariateStandardizer,
              invert_last=true)
mach = machine(p, X, y) |> fit!
@test !(predict(mach, X) ≈ doubler(y))

end


## OPERATION KEYWORD


true
