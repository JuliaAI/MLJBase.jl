module TestPipelines2

using MLJBase
using Test
using ..Models
using StableRNGs
using Tables

rng = StableRNG(698790187)

@testset "helpers" begin
    @test MLJBase.individuate([:x, :y, :x, :z, :y, :x]) ==
        [:x, :y, :x2, :z, :y2, :x3]
end


# # DUMMY MODELS

mutable struct MyTransformer2 <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer2, verbosity, X) =
    fill(:st, nrows(X))

mutable struct MyDeterministic <: Deterministic
    x::Symbol
end

MLJBase.fit(::MyDeterministic, args...) = nothing, nothing, nothing
MLJBase.transform(m::MyDeterministic, ::Any, Xnew) = fill(:dt, nrows(Xnew))
MLJBase.predict(m::MyDeterministic, ::Any, Xnew) = fill(:dp, nrows(Xnew))

mutable struct MyProbabilistic <: Probabilistic
    x::Symbol
end

mutable struct MyUnsupervised <: Unsupervised
    x::Symbol
end

MLJBase.fit(::MyUnsupervised, args...) = nothing, nothing, nothing
MLJBase.transform(m::MyUnsupervised, ::Any, Xnew) = fill(:ut, nrows(Xnew))
MLJBase.predict(m::MyUnsupervised, ::Any, Xnew) = fill(:up, nrows(Xnew))

mutable struct MyInterval <: Interval
    x::Symbol
end

d = MyDeterministic(:d)
p = MyProbabilistic(:p)
u = MyUnsupervised(:u)
s = MyTransformer2(:s) # Static
i = MyInterval(:i)
m = MLJBase.matrix
t = MLJBase.table

@testset "pipe_named_tuple" begin
    @test_throws MLJBase.ERR_EMPTY_PIPELINE MLJBase.pipe_named_tuple((),())
    @test_throws(MLJBase.ERR_TOO_MANY_SUPERVISED,
                 MLJBase.pipe_named_tuple((:foo, :foo, :foo), (d, u, p)))
    _names      = (:trf, :fun, :fun, :trf, :clf)
    components = (u,    m,    t,    u,    d)
    @test MLJBase.pipe_named_tuple(_names, components) ==
        NamedTuple{(:trf, :fun, :fun2, :trf2, :clf),
                   Tuple{Unsupervised,
                         Any,
                         Any,
                         Unsupervised,
                         Deterministic}}(components)
end

@testset "public constructor" begin
    # un-named components:
    @test Pipeline(m, t, u) isa UnsupervisedPipeline
    @test Pipeline(m, t, u, p) isa ProbabilisticPipeline
    @test Pipeline(m, t, u, p, operation=predict_mean) isa DeterministicPipeline
    @test Pipeline(u, p, u, operation=predict_mean) isa DeterministicPipeline
    @test Pipeline(m, t) isa StaticPipeline
    @test Pipeline(m, t, s) isa StaticPipeline
    @test Pipeline(m, t, s, d) isa DeterministicPipeline
    @test Pipeline(m, t, i) isa IntervalPipeline
    @test_logs((:info, MLJBase.INFO_TREATING_AS_DETERMINISTIC),
               @test Pipeline(m, t, u, p, u) isa DeterministicPipeline)
    @test_logs((:info, MLJBase.INFO_TREATING_AS_DETERMINISTIC),
               @test Pipeline(m, t, u, i, u) isa DeterministicPipeline)

    # if "hidden" supervised model is already deterministic,
    # no need for warning:
    @test_logs @test Pipeline(m, t, u, d, u) isa DeterministicPipeline

    # named components:
    @test Pipeline(c1=m, c2=t, c3=u) isa UnsupervisedPipeline
    @test Pipeline(c1=m, c2=t, c3=u, c5=p) isa ProbabilisticPipeline
    @test Pipeline(c1=m, c2=t) isa StaticPipeline
    @test Pipeline(c1=m, c2=t, c6=s) isa StaticPipeline
    @test Pipeline(c1=m, c2=t, c6=s, c7=d) isa DeterministicPipeline
    @test Pipeline(c1=m, c2=t, c8=i) isa IntervalPipeline
    @test_logs((:info, MLJBase.INFO_TREATING_AS_DETERMINISTIC),
               @test Pipeline(c1=m, c2=t, c3=u, c5=p, c4=u) isa
               DeterministicPipeline)
    @test(Pipeline(c1=m, c2=t, c3=u, c5=p, c4=u, prediction_type=:interval) isa
          IntervalPipeline)
    @test(Pipeline(c1=m, c2=t, c3=u, c5=p, c4=u,
                   prediction_type=:probabilistic) isa
          ProbabilisticPipeline)
    @test_logs((:info, MLJBase.INFO_TREATING_AS_DETERMINISTIC),
               @test Pipeline(c1=m, c2=t, c3=u, c8=i, c4=u) isa
               DeterministicPipeline)
    # if "hidden" supervised model is already deterministic,
    # no need for warning:
    @test_logs(@test Pipeline(c1=m, c2=t, c3=u, c7=d, c4=u) isa
               DeterministicPipeline)

    # errors and warnings:
    @test_throws MLJBase.ERR_MIXED_PIPELINE_SPEC Pipeline(m, mymodel=p)
    @test_throws(MLJBase.ERR_INVALID_OPERATION,
                 Pipeline(u, s, operation=cos))
    @test_throws(MLJBase.ERR_INVALID_PREDICTION_TYPE,
                 Pipeline(u=u, s=s, prediction_type=:ostrich))
    @test_logs((:warn, MLJBase.WARN_IGNORING_PREDICTION_TYPE),
               Pipeline(m, t, u, prediction_type=:deterministic))
    @test_throws(MLJBase.err_prediction_type_conflict(d, :probabilistic),
                 Pipeline(m, t, d, prediction_type=:probabilistic))

end

@testset "property access" begin
    pipe = Pipeline(m, u, u, s)

    # property names:
    @test propertynames(pipe) ===
        (:f, :my_unsupervised, :my_unsupervised2, :my_transformer2, :cache)

    # getproperty:
    @test pipe.my_unsupervised == u
    @test pipe.my_unsupervised2 == u
    @test pipe.cache == true

    # replacing a component with one whose abstract supertype is the same
    # or smaller:
    pipe.my_unsupervised = s
    @test pipe.my_unsupervised == s

    # attempting to replace a component with one whose abstract supertype
    # is bigger:
    @test_throws MethodError pipe.my_transformer2 = u

    # mutating the components themeselves:
    pipe.my_unsupervised.ftr = :z
    @test pipe.my_unsupervised.ftr == :z

    # or using MLJBase's recursive getproperty:
    MLJBase.recursive_setproperty!(pipe, :(my_unsupervised.ftr), :bonzai)
    @test pipe.my_unsupervised.ftr ==  :bonzai

    # more errors:
    @test_throws(MLJBase.err_pipeline_bad_property(pipe, :mount_fuji),
                 pipe.mount_fuji)
    @test_throws(MLJBase.err_pipeline_bad_property(pipe, :mount_fuji),
                 pipe.mount_fuji = 42)

end

@testset "show" begin
    io = IOBuffer()
    pipe = Pipeline(x-> x^2, m, t, p)
    show(io, MIME("text/plain"), pipe)
end

@testset "Front and extend" begin
    Xs = source(rand(3))
    ys = source(rand(3))
    front = MLJBase.Front(Xs, ys, false)
    @test front.predict() == Xs()
    @test front.transform() == ys()
    @test MLJBase.active(front)() == Xs()

    front = MLJBase.Front(Xs, Xs, true)
    front = MLJBase.extend(front, u, false, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, s, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:st, :st, :st]

    front = MLJBase.extend(front, u, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, d, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:dp, :dp, :dp]
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, x->string.(x), true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == string.([:dp, :dp, :dp])
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, u, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:ut, :ut, :ut]
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, s, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:st, :st, :st]
    @test tnode() == [:dt, :dt, :dt]
end

NN = 7
X = MLJBase.table(rand(rng, NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(rng,NN);
Xs = source(X); ys = source(y)

broadcast_mode(v) = mode.(v)
doubler(y) = 2*y

@testset "pipeline_network_machine" begin
    t = MLJBase.table
    m = MLJBase.matrix
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()

    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        Deterministic, true, predict, components, Xs, ys)
    tree = mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa DeterministicSurrogate
    @test tree.operation == predict
    @test tree.model == k
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1
    @test tree.train_arg2.source == ys

    components = [f, h]
    mach = MLJBase.pipeline_network_machine(
        Unsupervised, true, predict, components, Xs)
    tree = mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedSurrogate
    @test tree.operation == transform
    @test tree.model == h
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1

    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        Unsupervised, true, predict, components, Xs)
    tree = mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedSurrogate
    @test tree.operation == t
    @test tree.model == nothing
    @test tree.arg1.operation == m
    @test tree.arg1.model == nothing
    @test tree.arg1.arg1.source == Xs

    # check a probablistic case:
    components = [f, c]
    mach = MLJBase.pipeline_network_machine(
        Probabilistic, true, predict, components, Xs, ys)
    @test mach.model isa ProbabilisticSurrogate

    # check a static case:
    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        Static, true, predict, components, Xs, ys)
    @test mach.model isa StaticSurrogate

    # An integration test...

    # build a linear network for training:
    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        Deterministic, true, predict, components, Xs, ys)

    # build the same network by hand:
    fM = machine(f, Xs)
    Xt = transform(fM, Xs)
    uM = machine(u, ys)
    yt = transform(uM, ys)
    kM = machine(k, Xt, yt)
    zhat = predict(kM, Xt)
    N2 = inverse_transform(uM, zhat)

    # compare predictions
    fit!(mach, verbosity=0);
    fit!(N2, verbosity=0)
    yhat = predict(mach, X);
    @test yhat ≈ N2()
    k.K = 3; f.features = [:x3,]
    fit!(mach, verbosity=0);
    fit!(N2, verbosity=0)
    @test !(yhat ≈ predict(mach, X))
    @test predict(mach, X) ≈ N2()
    global hand_built = predict(mach, X);

end


# # INTEGRATION TESTS

@testset "integration 1" begin
    # check a simple pipeline prediction agrees with prediction of
    # hand-built learning network built earlier:
    p = Pipeline(FeatureSelector,
                 KNNRegressor,
                 prediction_type=:deterministic)
    p.knn_regressor.K = 3; p.feature_selector.features = [:x3,]
    mach = machine(p, X, y)
    fit!(mach, verbosity=0)
    @test MLJBase.tree(mach.fitresult.predict).model.K == 3
    MLJBase.tree(mach.fitresult.predict).arg1.model.features == [:x3, ]
    @test predict(mach, X) ≈ hand_built

    # test cache is set correctly internally:
    @test all(fitted_params(mach).machines) do m
        MLJBase._cache_status(m)  == " caches data"
    end

    # test correct error thrown for inverse_transform:
    @test_throws(MLJBase.ERR_INVERSION_NOT_SUPPORTED,
                 inverse_transform(mach, 3))
end

@testset "integration 2" begin
    # a simple probabilistic classifier pipeline:
    X = MLJBase.table(rand(rng,7,3));
    y = categorical(collect("ffmmfmf"));
    Xs = source(X)
    ys = source(y)
    p = Pipeline(OneHotEncoder, ConstantClassifier, cache=false)
    mach = machine(p, X, y)
    fit!(mach, verbosity=0)
    @test p isa ProbabilisticComposite
    pdf(predict(mach, X)[1], 'f') ≈ 4/7

    # test cache is set correctly internally:
    @test all(fitted_params(mach).machines) do m
        MLJBase._cache_status(m) == " does not cache data"
    end

    # test invalid replacement of classifier with regressor throws
    # informative error message:
    p.constant_classifier = ConstantRegressor()
    @test_logs((:error, r"^Problem"),
           (:info, r"^Running type"),
           (:warn, r"The scitype of"),
           (:info, r"It seems"),
           (:error, r"Problem"),
               @test_throws Exception fit!(mach, verbosity=-1))
end

@testset "integration 3" begin
    # test a simple deterministic classifier pipeline:
    X = MLJBase.table(rand(rng,7,3))
    y = categorical(collect("ffmmfmf"))
    Xs = source(X)
    ys = source(y)
    p = Pipeline(OneHotEncoder, ConstantClassifier, broadcast_mode,
                 prediction_type=:probabilistic)
    mach = machine(p, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill('f', 7)

    # test pipelines with weights:
    w = map(y) do η
        η == 'm' ? 100 : 1
    end
    mach = machine(p, X, y, w)
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill('m', 7)
end

age = [23, 45, 34, 25, 67]
X = (age = age,
     gender = categorical(['m', 'm', 'f', 'm', 'f']))
height = [67.0, 81.5, 55.6, 90.0, 61.1]

mutable struct MyTransformer3 <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer3, verbosity, X) =
     selectcols(X, transf.ftr)

@testset "integration 4" begin
    #static transformers in pipelines
    p99 = Pipeline(X -> coerce(X, :age=>Continuous),
                   OneHotEncoder,
                   MyTransformer3(:age))
    mach = fit!(machine(p99, X), verbosity=0)
    @test transform(mach, X) == float.(X.age)
end

@testset "integration 5" begin
    # pure static pipeline:
    p = Pipeline(X -> coerce(X, :age=>Continuous),
                 MyTransformer3(:age))

    mach = fit!(machine(p), verbosity=0) # no training arguments!
    @test transform(mach, X) == X.age

    # and another:
    p = Pipeline(exp, log, x-> 2*x)
    mach = fit!(machine(p), verbosity=0)
    @test transform(mach, 20) ≈ 40
end

@testset "integration 6" begin
    # operation different from predict:
    p = Pipeline(OneHotEncoder,
                 ConstantRegressor,
                 operation=predict_mean)
    @test p isa Deterministic
    mach = fit!(machine(p, X, height), verbosity=0)
    @test scitype(predict(mach, X)) == AbstractVector{Continuous}
end

@testset "integration 7" begin
    # inverse transform:
    p = Pipeline(UnivariateBoxCoxTransformer,
                 UnivariateStandardizer)
    xtrain = rand(rng, 10)
    mach = machine(p, xtrain)
    fit!(mach, verbosity=0)
    x = rand(rng, 5)
    y = transform(mach, x)
    x̂ = inverse_transform(mach, y)
    @test isapprox(x, x̂)
end

# A dummy clustering model:
mutable struct DummyClusterer <: Unsupervised
    n::Int
end
DummyClusterer(; n=3) = DummyClusterer(n)
function MLJBase.fit(model::DummyClusterer, verbosity::Int, X)
    Xmatrix = Tables.matrix(X)
    n = min(size(Xmatrix, 2), model.n)
    centres = Xmatrix[1:n, :]
    levels = categorical(1:n)
    report = (centres=centres,)
    fitresult = levels
    return fitresult, nothing, report
end
MLJBase.transform(model::DummyClusterer, fitresult, Xnew) =
    selectcols(Xnew, 1:length(fitresult))
MLJBase.predict(model::DummyClusterer, fitresult, Xnew) =
    [fill(fitresult[1], nrows(Xnew))...]

@testset "integration 8" begin
    # calling predict on unsupervised pipeline
    # https://github.com/JuliaAI/MLJClusteringInterface.jl/issues/10

    N = 20
    X = (a = rand(N), b = rand(N))

    p = Pipeline(PCA, DummyClusterer)
    mach = machine(p, X)
    fit!(mach, verbosity=0)
    y = predict(mach, X)
    @test y == fill(categorical(1:2)[1], N)
end

@testset "syntactic sugar" begin

    # recall u, s, p, m, are defined way above

    # unsupervised model |> static model:
    pipe1 = u |> s
    @test pipe1 == Pipeline(u, s)

    # unsupervised model |> supervised model:
    pipe2 = u |> p
    @test pipe2 == Pipeline(u, p)

    # pipe |> pipe:
    hose = pipe1 |> pipe2
    @test hose == Pipeline(u, s, u, p)

    # pipe |> model:
    @test Pipeline(u, s) |> p == Pipeline(u, s, p)

    # model |> pipe:
    @test u |> Pipeline(s, p) == Pipeline(u, s, p)

    # pipe |> function:
    @test Pipeline(u, s) |> m == Pipeline(u, s, m)

    # function |> pipe:
    @test m |> Pipeline(s, p) == Pipeline(m, s, p)

    # model |> function:
    @test u |> m == Pipeline(u, m)

    # function |> model:
    @test t |> u == Pipeline(t, u)

    @test_logs((:info, MLJBase.INFO_AMBIGUOUS_CACHE),
               Pipeline(u, cache=false) |> p)

    # with types
    @test PCA |> Standardizer() |> KNNRegressor ==
        Pipeline(PCA(), Standardizer(), KNNRegressor())
end

@testset "miscelleneous coverage" begin
    @test MLJBase.as_type(:unsupervised) == Unsupervised
end 

end

true
