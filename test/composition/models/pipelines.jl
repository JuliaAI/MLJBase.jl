module TestPipelines2

using MLJBase
using Test
using ..Models
using ..TestUtilities
using StableRNGs
using Tables
import MLJBase: Pred, Trans

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

MLJBase.fit(::MyDeterministic, args...) = nothing, nothing, (; tlosses=ones(3))
MLJBase.transform(m::MyDeterministic, ::Any, Xnew) = fill(:dt, nrows(Xnew))
MLJBase.predict(m::MyDeterministic, ::Any, Xnew) = fill(:dp, nrows(Xnew))
MLJBase.supports_training_losses(::Type{<:MyDeterministic}) = true
MLJBase.iteration_parameter(::Type{<:MyDeterministic}) = :x
MLJBase.training_losses(::MyDeterministic, report) = report.tlosses

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

mutable struct StaticKefir <: Static
    alpha::Float64 # non-zero to be invertible
end
MLJBase.reporting_operations(::Type{<:StaticKefir}) = (:transform, :inverse_transform)

# piece-wise linear function that is linear only for `alpha=1`:
kefir(x, alpha) = x > 0 ? x * alpha : x / alpha

MLJBase.transform(model::StaticKefir, _, X) = (
    broadcast(kefir, X, model.alpha),
    (; first = first(X)),
)

MLJBase.inverse_transform(model::StaticKefir, _, W) = (
    broadcast(kefir, W, 1/(model.alpha)),
    (; last = last(W)),
)

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
    flute = Pipeline(m, t, u)
    @test flute isa UnsupervisedPipeline
    @test MLJBase.constructor(flute) == Pipeline
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

    composite = (;
                 u1=u,
                 u2=u,
                 u3=u,
                 s1=s,
                 s2=s,
                 d=d,
                 callable=x->string.(x),
                 )

    Xs = source(rand(3))
    ys = source(rand(3))
    front = MLJBase.Front(Xs, ys, Pred())
    @test front.predict() == Xs()
    @test front.transform() == ys()
    @test MLJBase.active(front)() == Xs()

    front = MLJBase.Front(Xs, Xs, Trans())
    front = MLJBase.extend(front, u, :u1, false, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, s, :s1, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:st, :st, :st]

    front = MLJBase.extend(front, u, :u2, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, d, :d, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:dp, :dp, :dp]
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, x->string.(x), :callable, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == string.([:dp, :dp, :dp])
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, u, :u3, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:ut, :ut, :ut]
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, s, :s2, true, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0, composite=composite)
    fit!(tnode, verbosity=0, composite=composite)
    @test pnode() == [:st, :st, :st]
    @test tnode() == [:dt, :dt, :dt]
end

NN = 7
X = MLJBase.table(rand(rng, NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(rng,NN);
Xs = source(X); ys = source(y)

broadcast_mode(v) = mode.(v)
doubler(y) = 2*y

@testset "pipeline_network_interface" begin
    t = MLJBase.table
    m = MLJBase.matrix
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()

    component_name_pairs = [f => :f, k => :k]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
        ys,
    )
    tree = interface.predict |> MLJBase.tree
    @test tree.operation == predict
    @test tree.model == :k
    @test tree.arg1.operation == transform
    @test tree.arg1.model == :f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1
    @test tree.train_arg2.source == ys

    component_name_pairs = [f => :f, h => :h]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
    )
    tree = interface.transform |> MLJBase.tree
    @test tree.operation == transform
    @test tree.model == :h
    @test tree.arg1.operation == transform
    @test tree.arg1.model == :f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1

    component_name_pairs = [m => :m, t => :t]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
    )
    tree = interface.transform |> MLJBase.tree
    @test tree.operation == t
    @test tree.model == nothing
    @test tree.arg1.operation == m
    @test tree.arg1.model == nothing
    @test tree.arg1.arg1.source == Xs

    # check a probablistic case:
    component_name_pairs = [f => :f, c => :c]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
        ys,
    )

    # check a static case:
    component_name_pairs = [m => :m, t => :t]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
        ys,
    )

    # An integration test...

    # build a linear network and interface:
    component_name_pairs = [f => :f, k => :k]
    interface = MLJBase.pipeline_network_interface(
        true,
        predict,
        component_name_pairs,
        Xs,
        ys,
    )
    yhat1 = interface.predict

    # build the same network by hand:
    fM = machine(:f, Xs)
    Xt = transform(fM, Xs)
    kM = machine(k, Xt, ys)
    yhat2 = predict(kM, Xt)

    # compare predictions
    composite = (; f, k)
    verbosity = 0
    fit!(yhat1; verbosity, composite);
    fit!(yhat2; verbosity, composite);
    @test yhat1() ≈ yhat2()
    k.K = 3; f.features = [:x3,]
    fit!(yhat1; verbosity, composite);
    fit!(yhat2; verbosity, composite);

    @test yhat1() ≈ yhat2()
    global hand_built = yhat1()

end

struct FooCarrot <: Deterministic end

@testset "iteration parameter - nothing passes through" begin
    pipe = FeatureSelector() |> FooCarrot()
    @test iteration_parameter(pipe) === nothing
end

@testset "training_losses" begin
    model = MyDeterministic(:bla)
    pipe = Standardizer() |> model

    # test helpers:
    @test MLJBase.supervised_component_name(pipe) == :my_deterministic
    @test MLJBase.supervised_component(pipe) == model

    @test supports_training_losses(pipe)
    _, _, rp = MLJBase.fit(pipe, 0, X, y)
    @test training_losses(pipe, rp) == ones(3)
    @test iteration_parameter(pipe) ==
        :(my_deterministic.x)
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
    @test MLJBase.tree(MLJBase.unwrap(mach.fitresult).predict).model == :knn_regressor
    @test MLJBase.tree(MLJBase.unwrap(mach.fitresult).predict).arg1.model ==
        :feature_selector
    @test predict(mach, X) ≈ hand_built

    # Check target_scitype of a supervised pipeline is the same as the supervised component
    @test target_scitype(p) == target_scitype(KNNRegressor())

    # test cache is set correctly internally:
    machs = machines(glb(mach.fitresult))
    @test all(machs) do m
        MLJBase._cache_status(m)  == "caches model-specific representations of data"
    end

    # test correct error thrown for inverse_transform:
    @test_throws(MLJBase.err_unsupported_operation(:inverse_transform),
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
    @test p isa ProbabilisticNetworkComposite
    pdf(predict(mach, X)[1], 'f') ≈ 4/7

    # test cache is set correctly internally:
    machs = machines(glb(mach.fitresult))
    @test all(machs) do m
        MLJBase._cache_status(m) == "does not cache data"
    end

    # test invalid replacement of classifier with regressor throws
    # informative error message:
    p.constant_classifier = ConstantRegressor()
    @test_logs((:error, r"^Problem"),
           (:info, r"^Running type"),
               (:warn, MLJBase.alert_generic_scitype_mismatch(
                   scitype((X, y)),
                   MLJBase.fit_data_scitype(ConstantRegressor()),
                   typeof(ConstantRegressor())
               )),
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

    # Check target_scitype of a supervised pipeline is the same as the supervised component
    @test target_scitype(p) == target_scitype(ConstantClassifier())

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

    # Check target_scitype of an unsupervised pipeline is Unknown
    @test target_scitype(p99) == Unknown
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
    @test !target_in_fit(p)
    xtrain = rand(rng, 10)
    mach = machine(p, xtrain)
    fit!(mach, verbosity=0)
    x = rand(rng, 5)
    y = transform(mach, x)
    x̂ = inverse_transform(mach, y)
    @test isapprox(x, x̂)
    # Check target_scitype of an unsupervised pipeline is Unknown
    @test target_scitype(p) == Unknown
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

@testset "inverse transform for pipes with static components" begin
    X = randn(rng, 20)
    pipe = StaticKefir(3) |> UnivariateStandardizer() |>
        StaticKefir(5) |> UnivariateStandardizer()

    mach = machine(pipe, X)
    fit!(mach, verbosity=0)
    @test inverse_transform(mach, transform(mach, X)) ≈ X
    @test transform(mach, inverse_transform(mach, X)) ≈ X
end

@testset "accessing reports generated by Static models" begin
    X = Float64[4, 5, 6]
    pipe = UnivariateStandardizer() |> StaticKefir(3)
    mach = machine(pipe, X)
    fit!(mach, verbosity=0)
    @test isnothing(report(mach)) # tranform has not been called yet
    transform(mach, X) # adds to report of mach, ie mutates mach
    r = report(mach).static_kefir
    @test report(mach).static_kefir.first == -1
    transform(mach, [5, 6]) # mutates `mach`
    r = report(mach).static_kefir
    @test keys(r) == (:first, )
    @test r.first == 0
    inverse_transform(mach, [1, 2, 3])
    r = report(mach)
    @test r.inverse_transform.static_kefir.first == 0.0
    @test r.inverse_transform.static_kefir.last == 3
end

@testset "Test serializable of pipeline" begin
    filename = "pipe_mach.jls"
    X, y = make_regression(100, 1)
    pipe = Standardizer() |> KNNRegressor()
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)

    # Check serializable function
    smach = MLJBase.serializable(mach)
    TestUtilities.generic_tests(mach, smach)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))
    # Check data has been wiped out from models at the first level of composition
    submachines = machines(glb(mach.fitresult))
    ssubmachines = machines(glb(mach.fitresult))
    @test length(submachines) == length(ssubmachines)
    for submach in submachines
        TestUtilities.test_data(submach)
    end

    # End to end
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

@testset "feature importances" begin
    # the DecisionTreeClassifier in /test/_models/ supports feature importances.
    pipe = Standardizer |> DecisionTreeClassifier()
    @test reports_feature_importances(pipe)
    X, y = @load_iris
    fitresult, _, report = MLJBase.fit(pipe, 0, X, y)
    features = first.(feature_importances(pipe, fitresult, report))
    @test Set(features) == Set(keys(X))
end

struct SupervisedTransformer <: Unsupervised end

MLJBase.fit(::SupervisedTransformer, verbosity, X, y) = (mean(y), nothing, nothing)
MLJBase.transform(::SupervisedTransformer, fitresult, X) =
   fitresult*MLJBase.matrix(X) |> MLJBase.table
MLJBase.target_in_fit(::Type{<:SupervisedTransformer}) = true

struct DummyTransformer <: Unsupervised end
MLJBase.fit(::DummyTransformer, verbosity, X) = (nothing, nothing, nothing)
MLJBase.transform(::DummyTransformer, fitresult, X) = X

@testset "supervised transformers in a pipeline" begin
    X = MLJBase.table((a=fill(10.0, 3),))
    y = fill(2, 3)
    pipe = SupervisedTransformer() |> DeterministicConstantRegressor()
    @test target_in_fit(pipe)
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill(2.0, 3)

    pipe2 = DummyTransformer |> pipe
    @test target_in_fit(pipe2)
    mach = machine(pipe2, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill(2.0, 3)

    pipe3 = DummyTransformer |> SupervisedTransformer |> DummyTransformer
    @test target_in_fit(pipe3)
    mach = machine(pipe3, X, y)
    fit!(mach, verbosity=0)
    @test transform(mach, X).x1 == fill(20.0, 3)
end


end # module

true
