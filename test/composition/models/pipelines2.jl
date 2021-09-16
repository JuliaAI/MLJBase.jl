module TestPipelines2

using MLJBase
using Test
using ..Models
using StableRNGs

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
s = MyTransformer2(:s)
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

end

@testset "property access" begin
    pipe = Pipeline(m, u, u, s)

    # property names:
    @test propertynames(pipe) ===
        (:f, :my_unsupervised, :my_unsupervised2, :my_transformer2, :cache)

    # getindex:
    pipe.my_unsupervised == u
    pipe.my_unsupervised2 == u
    pipe.cache = true

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
    front = MLJBase.extend(front, u, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, s, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:st, :st, :st]

    front = MLJBase.extend(front, u, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:up, :up, :up]
    @test tnode() == [:ut, :ut, :ut]

    front = MLJBase.extend(front, d, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:dp, :dp, :dp]
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, x->string.(x), predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == string.([:dp, :dp, :dp])
    @test tnode() == [:dt, :dt, :dt]

    front = MLJBase.extend(front, u, predict, ys)
    pnode, tnode = front.predict, front.transform
    fit!(pnode, verbosity=0)
    fit!(tnode, verbosity=0)
    @test pnode() == [:ut, :ut, :ut]
    @test tnode() == [:dt, :dt, :dt]
end

NN = 7
X = MLJBase.table(rand(rng,NN, 3));
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

    components = [f, h]
    mach = MLJBase.pipeline_network_machine(
        Unsupervised, predict, components, Xs)
    tree = mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedSurrogate
    @test tree.operation == transform
    @test tree.model == h
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1

    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        Deterministic, predict, components, Xs, ys)
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

    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        Unsupervised, predict, components, Xs)
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
        Probabilistic, predict, components, Xs, ys)
    @test mach.model isa ProbabilisticSurrogate

    # check a static case:
    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        Static, predict, components, Xs, ys)
    @test mach.model isa StaticSurrogate

    # An integration test...

    # build a linear network for training:
    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        Deterministic, predict, components, Xs, ys)

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

end

true
