module TestPipelines2

using MLJBase
using Test

@testset "helpers" begin
    @test MLJBase.individuate([:x, :y, :x, :z, :y, :x]) ==
        [:x, :y, :x2, :z, :y2, :x3]
end


# # DUMMY MODELS

mutable struct MyTransformer2 <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer2, verbosity, X) =
    selectcols(X, transf.ftr)

mutable struct MyDeterministic <: Deterministic
    x::Symbol
end

mutable struct MyProbabilistic <: Probabilistic
    x::Symbol
end

mutable struct MyUnsupervised <: Unsupervised
    x::Symbol
end

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

end

true
