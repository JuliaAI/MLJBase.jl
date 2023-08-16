using Test
using MLJBase
include(joinpath("..", "..", "test", "_models", "models.jl"))
using .Models

@testset "_categorical" begin
    a = [1, 1, 2, 3]
    b = [3, 3, 4, 5]
    c = [missing, a...]
    d = [missing, b...]
    e = categorical(a)
    f = categorical(b)
    g = categorical(c)
    h = categorical(d)
    j = CategoricalArrays.CategoricalValue{Int64, UInt32}[e[1], e[1], e[1], e[1]]
    k = CategoricalArrays.CategoricalValue{Int64, UInt32}[e[4], e[4], e[4], e[4]]
    rhs = (Set(1:5), Set(1:5))
    @test Set.(levels.(MLJBase._categorical(a, b))) == rhs
    @test Set.(levels.(MLJBase._categorical(a, d))) == rhs
    @test Set.(levels.(MLJBase._categorical(c, b))) == rhs
    @test Set.(levels.(MLJBase._categorical(c, d))) == rhs
    @test Set.(levels.(MLJBase._categorical(a, f))) == rhs
    @test Set.(levels.(MLJBase._categorical(a, h))) == rhs
    @test Set.(levels.(MLJBase._categorical(b, a))) == rhs
    @test Set.(levels.(MLJBase._categorical(d, a))) == rhs
    @test Set.(levels.(MLJBase._categorical(b, c))) == rhs
    @test Set.(levels.(MLJBase._categorical(d, c))) == rhs
    @test Set.(levels.(MLJBase._categorical(f, a))) == rhs
    @test Set.(levels.(MLJBase._categorical(h, a))) == rhs

    @test Set.(levels.(MLJBase._categorical(j, k))) == (Set(1:3), Set(1:3))

    # case of ordinary vector with CategoricalValue eltype:
    acv = CategoricalArrays.CategoricalVector
end

@testset "basics" begin
    yraw = ['m',     'm', 'f', 'n', missing, 'f', 'm', 'n', 'n', 'm', 'f']
    ŷraw = [missing, 'f', 'f', 'm', 'f',     'f', 'n', 'm', 'n', 'm', 'f']
    y = categorical(yraw)
    ŷ = categorical(ŷraw)
    l = levels(y) # f, m, n
    cm = MLJBase._confmat(ŷ, y; warn=false)
    ŷ_clean, y_clean = MLJBase.skipinvalid(ŷ, y)
    ee(l,i,j) = sum((ŷ_clean .== l[i]) .& (y_clean .== l[j]))
    for i in 1:3, j in 1:3
        @test cm[i,j] == ee(l,i,j)
    end

    cm2 = @test_logs (:warn, r"The classes are") MLJBase._confmat(ŷraw, yraw)
    @test cm2.mat == cm.mat

    perm = [3, 1, 2]
    l2 = l[perm]
    cm2 = @test_logs MLJBase._confmat(ŷ, y; perm=perm)
    m = ConfusionMatrix(perm=perm)
    for i in 1:3, j in 1:3
        @test cm2[i,j] == ee(l2,i,j)
    end
    @test_logs (:warn, r"The classes are un") MLJBase._confmat(ŷ, y)
    ŷc = coerce(ŷ, Union{Missing,OrderedFactor})
    yc = coerce(y, Union{Missing,OrderedFactor})
    @test MLJBase._confmat(ŷc, yc).mat == cm.mat

    y = categorical(['a','b','a','b'])
    ŷ = categorical(['b','b','a','a'])
    @test_logs (:warn, r"The classes are un") MLJBase._confmat(ŷ, y)

    # more tests for coverage
    y = categorical([1,2,3,1,2,3,1,2,3])
    ŷ = categorical([1,2,3,1,2,3,1,2,3])
    @test_throws ArgumentError MLJBase._confmat(ŷ, y, rev=true)

    # silly test for display
    ŷ = coerce(y, OrderedFactor)
    y = coerce(y, OrderedFactor)
    iob = IOBuffer()
    Base.show(iob, MIME("text/plain"), MLJBase._confmat(ŷ, y))
    siob = String(take!(iob))
    @test strip(siob) == strip("""
              ┌──────────────┐
              │ Ground Truth │
    ┌─────────┼────┬────┬────┤
    │Predicted│ 1  │ 2  │ 3  │
    ├─────────┼────┼────┼────┤
    │    1    │ 3  │ 0  │ 0  │
    ├─────────┼────┼────┼────┤
    │    2    │ 0  │ 3  │ 0  │
    ├─────────┼────┼────┼────┤
    │    3    │ 0  │ 0  │ 3  │
    └─────────┴────┴────┴────┘""")
end

@testset "ConfusionMatrix measure" begin

    @test info(confmat).orientation == :other
    model = DeterministicConstantClassifier()

    X = (x=rand(10),)
    long = categorical(collect("abbaacaabbbbababcbac"), ordered=true)
    y = long[1:10]
    yhat =long[11:20]

    @test confmat(yhat, y).mat == [1 2 0; 3 1 1; 1 1 0]
    @test ConfusionMatrix(perm=[2, 1, 3])(yhat, y).mat ==
        MLJBase._confmat(yhat, y, perm=[2, 1, 3]).mat

    MLJBase.value(confmat, yhat, X, y, nothing)

    e = evaluate(model, X, y,
                 measures=[misclassification_rate, confmat],
                 resampling=Holdout(fraction_train=0.5))
    cm = e.measurement[2]
    @test cm.labels == ["a", "b", "c"]
    @test cm.mat == [2 2 1; 0 0 0; 0 0 0]
end
