using Test
using MLJBase
include(joinpath("..", "..", "test", "_models", "models.jl"))
using .Models

@testset "basics" begin
    y = categorical(['m', 'f', 'n', 'f', 'm', 'n', 'n', 'm', 'f'])
    ŷ = categorical(['f', 'f', 'm', 'f', 'n', 'm', 'n', 'm', 'f'])
    l = levels(y) # f, m, n
    cm = confmat(ŷ, y; warn=false)
    e(l,i,j) = sum((ŷ .== l[i]) .& (y .== l[j]))
    for i in 1:3, j in 1:3
        @test cm[i,j] == e(l,i,j)
    end
    perm = [3, 1, 2]
    l2 = l[perm]
    cm2 = confmat(ŷ, y; perm=perm) # no warning because permutation is given
    for i in 1:3, j in 1:3
        @test cm2[i,j] == e(l2,i,j)
    end
    @test_logs (:warn, "The classes are un-ordered,\nusing order: ['f', 'm', 'n'].\nTo suppress this warning, consider coercing to OrderedFactor.") confmat(ŷ, y)
    ŷc = coerce(ŷ, OrderedFactor)
    yc = coerce(y, OrderedFactor)
    @test confmat(ŷc, yc).mat == cm.mat

    y = categorical(['a','b','a','b'])
    ŷ = categorical(['b','b','a','a'])
    @test_logs (:warn, "The classes are un-ordered,\nusing: negative='a' and positive='b'.\nTo suppress this warning, consider coercing to OrderedFactor.") confmat(ŷ, y)

    # more tests for coverage
    y = categorical([1,2,3,1,2,3,1,2,3])
    ŷ = categorical([1,2,3,1,2,3,1,2,3])
    @test_throws ArgumentError confmat(ŷ, y, rev=true)

    # silly test for display
    ŷ = coerce(y, OrderedFactor)
    y = coerce(y, OrderedFactor)
    iob = IOBuffer()
    Base.show(iob, MIME("text/plain"), confmat(ŷ, y))
    siob = String(take!(iob))
    @test strip(siob) == strip("""
                         ┌─────────────────────────────────────────┐
                         │              Ground Truth               │
           ┌─────────────┼─────────────┬─────────────┬─────────────┤
           │  Predicted  │      1      │      2      │      3      │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
           │      1      │      3      │      0      │      0      │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
           │      2      │      0      │      3      │      0      │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
           │      3      │      0      │      0      │      3      │
           └─────────────┴─────────────┴─────────────┴─────────────┘""")

end

@testset "confmat as measure" begin

    @test info(confmat).orientation == :other
    model = DeterministicConstantClassifier()

    X = (x=rand(10),)
    long = categorical(collect("abbaacaabbbbababcbac"), ordered=true)
    y = long[1:10]
    yhat =long[11:20]

    confmat(yhat, y).mat == [1 2 0; 3 1 1; 1 1 0]

    MLJBase.value(confmat, yhat, X, y, nothing)

    # e = evaluate(model, X, y,
    #              measures=[misclassification_rate, confmat],
    #              resampling=Holdout(fraction_train=0.5))
    # cm = e.measurement[2]
    # @test cm.labels == ["a", "b", "c"]
    # @test cm.mat == [2 2 1; 0 0 0; 0 0 0]
end
