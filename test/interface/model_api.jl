module TestModelAPI

using Test
using MLJBase
import MLJModelInterface
using ..Models

@testset "predict_*" begin
    X = (x = rand(5),)
    yfinite = categorical(collect("abaaa"))
    ycont = float.(1:5)

    clf = ConstantClassifier()
    fitresult, _, _ = MLJBase.fit(clf, 1, X, yfinite)
    @test predict_mode(clf, fitresult, X)[1] == 'a'
    @test_throws ArgumentError predict_mean(clf, fitresult, X)
    @test_throws ArgumentError predict_median(clf, fitresult, X)

    rgs = ConstantRegressor()
    fitresult, _, _ = MLJBase.fit(rgs, 1, X, ycont)
    @test predict_mean(rgs, fitresult, X)[1] == 3
    @test predict_median(rgs, fitresult, X)[1] == 3
    @test_throws ArgumentError predict_mode(rgs, fitresult, X)
end

@testset "serialization" begin

    # train a model on some data:
    model = @load KNNRegressor
    X = (a = Float64[98, 53, 93, 67, 90, 68],
         b = Float64[64, 43, 66, 47, 16, 66],)
    Xnew = (a = Float64[82, 49, 16],
            b = Float64[36, 13, 36],)
    y =  [59.1, 28.6, 96.6, 83.3, 59.1, 48.0]
    fitresult, cache, report = MLJBase.fit(model, 0, X, y)
    pred = predict(model, fitresult, Xnew)
    filename = joinpath(@__DIR__, "test.jlso")

    # save to file:
    # To avoid complications to travis tests (ie, writing to file) the
    # next line was run once and then commented out:
    # save(filename, model, fitresult, report)

    # save to buffer:
    io = IOBuffer()
    save(io, model, fitresult, report, compression=:none)
    seekstart(io)

    # test restoring data:
    for input in [filename, io]
        eval(quote
             @show $input
             m, f, r = restore($input)
             p = predict(m, f, $Xnew)
             @test m == $model
             @test r == $report
             @test p ≈ $pred
             end)
    end

end

end

true
NamedTuple{(),Tuple{}}(())
