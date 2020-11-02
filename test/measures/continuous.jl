rng = StableRNG(666899)

@testset "built-in regressor measures" begin
    y    = [1, 2, 3, 4]
    yhat = [4, 3, 2, 1]
    w = [1, 2, 4, 3]
    @test isapprox(mae(yhat, y), 2)
    @test isapprox(mae(yhat, y, w), (1*3 + 2*1 + 4*1 + 3*3)/4)
    @test isapprox(rms(yhat, y), sqrt(5))
    @test isapprox(rms(yhat, y, w), sqrt((1*3^2 + 2*1^2 + 4*1^2 + 3*3^2)/4))
    @test isapprox(mean(l1(yhat, y)), 2)
    @test isapprox(mean(l1(yhat, y, w)), mae(yhat, y, w))
    @test isapprox(mean(l2(yhat, y)), 5)
    @test isapprox(mean(l2(yhat, y, w)), rms(yhat, y, w)^2)
    @test isapprox(mean(log_cosh(yhat, y)), 1.3715546675)
    @test isapprox(mean(MLJBase.softplus(yhat-y)), 1.1809245195)

    yhat = y .+ 1
    @test isapprox(rmsl(yhat, y),
                   sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
    @test isapprox(rmslp1(yhat, y),
                   sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
    @test isapprox(rmsp(yhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))
    @test isapprox(mape(yhat, y), (1/1 + 1/2 + 1/3 + 1/4)/4)
end


@testset "MLJBase.value" begin
    yhat = randn(rng,5)
    X = (weight=randn(rng,5), x1 = randn(rng,5))
    y = randn(rng,5)
    w = randn(rng,5)

    @test MLJBase.value(mae, yhat, nothing, y, nothing) ≈ mae(yhat, y)
    @test MLJBase.value(mae, yhat, nothing, y, w) ≈ mae(yhat, y, w)

    spooky(yhat, y) = abs.(yhat - y) |> mean
    @test MLJBase.value(spooky, yhat, nothing, y, nothing) ≈ mae(yhat, y)

    cool(yhat, y, w) = abs.(yhat - y) .* w |> mean
    MLJBase.supports_weights(::Type{typeof(cool)}) = true
    @test MLJBase.value(cool, yhat, nothing, y, w) ≈ mae(yhat, y, w)

    funky(yhat, X, y) = X.weight .* abs.(yhat - y) |> mean
    MLJBase.is_feature_dependent(::Type{typeof(funky)}) = true
    @test MLJBase.value(funky, yhat, X, y, nothing) ≈ mae(yhat, y, X.weight)

    weird(yhat, X, y, w) = w .* X.weight .* abs.(yhat - y) |> mean
    MLJBase.is_feature_dependent(::Type{typeof(weird)}) = true
    MLJBase.supports_weights(::Type{typeof(weird)}) = true
    @test MLJBase.value(weird, yhat, X, y, w) ≈ mae(yhat, y, X.weight .* w)
end
