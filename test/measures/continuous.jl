rng = StableRNG(666899)

@testset "regressor measures" begin
    y    = [1, 42,  2, 3, missing, 4]
    yhat = [4, NaN, 3, 2, 42,      1]
    w =    [1, 42,  2, 4, 42,      3]
    y    = [1,  2, 3, 4]
    yhat = [4, 3, 2,      1]
    w =    [1,  2, 4,      3]
    @test isapprox(mae(yhat, y), 2)
    @test isapprox(mae(yhat, y, w), (1*3 + 2*1 + 4*1 + 3*3)/4)
    @test isapprox(rms(yhat, y), sqrt(5))
    @test isapprox(rms(yhat, y, w), sqrt((1*3^2 + 2*1^2 + 4*1^2 + 3*3^2)/4))
    @test rsq(yhat, y) == -3
    @test isapprox(mean(skipinvalid(l1(yhat, y))), 2)
    @test isapprox(mean(skipinvalid(l1(yhat, y, w))), mae(yhat, y, w))
    @test isapprox(mean(skipinvalid(l2(yhat, y))), 5)
    @test isapprox(mean(skipinvalid(l2(yhat, y, w))), rms(yhat, y, w)^2)
    @test isapprox(mean(skipinvalid(log_cosh(yhat, y))), 1.3715546675)

    y    = [1, 42,  2, 3, missing, 4]
    yhat = [2, NaN, 3, 4, 42,      5]
    @test isapprox(rmsl(yhat, y),
                   sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
    @test isapprox(rmslp1(yhat, y),
                   sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
    @test isapprox(rmsp(yhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))
    @test isapprox(mape(yhat, y), (1/1 + 1/2 + 1/3 + 1/4)/4)
end

true
