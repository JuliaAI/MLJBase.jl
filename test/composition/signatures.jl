@testset "signature helpers" begin
    @test MLJBase._call(NamedTuple()) == NamedTuple()
    a = source(:a)
    b = source(:b)
    W = source(:W)
    yhat = source(:yhat)
    s = (transform=W,
         report=(a=a, b=b),
         predict=yhat)
    @test MLJBase._report_part(s) == (a=a, b=b)
    @test MLJBase._operation_part(s) == (transform=W, predict=yhat)
    @test MLJBase._nodes(s) == (W, yhat, a, b)
    @test MLJBase._operations(s) == (:transform, :predict)
end

true
