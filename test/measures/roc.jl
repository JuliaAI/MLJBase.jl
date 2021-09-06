@testset "ROC" begin
    y = [  0   0   0   1   0   1   1   0] |> vec |> categorical
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5] |> vec
    ŷ = UnivariateFinite([0, 1], s, augment=true, pool=y)

    fprs, tprs, ts = roc(ŷ, y)

    sk_fprs = [0. , 0.2, 0.4, 0.8, 1. ]
    sk_tprs = [0. , 0.33333333, 0.66666667, 1., 1.]

    @test fprs ≈ sk_fprs
    @test tprs ≈ sk_tprs
end
