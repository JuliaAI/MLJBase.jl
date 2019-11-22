@testset "aggregation" begin
    v = rand(5)
    MLJBase.aggregate(v, mav) ≈ mean(v)
    MLJBase.aggregate(v, TruePositive()) ≈ sum(v)
    MLJBase.aggregate(v, rms) ≈ sqrt(mean(v.^2))
end

@testset "metadata" begin
    measures()
    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
                  m.supports_weights)
end

@testset "coverage" begin
    # just checking that the  traits work not that they're correct
    @test orientation(BrierScore()) == :score
    @test orientation(auc) == :score
    @test orientation(rms) == :loss

    @test reports_each_observation(auc) == false
    @test is_feature_dependent(auc) == false

    @test_broken MLJBase.distribution_type(BrierScore{UnivariateFinite}) == UnivariateFinite
end

true
