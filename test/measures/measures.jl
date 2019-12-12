@testset "aggregation" begin
    v = rand(5)
    @test aggregate(v, mav) ≈ mean(v)
    @test aggregate(v, TruePositive()) ≈ sum(v)
    @test aggregate(v, rms) ≈ sqrt(mean(v.^2))
    λ = rand()
    @test aggregate(λ, rms) === λ
    @test aggregate(aggregate(v, l2), l2) == aggregate(v, l2)
end

@testset "metadata" begin
    measures()
    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
             m.supports_weights)
    info(rms)
end

@testset "coverage" begin
    # just checking that the traits work not that they're correct
    @test orientation(BrierScore()) == :score
    @test orientation(auc) == :score
    @test orientation(rms) == :loss

    @test reports_each_observation(auc) == false
    @test is_feature_dependent(auc) == false

    @test MLJBase.distribution_type(BrierScore{UnivariateFinite}) ==
        UnivariateFinite
end

true
