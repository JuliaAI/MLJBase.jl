@testset "aggregation" begin
    v = rand(5)
    aggregate(v, mav) ≈ mean(v)
    aggregate(v, TruePositive()) ≈ sum(v)
    aggregate(v, rms) ≈ sqrt(mean(v.^2))
end

@testset "metadata" begin
    measures()
    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
                  m.supports_weights)
end
true
