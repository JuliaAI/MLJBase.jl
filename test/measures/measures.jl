@testset "aggregation" begin
    v = rand(5)
    aggregate(v, mav) ≈ mean(v)
    aggregate(v, TruePositive()) ≈ sum(v)
    aggregate(v, rms) ≈ sqrt(mean(v.^2))
end
true
