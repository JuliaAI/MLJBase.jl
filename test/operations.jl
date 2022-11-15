module TestOperations

using Test
using MLJBase
using Serialization
using ..Models

@testset "Operations on non-composite models" begin
    # Unsupervised model
    X = rand(4)
    m = fit!(machine(UnivariateStandardizer(), X), verbosity=0)
    @test_throws ArgumentError inverse_transform(m)
    @test inverse_transform(m, transform(m)) ≈ X
    @test inverse_transform(m, transform(m, X)) ≈ X
    X = source(rand(4))
    m = fit!(machine(UnivariateStandardizer(), X), verbosity=0)
    @test_throws ArgumentError inverse_transform(m)
    @test inverse_transform(m, transform(m)) ≈ X() # test with node args
    @test inverse_transform(m, transform(m, X))() ≈ X()

    # Supervised model
    X = MLJBase.table(rand(4, 4))
    y = rand(4)
    m2 = fit!(machine(DeterministicConstantRegressor(), X, y; cache=false), verbosity=0)
    @test predict(m2) == fill(mean(y), length(y))

    # Check that an error is thrown when applying an operation to a serialized machine
    # with no args
    filename = "constant_regressor_machine"
    filename2 = "univariate_standadizer_machine"
    smach = serializable(m)
    smach2 = serializable(m2)
    Serialization.serialize(filename, smach)
    Serialization.serialize(filename2, smach2)
    smach = Serialization.deserialize(filename)
    smach2 = Serialization.deserialize(filename2)
    MLJBase.restore!(smach)
    MLJBase.restore!(smach2)
    @test_throws ArgumentError predict(smach)
    @test_throws ArgumentError predict(smach2)
    rm(filename)
    rm(filename2)

    # Static model
    y1, y2 = rand(4), rand(4)
    m = fit!(machine(Averager(mix = 0.5)), verbosity=0)
    m2 = fit!(machine(Averager(mix = 0.5); cache=false), verbosity=0) # non-cached version
    @test_throws ArgumentError transform(m)
    @test_throws ArgumentError transform(m2)
    @test_throws ArgumentError inverse_transform(m)
    average = 0.5 .* y1 .+ 0.5 .* y2
    @test transform(m, y1, y2) == average #(1 - 0.5) .* y1 .+ 0.5 .* y2
    @test transform(m, source(y1), source(y2))() == average
    # Check that error is thrown when at least one of the inputs to `transform` is wrong.
    # These tests are specific to the `Averager` static transformer
    @test_throws ArgumentError transform(m, y1, Tuple(y2))
    @test_throws ArgumentError transform(m, Tuple(y1), Tuple(y2))
end

@testset "operations on network-composite models" begin
    X = MLJBase.table(rand(4, 4))
    y = rand(4)
    m = fit!(machine(SimpleProbabilisticNetworkCompositeModel(), X, y), verbosity=0)
    predictions = first(MLJBase.output_and_report(m.fitresult, :predict, X))
    @test predict(m, X) == predictions
    @test predict_mode(m, X) == mode.(predictions)
    @test_throws ErrorException transform(m, X)
end

# Test below to be removed after next breaking release
@testset "operations on composite/surrogate models" begin
    X = MLJBase.table(rand(4, 4))
    y = rand(4)
    m = fit!(machine(SimpleDeterministicCompositeModel(), X, y), verbosity=0)
    @test predict(m, X) == m.fitresult.predict(X)
    @test_throws ErrorException transform(m, X)

    m = fit!(machine(SimpleProbabilisticCompositeModel(), X, y), verbosity=0)
    predictions = m.fitresult.predict(X)
    @test predict(m, X) == predictions
    @test predict_mode(m, X) == mode.(predictions)
    @test_throws ErrorException transform(m, X)
end

end

true
