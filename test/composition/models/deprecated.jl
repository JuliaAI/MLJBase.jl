module TestDeprecatedComposition

using Test
using ..Models
using ..TestUtilities
using MLJBase
using Tables
using StableRNGs
using Serialization
rng = StableRNG(616161)


# # IMPLEMENTATION OF MLJ MODEL INTERFACE FOR  <:Composite MODELS

@testset "Test serializable of pipeline" begin
    filename = "pipe_mach.jls"
    X, y = make_regression(100, 1)
    pipe = (X -> coerce(X, :x₁=>Continuous)) |> DecisionTreeRegressor()
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)

    smach = MLJBase.serializable(mach)

    TestUtilities.generic_tests(mach, smach)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))
    # Check data has been wiped out from models at the first level of composition
    @test length(machines(glb(smach))) == length(machines(glb(mach)))
    for submach in machines(glb(smach))
        TestUtilities.test_data(submach)
    end

    # End to end
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end


@testset "Test serializable of composite machines" begin
    filename = "stack_mach.jls"
    X, y = make_regression(100, 1)
    model = Stack(
        metalearner = DecisionTreeRegressor(),
        tree1 = DecisionTreeRegressor(min_samples_split=3),
        tree2 = DecisionTreeRegressor(),
        measures=rmse)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    # Check serializable function
    smach = MLJBase.serializable(mach)
    TestUtilities.generic_tests(mach, smach)
    # Check data has been wiped out from models at the first level of composition
    @test length(machines(glb(smach))) == length(machines(glb(mach)))
    for submach in machines(glb(smach))
        TestUtilities.test_data(submach)
    end

    # Testing extra report field : it is a deepcopy
    @test report(smach).cv_report === report(mach).cv_report

    @test smach.fitresult isa MLJBase.CompositeFitresult

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    MLJBase.restore!(smach)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))

    rm(filename)

    # End to end
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

@testset "Test serializable of nested composite machines" begin
    filename = "nested_stack_mach.jls"
    X, y = make_regression(100, 1)

    pipe = (X -> coerce(X, :x₁=>Continuous)) |> DecisionTreeRegressor()
    model = @test_logs (:warn, r"") Stack(
        metalearner = DecisionTreeRegressor(),
        pipe = pipe)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    MLJBase.save(filename, mach)
    smach = machine(filename)

    @test predict(smach, X) == predict(mach, X)

    # Test data as been erased at the first and second level of composition
    for submach in machines(glb(smach))
        TestUtilities.test_data(submach)
        if submach isa Machine{<:Composite}
            for subsubmach in machines(glb(submach))
                TestUtilities.test_data(subsubmach)
            end
        end
    end

    rm(filename)

end

struct DummyRangeCV
    cv
end

torange(x::UnitRange) = x
torange(x) = minimum(x):maximum(x)

function MLJBase.train_test_pairs(dcv::DummyRangeCV, rows, X, y)
    ttp = MLJBase.train_test_pairs(dcv.cv, rows)
    return [(torange(t),torange(e)) for (t,e) in ttp]
end

@testset "Test serialized filesize does not increase with datasize" begin
    # At the moment it is necessary to resort to a custom resampling strategy,
    # for this test. This is because partial functions working on nodes,
    # such as `selectrows`in learning networks store data.
    # A basic CV would store vectors which would grow in size as the dataset grows.

    dcv = DummyRangeCV(CV(nfolds=3))
    model = Stack(
        metalearner = FooBarRegressor(lambda=1.),
        resampling = dcv,
        model_1 = DeterministicConstantRegressor(),
        model_2=ConstantRegressor()
    )

    filesizes = []
    for n in [100, 500, 1000]
        filename = "serialized_temp_$n.jls"
        X, y = make_regression(n, 1)
        mach = machine(model, X, y)
        fit!(mach, verbosity=0)
        MLJBase.save(filename, mach)
        push!(filesizes, filesize(filename))
        rm(filename)
    end
    @test all(x==filesizes[1] for x in filesizes)
    # What if no serializable procedure had happened
    filename = "full_of_data.jls"
    X, y = make_regression(1000, 1)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    serialize(filename, mach)
    @test filesize(filename) > filesizes[1]

    @test_logs (:warn, MLJBase.warn_bad_deserialization(mach.state)) machine(filename)

    rm(filename)
end

mutable struct DummyComposite <: DeterministicComposite
    stand1
    stand2
end

@testset "issue 377" begin
    stand = Standardizer()
    model = DummyComposite(stand, stand)

    Xs = source()
    mach1 = machine(model.stand1, Xs)
    X1 = transform(mach1, Xs)
    mach2 = machine(model.stand2, X1)
    X2 = transform(mach2, X1)

    mach = machine(Unsupervised(), Xs; transform=X2)
    @test_logs((:error, r"The hyper"),
               @test_throws(ArgumentError,
                            MLJBase.network_model_names(model, mach)))
end


end # module

true
