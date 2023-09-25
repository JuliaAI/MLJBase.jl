module TestMachines

using MLJBase
using Test
using Statistics
using ..Models
using StableRNGs
using Serialization
using ..TestUtilities
using StatisticalMeasures

const MLJModelInterface = MLJBase.MLJModelInterface
const MMI = MLJModelInterface

verbosity = 0

N=50
X = (a=rand(N), b=rand(N), c=rand(N));
y = 2*X.a - X.c + 0.05*rand(N);

train, test = partition(eachindex(y), 0.7);

tree = DecisionTreeRegressor(max_depth=5)
knn = KNNRegressor()

pca = PCA()

@testset "_contains_unknown" begin
    @test MLJBase._contains_unknown(Unknown)
    @test MLJBase._contains_unknown(Tuple{Unknown})
    @test MLJBase._contains_unknown(Tuple{Unknown, Int})
    @test MLJBase._contains_unknown(Union{Tuple{Unknown}, Tuple{Int,Char}})
    @test MLJBase._contains_unknown(Union{Tuple{Int}, Tuple{Int,Unknown}})
    @test !MLJBase._contains_unknown(Int)
    @test !MLJBase._contains_unknown(Tuple{Int})
    @test !MLJBase._contains_unknown(Tuple{Char, Int})
    @test !MLJBase._contains_unknown(Union{Tuple{Int}, Tuple{Int,Char}})
    @test !MLJBase._contains_unknown(Union{Tuple{Int}, Tuple{Int,Char}})
end

struct StaticYoghurt <: Static end
MLJBase.transform(::StaticYoghurt, _, X) = X

@testset "machine constructor for Static models" begin
end

@testset "machine training and inpection" begin
    t = machine(tree, X, y)

    @test_throws MLJBase.NotTrainedError(t, :fitted_params) fitted_params(t)
    @test_throws MLJBase.NotTrainedError(t, :report) report(t)
    @test_throws MLJBase.NotTrainedError(t, :training_losses) training_losses(t)
    @test_throws MLJBase.NotTrainedError(t, :feature_importances) feature_importances(t)

    @test_logs (:info, r"Training") fit!(t)
    @test_logs (:info, r"Training") fit!(t, rows=train)
    @test_logs (:info, r"Not retraining") fit!(t, rows=train)
    @test_logs (:info, r"Training") fit!(t)
    t.model.max_depth = 1
    @test_logs (:info, r"Updating") fit!(t)

    # The following tests only pass when machine `t` has been fitted
    @test fitted_params(t) == MMI.fitted_params(t.model, t.fitresult)
    @test isnothing(report(t))
    @test training_losses(t) === nothing
    @test feature_importances(t) === nothing

    predict(t, selectrows(X,test));
    @test rms(predict(t, selectrows(X, test)), y[test]) < std(y)

    mach = machine(ConstantRegressor(), X, y)
    @test_logs (:info, r"Training") fit!(mach)
    yhat = predict_mean(mach, X);

    n = nrows(X)
    @test rms(yhat, y) ≈ std(y)*sqrt(1 - 1/n)

    # test an unsupervised univariate case:
    mach = machine(UnivariateStandardizer(), float.(1:5))
    @test_logs (:info, r"Training") fit!(mach)
    @test isempty(params(mach))

    # test a frozen Machine
    stand = machine(Standardizer(), source((x1=rand(10),)))
    freeze!(stand)
    @test_logs (:warn, r"not trained as it is frozen\.$") fit!(stand)
end

@testset "machine instantiation warnings" begin
    @test_throws DimensionMismatch machine(tree, X, y[1:end-1])

    # supervised model with bad target:
    @test_logs((:warn,
                MLJBase.alert_generic_scitype_mismatch(
                    Tuple{scitype(X), AbstractVector{Multiclass{N}}},
                    MLJBase.fit_data_scitype(tree),
                    typeof(tree)
                    )
                ),
               machine(tree, X, categorical(1:N)))

    # ordinary transformer:
    @test_logs((:warn,
                MLJBase.alert_generic_scitype_mismatch(
                    Tuple{scitype(42),},
                    MLJBase.fit_data_scitype(pca),
                    typeof(pca)
                    )
                ),
               machine(pca, 42))
    y2 = coerce(1:N, OrderedFactor);

    # bad weight vector:
    @test_logs((:warn,
                MLJBase.alert_generic_scitype_mismatch(
                    Tuple{scitype(X), scitype(y2), scitype(42)},
                    MLJBase.fit_data_scitype(ConstantClassifier()),
                    ConstantClassifier
                    )
                ),
               machine(ConstantClassifier(), X, y2, 42))
end

struct FooBar <: Model end

MLJBase.fit_data_scitype(::Type{<:FooBar}) =
    Union{Tuple{AbstractVector{Count}},
          Tuple{AbstractVector{Count},AbstractVector{Continuous}}}

struct FooBarUnknown <: Model end

@testset "machine scitype_check_level" begin

    X = [1, 2, 3, 4]
    y = rand(4)

    # with no Unknown scitypes

    model = FooBar()

    for scitype_check_level in [1, 2]
        @test_logs machine(model, X, y; scitype_check_level)
        @test_logs machine(model, X; scitype_check_level)
        @test_logs((:warn,
                    MLJBase.alert_generic_scitype_mismatch(Tuple{scitype(y)},
                                                          fit_data_scitype(model),
                                                          FooBar)),
                   machine(model, y; scitype_check_level))
    end

    scitype_check_level = 3
    @test_logs machine(model, X, y; scitype_check_level)
    @test_logs machine(model, X; scitype_check_level)
    @test_throws(ArgumentError(
        MLJBase.alert_generic_scitype_mismatch(Tuple{scitype(y)},
                                              fit_data_scitype(model),
                                              FooBar)),
                 machine(model, y; scitype_check_level))

    @test default_scitype_check_level() == 1
    default_scitype_check_level(3)
    @test default_scitype_check_level() == 3

    @test_logs machine(model, X, y)
    @test_logs machine(model, X)
    @test_throws(ArgumentError(
        MLJBase.alert_generic_scitype_mismatch(Tuple{scitype(y)},
                                              fit_data_scitype(model),
                                              FooBar)),
                 machine(model, y))
    default_scitype_check_level(1)

    # with Unknown scitypes

    model = FooBarUnknown()

    scitype_check_level = 1
    @test_logs machine(model, X, y; scitype_check_level)
    @test_logs machine(model, X; scitype_check_level)

    warning = MLJBase.WARN_UNKNOWN_SCITYPE
    for scitype_check_level in [2, 3]
        @test_logs (:warn, warning) machine(model, X, y; scitype_check_level)
        @test_logs (:warn, warning) machine(model, X; scitype_check_level)
    end

    scitype_check_level = 4
    @test_throws ArgumentError(warning) machine(model, X, y; scitype_check_level)
    @test_throws ArgumentError(warning) machine(model, X; scitype_check_level)

end

@testset "copy(::Machine) and replace(::Machine, ...)" begin
    mach = machine(tree, X, y)
    clone = copy(mach)
    @test all(fieldnames(typeof(mach))) do field
        field === :fit_okay && return true
        a = !isdefined(mach, field)
        b = !isdefined(mach, field)
        a ⊻ b && return false # xor
        a && b && return true
        getproperty(mach, field) === getproperty(clone, field)
    end
    @test typeof(mach) == typeof(clone)

    fit!(mach, verbosity=0)
    clone = copy(mach)
    @test all(fieldnames(typeof(mach))) do field
        field === :fit_okay && return true
        a = !isdefined(mach, field)
        b = !isdefined(mach, field)
        a ⊻ b && return false
        a && b && return true
        getproperty(mach, field) === getproperty(clone, field)
    end

    mach = machine(tree, X, y)
    s1 = source(42)
    s2 = source(21)
    mach_clone = replace(
        mach,
        :args=>(s1, s2),
        :report=>57,
    )
    @test mach_clone.args == (s1, s2)
    @test mach_clone.report == 57
    @test !isdefined(mach_clone, :fitresult)
    @test mach_clone.model == tree
    @test mach_clone.state == mach.state

    mach_clone = replace(mach, :model=>knn)
    @test mach_clone.model === knn
end

@testset "weights" begin
    yraw = ["Perry", "Antonia", "Perry", "Skater"]
    X = (x=rand(4),)
    y = categorical(yraw)
    w = [2, 3, 2, 5]

    # without weights:
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach, verbosity=0)
    d1 = predict(mach, X)[1]
    d2 = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [0.5, 0.25, 0.25])
    @test all([pdf(d1, c) ≈ pdf(d2, c) for c in MLJBase.classes(d1)])

    # with weights:
    mach = machine(ConstantClassifier(), X, y, w)
    fit!(mach, verbosity=0)
    d1 = predict(mach, X)[1]
    d2 = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [1/3, 1/4, 5/12])
    @test all([pdf(d1, c) ≈ pdf(d2, c) for c in MLJBase.classes(d1)])
end

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

function MLJBase.transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X * s.scaling
    MLJBase.table(s.scaling * MLJBase.matrix(X), prototype=X)
end

function MLJBase.inverse_transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X / s.scaling
    MLJBase.table(MLJBase.matrix(X) / s.scaling, prototype=X)
end

@testset "static transformer machines" begin
    s = Scale(2)
    X = ones(2, 3)

    mach = @test_logs machine(Scale(2))
    @test mach isa Machine{Scale, false}
    transform(mach, X) # triggers training of `mach`, ie is mutating
    @test report(mach) in [nothing, NamedTuple()]
    @test isnothing(fitted_params(mach))
    @test_throws(
        MLJBase.ERR_STATIC_ARGUMENTS,
        machine(Scale(2), X),
    )
    @test_throws(
        MLJBase.ERR_STATIC_ARGUMENTS,
        machine(Scale(2), source(X)),
    )

    @test_logs (:info, r"Not retraining") fit!(mach) # no-op
    state = mach.state


    R  = transform(mach, X)
    IR = inverse_transform(mach, R)
    @test IR ≈ X

    # changing rows does not alter state (and "training" is skipped):
    @test_logs (:info, r"Not retraining") fit!(mach, rows=1:3)
    @test mach.state == state

    # changing hyper-parameters *does* change state (and "training" is
    # not skipped):
    mach.model.scaling = 3.0
    @test_logs (:info, r"Updating") fit!(mach, rows=1:3)
    @test mach.state != state

    @test_throws ArgumentError transform(mach, rows=1:2)
end

mutable struct Box
    matrix::Matrix{Int}
end


## DUMMY UNSUPERVISED MODEL

mutable struct Fozy <: Unsupervised end
MLJBase.fit(model::Fozy, verbosity, X) = minimum(X.matrix), nothing, nothing
MLJBase.transform(model::Fozy, fitresult, newbox) =
    fill(fitresult, nrows(newbox.matrix))
MLJBase.MLJModelInterface.reformat(model::Fozy, user_data) =
    (Box(MLJBase.matrix(user_data)),)
MLJBase.selectrows(model::Fozy, I, X...) = (Box(X[1].matrix[I,:]),)


## BABY SUPERVISED MODEL WITH AN UPDATE METHOD AND FEATURE IMPORTANCE

mutable struct SomeModel <: Deterministic
    n::Int
end

function MLJModelInterface.fit(model::SomeModel,
                               verbosity,
                               A,
                               y)
    n = model.n
    cache = (A \ y)'  # all coefficients
    n_features = length(cache)

    # kill all but first n coefficients:
    fitresult = vcat(cache[1:n], fill(0.0, n_features - n))

    report = (n_features=n_features, )
    return fitresult, cache, report
end

function MLJModelInterface.update(model::SomeModel,
                                  verbosity,
                                  old_fitresult,
                                  old_cache,
                                  A,  # ignored in this case
                                  y)  # ignored in this case
    n = model.n
    cache = old_cache # coefficients already obtained in `fit`
    n_features = length(cache)

    # kill all but first n coefficients:
    fitresult = vcat(cache[1:n], fill(0.0, n_features - n))

    report = (n_features=n_features, )
    return fitresult, cache, report
end

function MLJModelInterface.predict(::SomeModel, fitresult, Xnew)
    Anew = MLJBase.matrix(Xnew)
    return Anew*fitresult
end

const dummy_importances = [:x1 => 1.0, ]
MLJModelInterface.reports_feature_importances(::Type{<:SomeModel}) = true
MLJModelInterface.feature_importances(::SomeModel, fitresult, report) = dummy_importances
MLJModelInterface.supports_training_losses(::Type{<:SomeModel}) = true
MLJModelInterface.training_losses(::SomeModel, report) = 1:report.n_features

MLJModelInterface.reformat(model::SomeModel, X, y) = (MLJBase.matrix(X), y)
MLJModelInterface.selectrows(model::SomeModel, I, A, y) =
    (view(A, I, :), view(y, I))

@testset "feature_importances" begin
    mach = machine(SomeModel(1), make_regression(10)...)
    fit!(mach, verbosity=0)
    @test feature_importances(mach) == dummy_importances

    mach = machine(KNNClassifier(), make_blobs(10)...)
    fit!(mach, verbosity=0)
    @test isnothing(feature_importances(mach))
end

@testset "overloading reformat(::Model, ...), selectrows(::Model, ...)" begin

    # dummy unsupervised model:
    model = Fozy()
    args = ((x1=[10, 30, 50], x2 = [20, 40, 60]),)
    data = MLJBase.MLJModelInterface.reformat(model, args...)
    @test data[1] isa Box && data[1].matrix == [10 20; 30 40; 50 60]
    @test selectrows(model, 2:3, data...)[1].matrix == [30 40; 50 60]
    @test fit(model, 1, data...)[1] == 10
    mach = machine(model, args...)
    @test_logs (:info, r"Training") fit!(mach, rows=2:3);
    @test transform(mach, (x1 = 1:4, x2 = 1:4)) == [30, 30, 30, 30]

    # supervised model with an update method:
    rng = StableRNGs.StableRNG(123)
    A = rand(rng, 8, 3)
    y = A*[1, 2, 3]
    X =  MLJBase.table(A)
    model = SomeModel(1)
    mach = machine(model, X, y)
    @test_mach_sequence fit!(mach, rows=1:4) [(:train, mach),]
    Xnew = selectrows(X, 1:4)
    @test predict(mach, Xnew) ≈ A[1:4,1]
    # mutate the model to trigger `update` call:
    model.n=3
    @test_mach_sequence fit!(mach, rows=1:4) [(:update, mach), ]
    @test predict(mach, Xnew) ≈ y[1:4]
    # change the rows to be sampled:
    @test_mach_sequence fit!(mach) [(:train, mach),]
    @test predict(mach, Xnew) ≈ y[1:4]
end


@testset "fit! for models with reformat front-end" begin
    X = (x1=ones(5), x2=2*ones(5))
    y = categorical(collect("abaaa"))

    clf = ConstantClassifier()
    clf = ConstantClassifier(testing=true)
    mach = machine(clf, X, y, cache=true)
    # first call to fit reformats data and resamples data:
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(mach, rows=1:3, verbosity=0))
    @test mach.data == (MLJBase.matrix(X), y)
    @test mach.resampled_data[1] == mach.data[1][1:3,:]
    @test mach.resampled_data[2] == y[1:3]
    yhat = @test_logs (:info, r"reformatting X") predict_mode(mach, X)
    @test yhat == fill('a', 5)
    yhat = @test_logs (:info, "resampling X") predict_mode(mach, rows=1:2)
    @test yhat == fill('a', 2)
    # calling fit! with new `rows` triggers resampling but no
    # reformatting:
    @test_logs((:info, "resampling X, y"),
               fit!(mach, rows=1:2, verbosity=0))

end

@testset "Test serializable method of Supervised Machine" begin
    X, y = make_regression(100, 1)
    filename = "decisiontree.jls"

    # test error for untrained machines:
    mach = machine(DecisionTreeRegressor(), X, y)
        @test_throws(
        MLJBase.ERR_SERIALIZING_UNTRAINED,
        serializable(mach),
    )
    fit!(mach, verbosity=0)

    # Check serializable function
    smach = MLJBase.serializable(mach)
    @test smach === MLJBase.serializable(smach)  # check no-op if repeated
    @test smach.report == mach.report
    @test smach.fitresult == mach.fitresult
    @test_throws(ArgumentError, predict(smach))
    @test_logs (:warn, MLJBase.WARN_SERIALIZABLE_MACH) predict(smach, X)

    TestUtilities.generic_tests(mach, smach)
    # Check restore! function
    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    MLJBase.restore!(smach)

    @test smach.state == 1
    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach) == report(mach)

    # repeated `restore!` makes no difference:
    MLJBase.restore!(smach)
    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)

    rm(filename)

    # End to end save and reload
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test smach.state == 1
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

@testset "Test serializable method of Unsupervised Machine" begin
    X, _ = make_regression(100, 1)
    filename = "standardizer.jls"
    mach = machine(Standardizer(), X)
    fit!(mach, verbosity=0)

    MLJBase.save(filename, mach)
    smach = machine(filename)

    @test transform(mach, X) == transform(smach, X)
    @test_throws(ArgumentError, transform(smach))

    # warning on non-restored machine
    smach = deserialize(filename)
    @test_logs (:warn, MLJBase.WARN_SERIALIZABLE_MACH) transform(smach, X)

    rm(filename)
end

struct ReportingDynamic <: Unsupervised end
MLJBase.fit(::ReportingDynamic, _, X) = nothing, 16, NamedTuple()
MLJBase.transform(::ReportingDynamic,_, X) = (X, (news=42,))
MLJBase.reporting_operations(::Type{<:ReportingDynamic}) = (:transform, )

@testset "corner case for operation applied to a reporting machinw" begin
    model = ReportingDynamic()
    mach = fit!(machine(model, [1,2,3]), verbosity=0)
    @test transform(mach, rows=:) == [1, 2, 3]
    @test transform(mach, rows=1:2) == [1, 2]
end

@testset "machines with symbolic model placeholders" begin

    struct CherryComposite
        rgs
    end

    composite = CherryComposite(SomeModel(1))
    X = MLJBase.table(
        [1.0 3.0
         2.0 4.0]
    )
    y = [1.0, 2.0]
    mach  = machine(:rgs, X, y)
    @test isnothing(last_model(mach))
    fit!(mach; composite, verbosity=0)
    @test last_model(mach) == composite.rgs

    @test fitted_params(mach).fitresult ≈ [1.0, 0.0]
    @test report(mach) == (; n_features = 2)
    @test training_losses(mach) == 1:2
    @test feature_importances(mach) == dummy_importances

    Xnew = MLJBase.table(rand(2, 2))
    @test size(predict(mach, Xnew)) == (2,)

    # if composite.rgs is changed, but keeps same type, get an update:
    composite = CherryComposite(SomeModel(2))
    @test_logs (:info, r"Updating") fit!(mach; composite)

    # if composite.rgs is changed, but type changes, retrain from scratch:
    composite = CherryComposite(StaticYoghurt())
    @test_logs (:info, r"Training") fit!(mach; composite)

    # no training arguments:
    composite = CherryComposite(StaticYoghurt())
    mach = machine(:rgs)
    @test isnothing(last_model(mach))
    @test_throws MLJBase.err_no_real_model(mach) transform(mach, X)
    fit!(mach; composite, verbosity=0)
    @test transform(mach, X) == X
end

end # module

true
