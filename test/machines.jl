module TestMachines

using MLJBase
using Test
using Statistics
using ..Models
const MLJModelInterface = MLJBase.MLJModelInterface
using StableRNGs
using ..TestUtilities

@load DecisionTreeRegressor

N=50
X = (a=rand(N), b=rand(N), c=rand(N));
y = 2*X.a - X.c + 0.05*rand(N);

train, test = partition(eachindex(y), 0.7);

tree = DecisionTreeRegressor(max_depth=5)

t = machine(tree, X, y)
@test_logs (:info, r"Training") fit!(t)
@test_logs (:info, r"Training") fit!(t, rows=train)
@test_logs (:info, r"Not retraining") fit!(t, rows=train)
@test_logs (:info, r"Training") fit!(t)
t.model.max_depth = 1
@test_logs (:info, r"Updating") fit!(t)

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

@testset "warnings" begin
    @test_throws DimensionMismatch machine(tree, X, y[1:end-1])
    @test_logs((:warn, r"The scitype of `y`"),
               machine(tree, X, categorical(1:N)))
    @test_logs((:warn, r"The scitype of `X`"),
               machine(tree, (x=categorical(1:N),), y))
end

@testset "weights" begin
    yraw = ["Perry", "Antonia", "Perry", "Skater"]
    X = (x=rand(4),)
    y = categorical(yraw)
    w = [2, 3, 2, 5]

    # without weights:
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach)
    d1 = predict(mach, X)[1]
    d2 = MLJBase.UnivariateFinite([y[1], y[2], y[4]], [0.5, 0.25, 0.25])
    @test all([pdf(d1, c) ≈ pdf(d2, c) for c in MLJBase.classes(d1)])

    # with weights:
    mach = machine(ConstantClassifier(), X, y, w)
    fit!(mach)
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
    Xt = MLJBase.table(X)

    @test_throws ArgumentError machine(Scale(2), X)
    @test_throws ArgumentError machine(Scale(2), source(X))

    mach = machine(Scale(2))
    @test_logs (:info, r"Training") fit!(mach) # no-op
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

@testset "serialization" begin

    @test MLJBase._filename("mymodel.jlso") == "mymodel"
    @test MLJBase._filename("mymodel.gz") == "mymodel"
    @test MLJBase._filename("mymodel") == "mymodel"

    model = @load DecisionTreeRegressor

    X = (a = Float64[98, 53, 93, 67, 90, 68],
         b = Float64[64, 43, 66, 47, 16, 66],)
    Xnew = (a = Float64[82, 49, 16],
            b = Float64[36, 13, 36],)
    y =  [59.1, 28.6, 96.6, 83.3, 59.1, 48.0]

    mach =machine(model, X, y)
    filename = joinpath(@__DIR__, "machine.jlso")
    io = IOBuffer()
    @test_throws Exception MLJBase.save(io, mach; compression=:none)

    fit!(mach)
    report = mach.report
    pred = predict(mach, Xnew)
    MLJBase.save(io, mach; compression=:none)
    # commented out for travis testing:
    #MLJBase.save(filename, mach)

    # test restoring data from filename:
    m = machine(filename)
    p = predict(m, Xnew)
    @test m.model == model
    @test m.report == report
    @test p ≈ pred
    m = machine(filename, X, y)
    fit!(m)
    p = predict(m, Xnew)
    @test p ≈ pred

    # test restoring data from io:
    seekstart(io)
    m = machine(io)
    p = predict(m, Xnew)
    @test m.model == model
    @test m.report == report
    @test p ≈ pred
    seekstart(io)
    m = machine(io, X, y)
    fit!(m)
    p = predict(m, Xnew)
    @test p ≈ pred

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


## BABY SUPERVISED MODEL WITH AN UPDATE METHOD

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

MLJModelInterface.reformat(model::SomeModel, X, y) = (MLJBase.matrix(X), y)
MLJModelInterface.selectrows(model::SomeModel, I, A, y) =
    (view(A, I, :), view(y, I))

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

    clf = @load ConstantClassifier
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

end # module

true
