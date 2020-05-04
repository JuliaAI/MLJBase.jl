module TestMachines

using MLJBase
using Test
using Statistics
using ..Models

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
# MLJBase.recursive_setproperty!(t, :(model.max_depth),  1)
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
    @test_logs((:info, r"does not support"),
               machine(tree, X, y, rand(N)))
    @test_throws DimensionMismatch machine(tree, X, y[1:end-1])
    @test_logs((:info, r"does not support"),
               @test_throws DimensionMismatch machine(tree, X, y, rand(N-1)))
    @test_logs((:info, r"does not support"),
               @test_throws ArgumentError machine(tree, X, y, fill('a', N)))
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
    X = randn(2, 3)
    Xt = MLJBase.table(X)

    @test_throws ArgumentError machine(Scale(2), X)
    @test_throws ArgumentError machine(Scale(2), source(X))

    mach = machine(Scale(2))
    fit!(mach) # no-op
    R  = transform(mach, X)
    IR = inverse_transform(mach, R)
    @test IR ≈ X
end

@testset "serialization" begin
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
    # MLJBase.save(filename, mach)

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

end # module
true
