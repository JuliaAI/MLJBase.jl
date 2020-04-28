module TestArrows

using MLJBase
using ..Models
using Test
using Random

@testset "|> syntax for pipelines" begin
    Random.seed!(142)
    @load RidgeRegressor pkg="MultivariateStats"
    @load KNNRegressor pkg="NearestNeighbors"
    X = MLJBase.table(randn(500, 5))
    y = abs.(randn(500))
    train, test = partition(eachindex(y), 0.7)

    # Feeding data directly to a supervised model
    knn = KNNRegressor(K=10)
    ŷ   = (X, y) |> knn
    fit!(ŷ, rows=train)

    # Describing a full pipeline using |> syntax.
    Xs, ys = source.((X, y))

    # "first layer"
    W = Xs |> Standardizer()
    z = ys |> UnivariateBoxCoxTransformer()
    # "second layer"
    ẑ = (W, z) |> RidgeRegressor(lambda=0.1)
    # "output layer"
    ŷ = ẑ |> inverse_transform(z)

    fit!(ŷ, rows=train)

    @test isapprox(rms(ŷ(rows=test), ys(rows=test)), 0.627123, atol=0.07)

    # shortcut to get and set hyperparameters of a node
    ẑ[:lambda] = 5.0
    fit!(ŷ, rows=train)
    @test isapprox(rms(ŷ(rows=test), ys(rows=test)), 0.62699, atol=0.07)
end

@testset "Auto-source" begin
    @load PCA
    @load RidgeRegressor pkg="MultivariateStats"
    Random.seed!(5615151)

    X = MLJBase.table(randn(500, 5))
    y = abs.(randn(500))

    pca = X |> Standardizer() |> PCA(maxoutdim=2)
    fit!(pca)

    W = pca()
    sch = schema(W)
    @test sch.names == (:x1, :x2)
    @test sch.scitypes == (Continuous, Continuous)
    @test sch.nrows == 500

    pipe = (pca, y) |> RidgeRegressor()
    fit!(pipe)

    ŷ = pipe()
    @test ŷ isa Vector{Float64}
    @test length(ŷ) == 500
end

@testset "Auto-table" begin
    @load PCA
    @load RidgeRegressor pkg="MultivariateStats"
    Random.seed!(5615151)

    X = randn(500, 5)
    y = abs.(randn(500))

    pca = X |> Standardizer() |> PCA(maxoutdim=2)
    pipe = (pca, y) |> RidgeRegressor()
    fit!(pipe)

    ŷ = pipe()
    @test ŷ isa Vector{Float64}
    @test length(ŷ) == 500
end

@testset "Stacking" begin
    @load PCA
    @load RidgeRegressor pkg=MultivariateStats
    @load DecisionTreeRegressor pkg=DecisionTree
    Random.seed!(5615151)

    X = randn(500, 5)
    y = abs.(randn(500))

    W = X |> Standardizer() |> PCA(maxoutdim=3)
    z = y |> UnivariateBoxCoxTransformer()
    ẑ₁ = (W, z) |> RidgeRegressor()
    ẑ₂ = (W, z) |> DecisionTreeRegressor()
    R = hcat(ẑ₁, ẑ₂)
    ẑ = (R, z) |> DecisionTreeRegressor()
    ŷ = ẑ |> inverse_transform(z)

    fit!(ŷ)

    p̂ = ŷ()
    @test p̂ isa Vector{Float64}
    @test length(p̂) == 500
end


@testset "functions and static transfomers" begin
    x1 = rand(30)
    x2 = rand(30)
    x3 = rand(30)
    y = exp.(x1 - x2 -2x3 + 0.1*rand(30))
    X = (x1=x1, x2=x2, x3=x3)

    f(X) = (a=selectcols(X, :x1), b=selectcols(X, :x2))

    knn = @load KNNRegressor

    Xs = source(X)
    ys = source(y, kind=:target)

    f(X::AbstractNode) = node(f, X)
    W = Xs |> f |> Standardizer()
    z = ys |> UnivariateBoxCoxTransformer()
    zhat = (W, z) |> knn
    yhat = zhat |> inverse_transform(z)
    fit!(yhat)
    pred = yhat()

    mutable struct MyTransformer <: Static
        ftr::Symbol
    end
    MLJBase.transform(transf::MyTransformer, verbosity, X) =
        (a=selectcols(X, transf.ftr), b=selectcols(X, :x2))
    inserter = MyTransformer(:x1)
    W = Xs |> inserter |> Standardizer()
    z = ys |> UnivariateBoxCoxTransformer()
    zhat = (W, z) |> knn
    yhat = zhat |> inverse_transform(z)
    fit!(yhat)
    @test yhat() ≈ pred
end

end
true
