module TestTransformedTargetModel

using MLJBase
using Test
using ..Models
using StableRNGs
const MMI = MLJBase.MLJModelInterface

rng = StableRNG(698790187)

atom = DeterministicConstantRegressor()
p_atom = ConstantRegressor()
whitener = UnivariateStandardizer()

@testset "constructor and clean!" begin
    model = @test_throws(
        MLJBase.ERR_TRANSFORMER_UNSPECIFIED,
        TransformedTargetModel(atom),
    )
    @test_logs TransformedTargetModel(atom, transformer=UnivariateStandardizer)
    model = @test_logs TransformedTargetModel(atom, transformer=whitener)
    @test model.model == atom
    @test model.inverse == nothing
    @test model.transformer == whitener
    @test model.cache

    model = @test_logs TransformedTargetModel(model=atom, transformer=whitener)
    @test model.model == atom
    @test model.inverse == nothing

    model = @test_logs(
        TransformedTargetModel(atom, transformer=whitener, inverse=identity))
    @test model.model == atom
    @test model.inverse == identity

    @test_throws(MLJBase.ERR_MODEL_UNSPECIFIED,
                 TransformedTargetModel(transformer=whitener))

    @test_throws(MLJBase.ERR_TOO_MANY_ARGUMENTS,
                 TransformedTargetModel(atom, whitener))

    @test_throws(MLJBase.err_tt_unsupported(whitener),
                 TransformedTargetModel(whitener, transformer=whitener))

    model = @test_logs((:warn, MLJBase.WARN_IDENTITY_INVERSE),
                       TransformedTargetModel(p_atom, transformer=whitener))
    @test model.inverse == identity

    model = @test_logs((:warn, MLJBase.WARN_MISSING_INVERSE),
                       TransformedTargetModel(atom, transformer=y->log.(y)))
    @test model.inverse == identity

end

# a function for transforming and it's inverse:
f(y) = sin.(y)
g(y) = asin.(y)

# implemented as a static model:
mutable struct Nonlinear <: Static
    λ::Float64 # ignored
end
Nonlinear(; λ=1) = Nonlinear(λ)
MMI.transform(::Nonlinear, _, y) = f(y)
MMI.inverse_transform(::Nonlinear, _, z) = g(z)

# some data:
X, _ = make_regression(5)
z = rand(rng, 5)
y = asin.(z)
@test sin.(y) ≈ z

# average of y on untransformed scale:
avg = mean(y)

# average of y on the non-linear scale defined by f:
avg_nonlinear = g(mean(f(y))) # = g(mean(z))

@testset begin "fit and predict"

    # Remember that `atom` predicts the constant mean of the training
    # target on all new observations. Let's
    # check it's expected behaviour before wrapping:
    fr, _, _ = MMI.fit(atom, 0, X, y)
    @test predict(atom, fr, X) ≈ fill(avg, 5)

    # Test wrapping using f and g:
    model = TransformedTargetModel(atom, transformer=f, inverse=g)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test first(predict(model, fr1, X)) ≈ fill(avg_nonlinear, 5)

    # Test wrapping using a `Static` transformer:
    model = TransformedTargetModel(atom, transformer=Nonlinear())
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test first(predict(model, fr1, X)) ≈ fill(avg_nonlinear, 5)

    # Test wrapping using a non-static `Unsupervised` model:
    model = TransformedTargetModel(atom, transformer=whitener)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test first(predict(model, fr1, X)) ≈ fill(avg, 5) # whitener is linear

    # Test with `inverse=identity`:
    model = TransformedTargetModel(atom, transformer=Nonlinear(), inverse=identity)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test first(predict(model, fr1, X)) ≈ fill(mean(z), 5)

    # Test a probablistic model:
    model = TransformedTargetModel(p_atom, transformer=whitener, inverse=identity)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    yhat = predict(model, fr1, X) |> first
    @test isapprox(first(yhat).μ, 0, atol=1e-15)
end

MMI.iteration_parameter(::Type{DeterministicConstantRegressor}) = :n

@testset "traits" begin
    model = TransformedTargetModel(atom, transformer=Nonlinear())
    @test input_scitype(model) == input_scitype(atom)
    @test target_scitype(model) == target_scitype(atom)
    @test is_wrapper(model)
    @test iteration_parameter(model) == :(model.n)
    @test package_name(model) == "MLJBase"
    @test occursin("2229d", package_uuid(model))
    @test package_license(model) == "MIT"
    @test package_url(model) == "https://github.com/JuliaAI/MLJBase.jl"
end

@testset "integration 1" begin
    model = TransformedTargetModel(atom, transformer=Nonlinear())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) ≈ fill(avg_nonlinear, 5)
    @test issubset([:model,], keys(fitted_params(mach)))
    @test fitted_params(mach).model.fitresult ≈ mean(z)
end

@testset "integration 2" begin
    model = TransformedTargetModel(atom, transformer=UnivariateBoxCoxTransformer())
    mach = machine(model, X, y)
    fit!(mach, verbosity=2)
    @test predict(mach, X) isa Vector
    @test issubset([:model, :transformer], keys(fitted_params(mach)))
    @test issubset([:λ, :c], keys(fitted_params(mach).transformer))
end

@testset "integration 3" begin
    model = TransformedTargetModel(atom, transformer=v->log.(v), inverse=v->exp.(v))
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) isa Vector
    @test keys(fitted_params(mach)) == (:model,)
end

mutable struct FooModel46 <: Deterministic
    epochs
end

MMI.fit(::FooModel46, verbosity, X, y) =
    nothing, nothing, (training_losses=ones(length(y)),)
MMI.predict(::FooModel46, fitresult, Xnew) = ones(nrows(Xnew))
MMI.supports_training_losses(::Type{<:FooModel46}) = true
MMI.iteration_parameter(::Type{<:FooModel46}) = :epochs
MMI.training_losses(::FooModel46, report) = report.training_losses

X = rand(5)
y = rand(5)

@testset "training_losses" begin
    atom = FooModel46(10)
    model = TransformedTargetModel(atom, transformer=Nonlinear())
    @test supports_training_losses(model)
    @test iteration_parameter(model) == :(model.epochs)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    rep = report(mach)
    @test rep.model.training_losses == ones(5)
    @test training_losses(mach) == ones(5)
end

@testset "feature_importances" begin
    X, y = @load_iris
    atom = DecisionTreeClassifier()
    model = TransformedTargetModel(atom, transformer=identity, inverse=identity)
    @test reports_feature_importances(model)
    fitresult, _, rpt = MMI.fit(model, 0, X, y)
    @test Set(first.(feature_importances(model, fitresult, rpt))) == Set(keys(X))
end

end
true
